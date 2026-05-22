"""Coval OpenTelemetry tracing for the Vapi healthcare webhook.

This service receives PSTN/Vapi webhooks, so Coval simulation IDs are correlated
via `/register-simulation`. Spans are buffered per call and exported once the
call closes and a simulation ID is known.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, Status, StatusCode

logger = logging.getLogger(__name__)

COVAL_TRACES_ENDPOINT = os.environ.get("COVAL_TRACES_ENDPOINT", "https://api.coval.dev/v1/traces")
SERVICE = os.environ.get("COVAL_SERVICE_NAME", "coval-vapi-healthcare-agent")
CORRELATION_TTL_SECONDS = int(os.environ.get("COVAL_CORRELATION_TTL_SECONDS", "900"))
MAX_PENDING_SIMULATIONS = int(os.environ.get("COVAL_MAX_PENDING_SIMULATIONS", "25"))
CLOSED_CLAIM_WINDOW_SECONDS = int(os.environ.get("COVAL_CLOSED_CLAIM_WINDOW_SECONDS", "60"))
SENSITIVE_ARGUMENT_KEYS = {
    "account_number",
    "card_last_four",
    "date_of_birth",
    "last_four_ssn",
    "medication_name",
    "name",
    "patient_id",
    "phone",
    "symptoms",
}


@dataclass
class PendingSimulation:
    simulation_id: str
    run_id: str | None
    registered_at: float = field(default_factory=time.time)


@dataclass
class CallTraceState:
    call_id: str
    created_at: float = field(default_factory=time.time)
    closed_at: float | None = None
    simulation_id: str | None = None
    run_id: str | None = None
    closed: bool = False
    exported: bool = False
    export_error: str | None = None
    tool_call_count: int = 0
    tool_failure_count: int = 0
    dependency_blocked: int = 0
    fallback_used: int = 0
    event_counts: dict[str, int] = field(default_factory=dict)
    provider: TracerProvider | None = None
    in_memory_exporter: InMemorySpanExporter | None = None
    root_span: Any = None


_pending_simulations: deque[PendingSimulation] = deque()
_calls: dict[str, CallTraceState] = {}


def _now_ns() -> int:
    return time.time_ns()


def _parse_timestamp_ns(value: Any) -> int | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return int(datetime.fromisoformat(normalized).timestamp() * 1_000_000_000)
    except ValueError:
        return None


def _set_string_attr(span: Any, key: str, value: Any, limit: int = 500) -> None:
    if value is not None and value != "":
        span.set_attribute(key, str(value)[:limit])


def _set_bool_attr(span: Any, key: str, value: Any) -> None:
    if value is not None:
        span.set_attribute(key, bool(value))


def _set_number_attr(span: Any, key: str, value: Any) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        span.set_attribute(key, float(value))
        return
    if isinstance(value, str) and value:
        try:
            span.set_attribute(key, float(value))
        except ValueError:
            return


def _new_trace_state(call_id: str) -> CallTraceState:
    resource = Resource.create(
        {
            SERVICE_NAME: SERVICE,
            "service.namespace": "coval-test-agents",
            "agent.provider": "vapi",
            "agent.name": "Brontemoor Medical Nurse Bronancy",
            "coval.agent_type": "inbound_pstn_voice",
        }
    )
    in_memory_exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(in_memory_exporter))
    tracer = provider.get_tracer(SERVICE)
    root_span = tracer.start_span(
        "conversation",
        kind=SpanKind.SERVER,
        start_time=_now_ns(),
    )
    root_span.set_attribute("session.id", call_id)
    root_span.set_attribute("customer.workflow", "healthcare")
    root_span.set_attribute("coval.correlation.method", "pstn_registration_fifo")
    root_span.set_attribute("telephony.provider", "vapi")
    return CallTraceState(
        call_id=call_id,
        provider=provider,
        in_memory_exporter=in_memory_exporter,
        root_span=root_span,
    )


def _cleanup_stale() -> None:
    cutoff = time.time() - CORRELATION_TTL_SECONDS
    while _pending_simulations and _pending_simulations[0].registered_at < cutoff:
        dropped = _pending_simulations.popleft()
        logger.warning("Dropped stale Coval simulation registration: simulation_id=%s", dropped.simulation_id)

    for call_id, state in list(_calls.items()):
        if state.exported and state.created_at < cutoff:
            _calls.pop(call_id, None)
        elif not state.exported and state.closed and state.created_at < cutoff:
            logger.warning("Dropping unexported stale trace buffer: call=%s", call_id)
            _calls.pop(call_id, None)


def _claim_pending_simulation(state: CallTraceState) -> bool:
    if state.simulation_id or not _pending_simulations:
        return False
    pending = _pending_simulations.popleft()
    state.simulation_id = pending.simulation_id
    state.run_id = pending.run_id
    state.root_span.set_attribute("coval.simulation_id", pending.simulation_id)
    if pending.run_id:
        state.root_span.set_attribute("coval.run_id", pending.run_id)
    state.root_span.add_event("simulation_id_received", {"coval.simulation_id": pending.simulation_id})
    logger.info("Activated Coval trace export: call=%s simulation_id=%s", state.call_id, pending.simulation_id)
    return True


def _get_call(call_id: str) -> CallTraceState:
    _cleanup_stale()
    safe_call_id = call_id or f"missing-call-id-{len(_calls) + 1}"
    state = _calls.get(safe_call_id)
    if state is None:
        state = _new_trace_state(safe_call_id)
        _calls[safe_call_id] = state
    _claim_pending_simulation(state)
    return state


def _find_claimable_call() -> CallTraceState | None:
    open_candidates = [
        state
        for state in _calls.values()
        if not state.closed and not state.simulation_id and not state.exported
    ]
    if open_candidates:
        return max(open_candidates, key=lambda item: item.created_at)

    now = time.time()
    closed_candidates = [
        state
        for state in _calls.values()
        if state.closed
        and not state.simulation_id
        and not state.exported
        and state.closed_at is not None
        and (now - state.closed_at) <= CLOSED_CLAIM_WINDOW_SECONDS
    ]
    if closed_candidates:
        return max(closed_candidates, key=lambda item: item.closed_at or 0.0)
    return None


def _safe_arguments(args: dict[str, Any]) -> str:
    safe: dict[str, Any] = {}
    for key, value in args.items():
        if key in SENSITIVE_ARGUMENT_KEYS:
            safe[key] = "<redacted>"
        elif isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = type(value).__name__
    return json.dumps(safe, sort_keys=True)[:500]


def _start_child_span(
    state: CallTraceState,
    name: str,
    *,
    start_ns: int | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
) -> Any:
    parent_context = trace.set_span_in_context(state.root_span)
    tracer = state.provider.get_tracer(SERVICE) if state.provider else trace.get_tracer(SERVICE)
    return tracer.start_span(
        name,
        context=parent_context,
        kind=kind,
        start_time=start_ns or _now_ns(),
    )


def _message_timing(
    state: CallTraceState,
    message: dict[str, Any],
    fallback_latency_ms: float | None = None,
) -> tuple[int, int]:
    end_ns = _now_ns()
    for key in ("timestamp", "createdAt", "time"):
        parsed = _parse_timestamp_ns(message.get(key))
        if parsed:
            end_ns = parsed
            break
    if fallback_latency_ms is None:
        return end_ns, end_ns
    start_ns = max(end_ns - int(fallback_latency_ms * 1_000_000), state.root_span.start_time or 0)
    return start_ns, end_ns


def _extract_messages(end_report: dict[str, Any]) -> list[dict[str, Any]]:
    artifact = end_report.get("artifact") if isinstance(end_report.get("artifact"), dict) else {}
    candidates = [
        end_report.get("messages"),
        artifact.get("messages"),
        artifact.get("conversation"),
    ]
    for candidate in candidates:
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, dict)]
    return []


def _message_text(message: dict[str, Any]) -> str:
    for key in ("message", "text", "content", "transcript"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:1000]
    return ""


def _message_role(message: dict[str, Any]) -> str:
    role = str(message.get("role") or message.get("speaker") or message.get("type") or "unknown")
    if role.lower() == "bot":
        return "assistant"
    return role.lower().replace("-", "_")[:80]


def _call_bounds_ns(end_report: dict[str, Any]) -> tuple[int, int]:
    call = end_report.get("call") if isinstance(end_report.get("call"), dict) else {}
    started_at = (
        end_report.get("startedAt")
        or call.get("startedAt")
        or call.get("createdAt")
        or end_report.get("phoneCallProviderStartedAt")
    )
    ended_at = end_report.get("endedAt") or call.get("endedAt")
    end_ns = _parse_timestamp_ns(ended_at) or _now_ns()
    start_ns = _parse_timestamp_ns(started_at)

    duration_seconds = end_report.get("durationSeconds") or call.get("durationSeconds") or call.get("duration")
    if start_ns is None and isinstance(duration_seconds, (int, float)):
        start_ns = end_ns - int(float(duration_seconds) * 1_000_000_000)
    if start_ns is None:
        start_ns = end_ns - 1_000_000_000
    if end_ns <= start_ns:
        end_ns = start_ns + 1_000_000
    return start_ns, end_ns


def _coerce_message_time_ns(message: dict[str, Any], call_start_ns: int) -> int | None:
    absolute = _parse_timestamp_ns(message.get("time") or message.get("startTime") or message.get("timestamp"))
    if absolute:
        return absolute

    raw_time = message.get("time") or message.get("startTime")
    if isinstance(raw_time, (int, float)) and not isinstance(raw_time, bool):
        value = float(raw_time)
        if value > 10_000_000_000:
            return int(value * 1_000_000 if value < 10_000_000_000_000 else value)
        return call_start_ns + int(value * 1_000_000_000)

    seconds_from_start = message.get("secondsFromStart")
    if isinstance(seconds_from_start, (int, float)) and not isinstance(seconds_from_start, bool):
        return call_start_ns + int(float(seconds_from_start) * 1_000_000_000)
    return None


def _message_duration_ns(message: dict[str, Any]) -> int | None:
    duration = message.get("duration")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool):
        return None
    value = float(duration)
    if value > 1000:
        return int(value * 1_000_000)
    return int(value * 1_000_000_000)


def _message_window_ns(
    message: dict[str, Any],
    next_message: dict[str, Any] | None,
    call_start_ns: int,
    call_end_ns: int,
    fallback_start_ns: int,
) -> tuple[int, int]:
    start_ns = _coerce_message_time_ns(message, call_start_ns) or fallback_start_ns

    end_ns = _parse_timestamp_ns(message.get("endTime"))
    if end_ns is None:
        raw_end = message.get("endTime")
        if isinstance(raw_end, (int, float)) and not isinstance(raw_end, bool):
            value = float(raw_end)
            if value > 10_000_000_000:
                end_ns = int(value * 1_000_000 if value < 10_000_000_000_000 else value)
            else:
                end_ns = call_start_ns + int(value * 1_000_000_000)
    if end_ns is None:
        duration_ns = _message_duration_ns(message)
        if duration_ns:
            end_ns = start_ns + duration_ns
    if end_ns is None and next_message:
        end_ns = _coerce_message_time_ns(next_message, call_start_ns)
    if end_ns is None:
        end_ns = min(start_ns + 500_000_000, call_end_ns)
    if end_ns <= start_ns:
        end_ns = start_ns + 1_000_000
    return start_ns, min(end_ns, max(call_end_ns, end_ns))


def _metadata_marker_window_ns(start_ns: int, end_ns: int, *, at_end: bool = False) -> tuple[int, int]:
    duration_ns = 1_000_000
    if at_end:
        marker_end = max(end_ns, start_ns + 1)
        marker_start = max(start_ns, marker_end - duration_ns)
        return marker_start, marker_end
    marker_end = min(start_ns + duration_ns, max(end_ns, start_ns + 1))
    return start_ns, marker_end


def _count_transcript_roles(messages: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in messages:
        role = _message_role(item)
        counts[role] = counts.get(role, 0) + 1
    return counts


def _extract_end_report_text(end_report: dict[str, Any]) -> dict[str, Any]:
    artifact = end_report.get("artifact") if isinstance(end_report.get("artifact"), dict) else {}
    analysis = end_report.get("analysis") if isinstance(end_report.get("analysis"), dict) else {}
    return {
        "summary": end_report.get("summary") or analysis.get("summary") or artifact.get("summary"),
        "transcript": end_report.get("transcript") or artifact.get("transcript"),
        "recording_url": end_report.get("recordingUrl") or artifact.get("recordingUrl"),
        "stereo_recording_url": end_report.get("stereoRecordingUrl") or artifact.get("stereoRecordingUrl"),
    }


def _set_call_report_attributes(state: CallTraceState, end_report: dict[str, Any]) -> None:
    call = end_report.get("call") if isinstance(end_report.get("call"), dict) else {}
    analysis = end_report.get("analysis") if isinstance(end_report.get("analysis"), dict) else {}
    text_fields = _extract_end_report_text(end_report)
    messages = _extract_messages(end_report)
    role_counts = _count_transcript_roles(messages)

    _set_string_attr(state.root_span, "vapi.call.id", call.get("id") or state.call_id, 200)
    _set_string_attr(state.root_span, "vapi.call.status", call.get("status"), 100)
    _set_string_attr(state.root_span, "vapi.ended_reason", end_report.get("endedReason") or call.get("endedReason"), 200)
    _set_string_attr(state.root_span, "conversation.summary", text_fields["summary"], 1000)
    _set_bool_attr(state.root_span, "call.recording_url_present", bool(text_fields["recording_url"]))
    _set_bool_attr(state.root_span, "call.stereo_recording_url_present", bool(text_fields["stereo_recording_url"]))
    _set_number_attr(state.root_span, "call.cost_usd", end_report.get("cost") or call.get("cost"))
    _set_number_attr(state.root_span, "vapi.analysis.success_evaluation", analysis.get("successEvaluation"))
    _set_string_attr(state.root_span, "vapi.analysis.structured_data", analysis.get("structuredData"), 500)

    if text_fields["transcript"]:
        state.root_span.set_attribute("transcript.length_chars", len(str(text_fields["transcript"])))
    if messages:
        state.root_span.set_attribute("transcript.turn.count", len(messages))
        for role, count in sorted(role_counts.items()):
            state.root_span.set_attribute(f"transcript.role.{role}.count", count)

    ended_at_ns = _parse_timestamp_ns(end_report.get("endedAt") or call.get("endedAt"))
    started_at_ns = _parse_timestamp_ns(end_report.get("startedAt") or call.get("startedAt"))
    if started_at_ns and ended_at_ns and ended_at_ns > started_at_ns:
        state.root_span.set_attribute("call.duration_seconds", (ended_at_ns - started_at_ns) / 1_000_000_000)


def _record_end_report_span(state: CallTraceState, end_report: dict[str, Any]) -> None:
    messages = _extract_messages(end_report)
    text_fields = _extract_end_report_text(end_report)
    start_ns = _parse_timestamp_ns(end_report.get("endedAt")) or _now_ns()
    span = _start_child_span(state, "vapi.end_of_call_report", start_ns=start_ns, kind=SpanKind.SERVER)
    span.set_attribute("vapi.message.type", "end-of-call-report")
    span.set_attribute("transcript.turn.count", len(messages))
    if text_fields["transcript"]:
        span.set_attribute("transcript.length_chars", len(str(text_fields["transcript"])))
    if text_fields["summary"]:
        span.set_attribute("conversation.summary.length_chars", len(str(text_fields["summary"])))
    span.set_status(Status(StatusCode.OK))
    span.end(end_time=start_ns)


def _record_vapi_artifact_spans(state: CallTraceState, end_report: dict[str, Any]) -> None:
    messages = _extract_messages(end_report)
    usable = [message for message in messages if _message_text(message)]
    if not usable:
        return

    call_start_ns, call_end_ns = _call_bounds_ns(end_report)
    total = max(call_end_ns - call_start_ns, 1)
    fallback_step = max(total // (len(usable) + 1), 1_000_000)

    turn_index = 0
    for index, message in enumerate(usable):
        next_message = usable[index + 1] if index + 1 < len(usable) else None
        fallback_start_ns = call_start_ns + fallback_step * (index + 1)
        start_ns, end_ns = _message_window_ns(message, next_message, call_start_ns, call_end_ns, fallback_start_ns)
        role = _message_role(message)
        text = _message_text(message)
        duration_seconds = max((end_ns - start_ns) / 1_000_000_000, 0.0)

        if role == "user":
            turn_span = _start_child_span(state, "turn.user", start_ns=start_ns)
            turn_span.set_attribute("turn.index", turn_index)
            turn_span.set_attribute("turn.role", role)
            turn_span.set_attribute("turn.text_length", len(text))
            turn_span.set_attribute("turn.text", text[:200])
            turn_span.set_attribute("turn.duration_seconds", duration_seconds)
            turn_span.set_attribute("trace.source", "vapi_artifact")
            turn_span.set_attribute("trace.timing", "vapi_message_window")
            turn_span.set_status(Status(StatusCode.OK))
            turn_span.end(end_time=end_ns)
            turn_index += 1

            marker_start_ns, marker_end_ns = _metadata_marker_window_ns(start_ns, end_ns)
            span = _start_child_span(state, "stt", start_ns=marker_start_ns)
            span.set_attribute("transcript", text)
            span.set_attribute("artifact.message_duration_seconds", duration_seconds)
            span.set_attribute("trace.source", "vapi_artifact")
            span.set_attribute("trace.timing", "metadata_marker")
            span.set_attribute("trace.duration_note", "marker only; Vapi artifact does not expose provider STT latency")
            span.set_status(Status(StatusCode.OK))
            span.end(end_time=marker_end_ns)
        elif role == "assistant":
            turn_span = _start_child_span(state, "turn.assistant", start_ns=start_ns)
            turn_span.set_attribute("turn.index", turn_index)
            turn_span.set_attribute("turn.role", role)
            turn_span.set_attribute("turn.text_length", len(text))
            turn_span.set_attribute("turn.text", text[:200])
            turn_span.set_attribute("turn.duration_seconds", duration_seconds)
            turn_span.set_attribute("trace.source", "vapi_artifact")
            turn_span.set_attribute("trace.timing", "vapi_message_window")
            turn_span.set_status(Status(StatusCode.OK))
            turn_span.end(end_time=end_ns)
            turn_index += 1

            llm_attrs: dict[str, Any] = {
                "response.length": len(text),
                "trace.source": "vapi_artifact",
                "trace.timing": "metadata_marker",
                "trace.duration_note": "marker only; Vapi artifact does not expose provider LLM latency",
            }
            for source_key, attr_key in (
                ("promptTokens", "gen_ai.usage.input_tokens"),
                ("inputTokens", "gen_ai.usage.input_tokens"),
                ("completionTokens", "gen_ai.usage.output_tokens"),
                ("outputTokens", "gen_ai.usage.output_tokens"),
            ):
                value = message.get(source_key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    llm_attrs[attr_key] = int(value)
            if message.get("model"):
                llm_attrs["llm.model"] = str(message["model"])[:200]
            if message.get("finishReason"):
                llm_attrs["llm.finish_reason"] = str(message["finishReason"])[:200]
            if isinstance(message.get("ttfb"), (int, float)):
                llm_attrs["metrics.ttfb"] = float(message["ttfb"])

            llm_start_ns, llm_end_ns = _metadata_marker_window_ns(start_ns, end_ns)
            llm_span = _start_child_span(state, "llm", start_ns=llm_start_ns)
            for key, value in llm_attrs.items():
                llm_span.set_attribute(key, value)
            llm_span.set_status(Status(StatusCode.OK))
            llm_span.end(end_time=llm_end_ns)

            tts_start_ns, tts_end_ns = _metadata_marker_window_ns(start_ns, end_ns, at_end=True)
            tts_span = _start_child_span(state, "tts", start_ns=tts_start_ns)
            tts_span.set_attribute("tts.text_length", len(text))
            tts_span.set_attribute("artifact.message_duration_seconds", duration_seconds)
            tts_span.set_attribute("trace.source", "vapi_artifact")
            tts_span.set_attribute("trace.timing", "metadata_marker")
            tts_span.set_attribute("trace.duration_note", "marker only; Vapi artifact does not expose provider TTS latency")
            tts_span.set_status(Status(StatusCode.OK))
            tts_span.end(end_time=tts_end_ns)


def _parse_tool_result(result: str) -> dict[str, Any]:
    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return {"tool_error": 0, "result_count": 1}

    if not isinstance(parsed, dict):
        return {"tool_error": 0, "result_count": 1}

    error_type = parsed.get("error")
    list_lengths = [
        len(value)
        for value in parsed.values()
        if isinstance(value, list)
    ]
    result_count = max(list_lengths, default=1)
    return {
        "error_type": str(error_type) if error_type else "",
        "tool_error": 1 if error_type else 0,
        "dependency_unavailable": 1 if error_type in {"SERVICE_UNAVAILABLE", "TIMEOUT"} else 0,
        "fallback_used": 1 if parsed.get("hint") == "stt_mistranscription_likely" else 0,
        "result_count": result_count,
    }


def registration_secret_configured() -> bool:
    return bool(os.environ.get("COVAL_TRACE_REGISTRATION_SECRET") or os.environ.get("COVAL_API_KEY"))


def registration_authorized(headers: dict[str, str]) -> bool:
    expected = os.environ.get("COVAL_TRACE_REGISTRATION_SECRET") or os.environ.get("COVAL_API_KEY")
    if not expected:
        return False

    auth_header = headers.get("authorization", "")
    bearer = auth_header.removeprefix("Bearer ").strip() if auth_header.startswith("Bearer ") else ""
    candidates = [
        headers.get("x-coval-registration-secret", ""),
        headers.get("x-api-key", ""),
        bearer,
    ]
    return any(candidate and hmac.compare_digest(candidate, expected) for candidate in candidates)


def register_simulation(simulation_id: str, run_id: str | None = None) -> dict[str, Any]:
    _cleanup_stale()
    if not simulation_id or not isinstance(simulation_id, str):
        raise ValueError("simulation_id is required")

    state = _find_claimable_call()
    if state:
        state.simulation_id = simulation_id
        state.run_id = run_id
        state.root_span.set_attribute("coval.simulation_id", simulation_id)
        if run_id:
            state.root_span.set_attribute("coval.run_id", run_id)
        state.root_span.add_event("simulation_id_received", {"coval.simulation_id": simulation_id})
        logger.info("Registered Coval simulation to active call: call=%s simulation_id=%s", state.call_id, simulation_id)
        if state.closed:
            export_call(state.call_id)
        return {"status": "claimed", "call_id": state.call_id}

    if len(_pending_simulations) >= MAX_PENDING_SIMULATIONS:
        _pending_simulations.popleft()
    _pending_simulations.append(PendingSimulation(simulation_id=simulation_id, run_id=run_id))
    logger.info("Queued Coval simulation registration: simulation_id=%s", simulation_id)
    return {"status": "queued", "pending": len(_pending_simulations)}


def record_webhook(
    *,
    call_id: str,
    msg_type: str,
    message: dict[str, Any] | None = None,
    latency_ms: float | None = None,
    attributes: dict[str, Any] | None = None,
) -> None:
    state = _get_call(call_id)
    safe_type = (msg_type or "unknown").replace("-", "_")[:80]
    start_ns, end_ns = _message_timing(state, message or {}, latency_ms)
    span = _start_child_span(state, f"vapi.webhook.{safe_type}", start_ns=start_ns, kind=SpanKind.SERVER)
    span.set_attribute("messaging.system", "vapi")
    span.set_attribute("vapi.message.type", msg_type or "unknown")
    if latency_ms is not None:
        span.set_attribute("webhook.processing_latency_ms", float(latency_ms))
    for key, value in (attributes or {}).items():
        if isinstance(value, bool):
            span.set_attribute(key, value)
        elif isinstance(value, (int, float)):
            span.set_attribute(key, value)
        elif value is not None:
            span.set_attribute(key, str(value)[:500])
    span.set_status(Status(StatusCode.OK))
    span.end(end_time=end_ns)


def record_assistant_request(
    call_id: str,
    message: dict[str, Any] | None = None,
    latency_ms: float | None = None,
) -> None:
    call = (message or {}).get("call") if isinstance((message or {}).get("call"), dict) else {}
    record_webhook(
        call_id=call_id,
        msg_type="assistant-request",
        message=message,
        latency_ms=latency_ms,
        attributes={
            "vapi.assistant_id": (message or {}).get("assistantId") or call.get("assistantId"),
            "vapi.call.id": call.get("id") or call_id,
        },
    )


def record_vapi_event(call_id: str, msg_type: str) -> None:
    state = _get_call(call_id)
    safe_type = (msg_type or "unknown").replace("-", "_")[:80]
    state.event_counts[safe_type] = state.event_counts.get(safe_type, 0) + 1
    if safe_type in {"user_interrupted", "status_update"}:
        state.root_span.add_event(f"vapi.{safe_type}", {"vapi.message.type": msg_type})


def record_tool_call(
    *,
    call_id: str,
    tool_call_id: str,
    name: str,
    args: dict[str, Any],
    result: str,
    latency_ms: float,
) -> None:
    state = _get_call(call_id)
    details = _parse_tool_result(result)
    state.tool_call_count += 1
    state.tool_failure_count += int(details["tool_error"])
    state.dependency_blocked = max(state.dependency_blocked, int(details["dependency_unavailable"]))
    state.fallback_used = max(state.fallback_used, int(details["fallback_used"]))

    parent_context = trace.set_span_in_context(state.root_span)
    tracer = state.provider.get_tracer(SERVICE) if state.provider else trace.get_tracer(SERVICE)
    end_ns = _now_ns()
    start_ns = max(end_ns - int(latency_ms * 1_000_000), state.root_span.start_time or end_ns)
    span = tracer.start_span("llm_tool_call", context=parent_context, kind=SpanKind.INTERNAL, start_time=start_ns)
    span.set_attribute("function.name", name)
    span.set_attribute("tool_call_id", tool_call_id)
    span.set_attribute("function.arguments", _safe_arguments(args))
    span.set_attribute("tool.latency_ms", float(latency_ms))
    span.set_attribute("tool.error", int(details["tool_error"]))
    span.set_attribute("tool.dependency_unavailable", int(details["dependency_unavailable"]))
    span.set_attribute("tool.result.count", int(details["result_count"]))
    if details["error_type"]:
        span.set_attribute("error.type", details["error_type"])
        span.set_status(Status(StatusCode.ERROR, details["error_type"]))
    else:
        span.set_status(Status(StatusCode.OK))
    span.end(end_time=end_ns)

    workflow_span = _start_child_span(state, f"healthcare.workflow.{name}", start_ns=start_ns)
    workflow_span.set_attribute("workflow.step", name)
    workflow_span.set_attribute("tool_call_id", tool_call_id)
    workflow_span.set_attribute("tool.error", int(details["tool_error"]))
    workflow_span.set_attribute("tool.result.count", int(details["result_count"]))
    if details["error_type"]:
        workflow_span.set_attribute("error.type", details["error_type"])
        workflow_span.set_status(Status(StatusCode.ERROR, details["error_type"]))
    else:
        workflow_span.set_status(Status(StatusCode.OK))
    workflow_span.end(end_time=end_ns)


def finish_call(
    call_id: str,
    end_report: dict[str, Any] | None = None,
    latency_ms: float | None = None,
) -> dict[str, Any]:
    state = _get_call(call_id)
    if state.closed:
        return export_call(state.call_id)

    end_report = end_report or {}
    ended_reason = end_report.get("endedReason", "")
    call = end_report.get("call") if isinstance(end_report.get("call"), dict) else {}
    if not ended_reason:
        ended_reason = call.get("endedReason", "")

    if end_report:
        record_webhook(
            call_id=call_id,
            msg_type="end-of-call-report",
            message=end_report,
            latency_ms=latency_ms,
            attributes={
                "vapi.call.id": call.get("id") or call_id,
                "vapi.ended_reason": ended_reason,
            },
        )
        _set_call_report_attributes(state, end_report)
        _record_end_report_span(state, end_report)
        _record_vapi_artifact_spans(state, end_report)

    state.root_span.set_attribute("tool.call.count", state.tool_call_count)
    state.root_span.set_attribute("tool.failure.count", state.tool_failure_count)
    state.root_span.set_attribute("workflow.completed", 1 if not state.tool_failure_count and not state.dependency_blocked else 0)
    state.root_span.set_attribute("workflow.dependency_blocked", state.dependency_blocked)
    state.root_span.set_attribute("workflow.fallback_used", state.fallback_used)
    state.root_span.set_attribute("user.interruption.count", state.event_counts.get("user_interrupted", 0))
    for event_type, count in sorted(state.event_counts.items()):
        state.root_span.set_attribute(f"vapi.event.{event_type}.count", count)
    if not end_report:
        state.root_span.set_attribute("call.duration_seconds", max(time.time() - state.created_at, 0.0))
    if ended_reason:
        state.root_span.set_attribute("vapi.ended_reason", ended_reason[:200])
    state.root_span.add_event("conversation_end")
    state.root_span.set_status(Status(StatusCode.OK))
    state.root_span.end(end_time=_now_ns())
    state.closed = True
    state.closed_at = time.time()
    return export_call(state.call_id)


def export_call(call_id: str) -> dict[str, Any]:
    state = _calls.get(call_id)
    if not state:
        return {"exported": False, "reason": "unknown_call"}
    if state.exported:
        return {"exported": True, "span_count": 0, "reason": "already_exported"}
    if not state.closed:
        return {"exported": False, "reason": "call_open"}
    if not state.simulation_id:
        logger.warning("Coval trace buffered without simulation ID: call=%s", call_id)
        return {"exported": False, "reason": "missing_simulation_id"}

    api_key = os.environ.get("COVAL_API_KEY", "")
    if not api_key:
        logger.warning("COVAL_API_KEY env var not set; trace buffered but not exported: call=%s", call_id)
        return {"exported": False, "reason": "missing_api_key"}

    spans = list(state.in_memory_exporter.get_finished_spans() if state.in_memory_exporter else [])
    if not spans:
        return {"exported": False, "reason": "no_finished_spans"}

    exporter = OTLPSpanExporter(
        endpoint=COVAL_TRACES_ENDPOINT,
        headers={"X-API-Key": api_key, "X-Simulation-Id": state.simulation_id},
        timeout=30,
    )
    result = exporter.export(spans)
    if result == SpanExportResult.SUCCESS:
        state.exported = True
        state.export_error = None
        if state.in_memory_exporter:
            state.in_memory_exporter.clear()
        if state.provider:
            state.provider.shutdown()
        logger.info(
            "Coval trace export accepted: call=%s simulation_id=%s span_count=%s",
            call_id,
            state.simulation_id,
            len(spans),
        )
        return {"exported": True, "span_count": len(spans)}

    state.export_error = str(result)
    logger.error("Coval trace export failed: call=%s simulation_id=%s result=%s", call_id, state.simulation_id, result)
    return {"exported": False, "reason": "export_failed", "result": str(result), "span_count": len(spans)}


def debug_status() -> dict[str, Any]:
    _cleanup_stale()
    return {
        "pending_simulations": len(_pending_simulations),
        "active_calls": len([state for state in _calls.values() if not state.closed]),
        "closed_unexported_calls": len([state for state in _calls.values() if state.closed and not state.exported]),
        "registration_auth_configured": registration_secret_configured(),
    }
