"""Monitoring conversation cron job.

Periodically submits a random transcript from `pool/transcripts.json` to
Coval's `/v1/conversations:submit` endpoint as a monitoring conversation.

Run via Fly machine schedule (e.g. hourly).

Env:
  COVAL_API_KEY       - x-api-key for Coval API
  COVAL_AGENT_ID      - target agent id
  COVAL_API_BASE      - default https://api.coval.dev
"""

import json
import logging
import os
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

POOL_PATH = Path(__file__).parent / "pool" / "transcripts.json"
COVAL_API_BASE = os.environ.get("COVAL_API_BASE", "https://api.coval.dev")
COVAL_API_KEY = os.environ.get("COVAL_API_KEY", "")
COVAL_AGENT_ID = os.environ.get("COVAL_AGENT_ID", "")


def load_pool() -> list:
    if not POOL_PATH.exists():
        logger.warning(f"Pool file not found: {POOL_PATH}")
        return []
    try:
        data = json.loads(POOL_PATH.read_text())
    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse pool: {exc}")
        return []
    if not isinstance(data, list):
        logger.error("Pool is not a list")
        return []
    return data


def submit_conversation(transcript) -> bool:
    if not COVAL_API_KEY or not COVAL_AGENT_ID:
        logger.error("Missing COVAL_API_KEY or COVAL_AGENT_ID")
        return False

    # Normalize transcript: accept list-of-messages, dict with .transcript, or a string
    if isinstance(transcript, dict):
        transcript_payload = transcript.get("transcript", transcript)
    else:
        transcript_payload = transcript

    body = {
        "agent_id": COVAL_AGENT_ID,
        "external_id": f"brontemoor-cron-{uuid.uuid4()}",
        "occurred_at": datetime.now(tz=timezone.utc).isoformat(),
        "transcript": transcript_payload,
        "metadata": {
            "source": "brontemoor-fly-cron",
            "brand": "Brontemoor Medical Group",
            "agent_persona": "Nurse Bronancy",
        },
    }

    url = f"{COVAL_API_BASE}/v1/conversations:submit"
    headers = {"x-api-key": COVAL_API_KEY, "content-type": "application/json"}

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=headers, json=body)
        logger.info(f"submit status={resp.status_code} body={resp.text[:300]}")
        return resp.status_code < 300
    except httpx.HTTPError as exc:
        logger.error(f"submit failed: {exc}")
        return False


def main() -> int:
    pool = load_pool()
    if not pool:
        logger.warning("Pool empty - nothing to submit")
        return 0
    transcript = random.choice(pool)
    ok = submit_conversation(transcript)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
