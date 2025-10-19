"""Simple healthcheck script for TraderX."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from time import time


HEALTH_FILE = Path("data/logs/healthcheck.jsonl")
TARGETS_FILE = Path("data/targets/targets.csv")
STALE_THRESHOLD_SECONDS = 600


def main() -> int:
    now = time()
    status = {
        "timestamp": now,
        "targets_exists": TARGETS_FILE.exists(),
        "targets_age_seconds": None,
        "status": "ok",
    }
    if TARGETS_FILE.exists():
        age = now - TARGETS_FILE.stat().st_mtime
        status["targets_age_seconds"] = age
        if age > STALE_THRESHOLD_SECONDS:
            status["status"] = "stale"
    else:
        status["status"] = "missing"

    HEALTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    with HEALTH_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(status) + "\n")
    return 0 if status["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
