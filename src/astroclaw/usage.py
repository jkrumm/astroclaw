"""Cost of every paid model call, recorded locally and reported to a usage ledger.

Replicate bills public community models by the second of GPU time the prediction was actively
processing (`metrics.predict_time`); setup, queue and idle time are free. The API reports the seconds
but neither the hardware nor the rate, so both are pinned per model here, from the model page and
replicate.com/pricing. `cost_source` stays "computed": a gateway in front of Replicate may bill
differently (per prediction, with a markup), and only the raw seconds survive that question.

Every call is appended to `~/.local/share/astroclaw/usage.jsonl` first, so nothing is lost when the
ledger is unreachable; `astroclaw usage sync` re-sends whatever was not accepted. The ledger is any
endpoint accepting `POST {"records": [...]}` with a bearer token (Argo's `/usage/records` shape):

  ASTROCLAW_USAGE_URL      ledger endpoint; unset = local log only
  ASTROCLAW_USAGE_TOKEN    bearer token
  ASTROCLAW_BILLING        billing lane label (default: "iu" behind a gateway, "replicate" direct)
  ASTROCLAW_MACHINE        machine label (default: short hostname)
"""
import json
import os
import socket
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# USD per second of predict_time, replicate.com/pricing (verified 2026-09-13)
HARDWARE_USD_PER_S = {
    "cpu": 0.000100,
    "t4": 0.000225,
    "l40s": 0.000975,
    "a100-80gb": 0.001400,
    "h100": 0.001525,
}

# model -> hardware, from each model page's "Run time and cost" (verified 2026-09-13)
MODEL_HARDWARE = {
    "men1scus/birefnet": "a100-80gb",
    "schananas/grounded_sam": "l40s",
}

LOG = Path(os.environ.get("ASTROCLAW_USAGE_LOG", Path.home() / ".local/share/astroclaw/usage.jsonl"))


def price(model: str, prediction: dict) -> tuple[float | None, dict]:
    seconds = (prediction.get("metrics") or {}).get("predict_time")
    hardware = MODEL_HARDWARE.get(model)
    rate = HARDWARE_USD_PER_S.get(hardware or "")
    cost = round(seconds * rate, 6) if seconds is not None and rate is not None else None
    return cost, {"predict_time_s": seconds, "total_time_s": (prediction.get("metrics") or {}).get("total_time"),
                  "hardware": hardware, "usd_per_s": rate, "prediction_id": prediction.get("id"),
                  "version": prediction.get("version")}


def _billing() -> str:
    if os.environ.get("ASTROCLAW_BILLING"):
        return os.environ["ASTROCLAW_BILLING"]
    base = os.environ.get("REPLICATE_BASE_URL", "https://api.replicate.com/v1")
    return "replicate" if base.startswith("https://api.replicate.com") else "iu"


def record_prediction(model: str, sub_tool: str, prediction: dict, workdir: Path | None) -> dict:
    """Log one finished (or failed) prediction and try to report it. Never raises into the caller."""
    cost, raw = price(model, prediction)
    now = datetime.now(timezone.utc).isoformat()
    record = {
        "source": "astroclaw",
        "source_id": prediction.get("id") or f"astroclaw-{now}",
        "grain": "request",
        "ts": prediction.get("completed_at") or now,
        "ingested_at": now,
        "model": model,
        "model_norm": model,
        "project": "astroclaw",
        "workspace": "private",
        "sub_tool": sub_tool,
        "machine": os.environ.get("ASTROCLAW_MACHINE") or socket.gethostname().split(".")[0],
        "billing": _billing(),
        "outcome": "ok" if prediction.get("status") == "succeeded" else "error",
        "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "duration_ms": round(raw["predict_time_s"] * 1000) if raw["predict_time_s"] is not None else None,
        "cost_usd": cost,
        "cost_source": "computed" if cost is not None else "none",
        "raw": {**raw, "workdir": workdir.name if workdir else None},
    }
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a") as fh:
            fh.write(json.dumps({"record": record, "synced": False}) + "\n")
        sync()
    except Exception:   # cost accounting must never fail an image job
        pass
    return record


def _post(records: list[dict]) -> bool:
    url, token = os.environ.get("ASTROCLAW_USAGE_URL"), os.environ.get("ASTROCLAW_USAGE_TOKEN")
    if not url or not token:
        return False
    req = urllib.request.Request(url, data=json.dumps({"records": records}).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def sync() -> dict:
    """Send every unsynced record; the ledger upserts on (source, source_id), so a resend is harmless."""
    if not LOG.exists():
        return {"sent": 0, "pending": 0}
    lines = [json.loads(line) for line in LOG.read_text().splitlines() if line.strip()]
    pending = [entry for entry in lines if not entry["synced"]]
    if pending and _post([entry["record"] for entry in pending]):
        for entry in pending:
            entry["synced"] = True
        LOG.write_text("".join(json.dumps(entry) + "\n" for entry in lines))
        return {"sent": len(pending), "pending": 0}
    return {"sent": 0, "pending": len(pending)}


def summary() -> dict:
    if not LOG.exists():
        return {"calls": 0, "cost_usd": 0.0}
    records = [json.loads(line)["record"] for line in LOG.read_text().splitlines() if line.strip()]
    by_model: dict[str, dict] = {}
    for r in records:
        m = by_model.setdefault(r["model"], {"calls": 0, "cost_usd": 0.0})
        m["calls"] += 1
        m["cost_usd"] = round(m["cost_usd"] + (r["cost_usd"] or 0.0), 6)
    return {"calls": len(records), "cost_usd": round(sum(r["cost_usd"] or 0.0 for r in records), 6),
            "by_model": by_model}
