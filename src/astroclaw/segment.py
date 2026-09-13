"""Foreground segmentation with BiRefNet on Replicate, for structures thresholds cannot follow.

A lattice tower, antenna rods and branches are thinner than any safe dilation and share the sky's
brightness range, so a deviation-from-background mask either misses them or swallows the sky between
them. BiRefNet (high-resolution dichotomous segmentation) is trained for exactly this class of object.

Configuration, environment only:
  REPLICATE_API_TOKEN   bearer token
  REPLICATE_BASE_URL    default https://api.replicate.com/v1 (any Replicate-compatible gateway)
"""
import base64
import json
import os
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

BIREFNET_VERSION = "f74986db0355b58403ed20963af156525e2891ea3c2d499bfbfb2a28cd87c5d7"  # men1scus/birefnet
UPLOAD_LONG_EDGE = 2016  # the model's own input size is 1024; more upload only buys JPEG detail on thin members


def _api(path: str, body: dict | None = None) -> dict:
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise SystemExit("REPLICATE_API_TOKEN is not set (segmentation runs on Replicate)")
    base = os.environ.get("REPLICATE_BASE_URL", "https://api.replicate.com/v1").rstrip("/")
    url = path if path.startswith("http") else f"{base}{path}"
    if url.startswith("https://api.replicate.com/v1") and not base.startswith("https://api.replicate.com"):
        url = base + url.removeprefix("https://api.replicate.com/v1")   # polling URLs point at the origin
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                 method="POST" if body else "GET",
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                                          "Prefer": "wait=60"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.load(resp)


def birefnet_mask(image_rgb8: np.ndarray, cache: Path) -> np.ndarray:
    """Foreground probability 0..1 at the input's resolution. Cached: a re-blend never re-uploads."""
    h, w = image_rgb8.shape[:2]
    if cache.exists():
        alpha = cv2.imread(str(cache), cv2.IMREAD_GRAYSCALE)
    else:
        scale = UPLOAD_LONG_EDGE / max(h, w)
        small = cv2.resize(image_rgb8, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        uri = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()
        pred = _api("/predictions", {"version": BIREFNET_VERSION, "input": {"image": uri, "resolution": "1024x1024"}})
        while pred.get("status") not in ("succeeded", "failed", "canceled"):
            time.sleep(2)
            pred = _api(pred["urls"]["get"])
        if pred["status"] != "succeeded":
            raise SystemExit(f"segmentation failed: {pred.get('error')}")
        with urllib.request.urlopen(pred["output"], timeout=180) as resp:
            rgba = cv2.imdecode(np.frombuffer(resp.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        alpha = rgba[..., 3] if rgba.ndim == 3 and rgba.shape[2] == 4 else cv2.cvtColor(rgba, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(cache), alpha)
    return cv2.resize(alpha, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
