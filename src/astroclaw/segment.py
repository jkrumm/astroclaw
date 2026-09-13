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
GROUNDED_SAM_VERSION = "ee871c19efb1941f55f66a3d7d960428c8a5afcb77449547fe8e5a3ab9ebc21c"  # schananas/grounded_sam
GROUND_PROMPT = "building,fence,ground,grass,hill,mountain,rock,road,wall"
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


def _predict(model: str, version: str, inputs: dict, cache: Path) -> list[str]:
    from astroclaw.usage import record_prediction
    pred = _api("/predictions", {"version": version, "input": inputs})
    while pred.get("status") not in ("succeeded", "failed", "canceled"):
        time.sleep(2)
        pred = _api(pred["urls"]["get"])
    record_prediction(model, f"segment:{model.split('/')[-1]}", pred, cache.parent)
    if pred["status"] != "succeeded":
        raise SystemExit(f"segmentation failed: {pred.get('error')}")
    out = pred["output"]
    return [out] if isinstance(out, str) else list(out)


def _fetch(url: str) -> np.ndarray:
    with urllib.request.urlopen(url, timeout=180) as resp:
        return cv2.imdecode(np.frombuffer(resp.read(), np.uint8), cv2.IMREAD_UNCHANGED)


def _upload(image_rgb8: np.ndarray) -> str:
    h, w = image_rgb8.shape[:2]
    scale = UPLOAD_LONG_EDGE / max(h, w)
    small = cv2.resize(image_rgb8, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()


def _cached(cache: Path, shape: tuple[int, int], compute) -> np.ndarray:
    if cache.exists():
        mask = cv2.imread(str(cache), cv2.IMREAD_GRAYSCALE)
    else:
        mask = compute()
        cv2.imwrite(str(cache), mask)
    return cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0


def birefnet_mask(image_rgb8: np.ndarray, cache: Path) -> np.ndarray:
    """The main object, to the rod: foreground probability 0..1 at the input's resolution. Cached."""
    def compute():
        rgba = _fetch(_predict("men1scus/birefnet", BIREFNET_VERSION,
                               {"image": _upload(image_rgb8), "resolution": "1024x1024"}, cache)[0])
        return rgba[..., 3] if rgba.ndim == 3 and rgba.shape[2] == 4 else cv2.cvtColor(rgba, cv2.COLOR_BGR2GRAY)
    return _cached(cache, image_rgb8.shape[:2], compute)


def ground_objects_mask(image_rgb8: np.ndarray, cache: Path, prompt: str = GROUND_PROMPT) -> np.ndarray:
    """Everything at ground level a salient-object model ignores (Grounded-SAM, text prompt). Cached."""
    def compute():
        outputs = _predict("schananas/grounded_sam", GROUNDED_SAM_VERSION,
                           {"image": _upload(image_rgb8), "mask_prompt": prompt,
                            "negative_mask_prompt": "sky", "adjustment_factor": 0}, cache)
        mask = next((u for u in outputs if u.rsplit("/", 1)[-1] == "mask.jpg"), None)
        if mask is None:
            raise SystemExit(f"grounded-sam returned no mask.jpg: {outputs}")
        img = _fetch(mask)
        return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return _cached(cache, image_rgb8.shape[:2], compute)
