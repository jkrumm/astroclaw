"""Optional blend inputs: a long-exposure foreground frame, and sky-only gradient removal.

Both work in linear light. The stacks are gamma-encoded camera JPG data, and subtracting a glow or
scaling a brightness is only physically meaningful on linear values.
"""
from pathlib import Path

import cv2
import numpy as np

MAX_ALIGN_SHIFT_PX = 30   # a tripod nudge; anything larger is a different composition
MIN_ALIGN_CC = 0.6


def to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055).astype(np.float32)


def _gray(img: np.ndarray) -> np.ndarray:
    return img @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _erode(mask: np.ndarray, px: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.erode((mask > 0.5).astype(np.uint8), k)


def load_foreground_frame(path: Path, shape: tuple[int, int]) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)   # applies EXIF orientation: display frame, like the blend
    if bgr is None:
        raise SystemExit(f"cannot read {path}")
    if bgr.shape[:2] != shape:
        raise SystemExit(f"{path.name} is {bgr.shape[1]}x{bgr.shape[0]}, the stack is {shape[1]}x{shape[0]}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def align_and_match(frame: np.ndarray, fg_stack: np.ndarray, ground: np.ndarray,
                    tone: str = "lift", sky_level: float | None = None) -> tuple[np.ndarray, dict]:
    """Translate `frame` onto the stack using ground pixels only; with tone="stack" also match its brightness.

    tone="stack" pulls it down to the stack's tonal range. tone="frame" keeps the long exposure's own
    brightness, which also keeps its glow: a hazy far ridge then outshines the de-glowed sky above it
    and reads as a lit band. tone="lift" (default) is the long exposure where the ground is dark and the
    stack's tone where the ground is as bright as the horizon sky (`sky_level`, linear luminance).

    ECC is invariant to gain and offset, so a frame one stop brighter still aligns. Brightness is
    matched on block medians of the ground, so the result sits in the stack's tonal range and the
    seam against the sky stays invisible.
    """
    s = 4
    h, w = fg_stack.shape[:2]
    small = lambda im: cv2.resize(im, (w // s, h // s), interpolation=cv2.INTER_AREA)
    region = _erode(small(ground), 8)
    if region.sum() < 1000:
        raise SystemExit("not enough foreground to align on")
    template = cv2.GaussianBlur(small(_gray(fg_stack)), (0, 0), 1.5)
    moving = cv2.GaussianBlur(small(_gray(frame)), (0, 0), 1.5)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    cc, warp = cv2.findTransformECC(template, moving, warp, cv2.MOTION_TRANSLATION, criteria, region, 5)
    dx, dy = float(warp[0, 2] * s), float(warp[1, 2] * s)
    if cc < MIN_ALIGN_CC or max(abs(dx), abs(dy)) > MAX_ALIGN_SHIFT_PX:
        raise SystemExit(f"foreground frame does not match the stack (cc={cc:.2f}, shift={dx:.1f},{dy:.1f} px)")
    full = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    aligned = cv2.warpAffine(frame, full, (w, h), flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REFLECT)

    if tone == "frame":
        return aligned, {"shift_px": [round(dx, 1), round(dy, 1)], "ecc": round(float(cc), 3), "tone": "frame"}
    # brightness: a per-channel tone curve, by matching the two histograms over the same ground pixels.
    # A straight gain fitted to dark near hills left hazy far ridges and cloud seas visibly too light.
    sel = _erode(ground, 24) > 0
    sel_small = cv2.resize(sel.astype(np.uint8), (w // s, h // s), interpolation=cv2.INTER_NEAREST) > 0
    frame_small, stack_small = small(aligned)[sel_small], small(fg_stack)[sel_small]
    q = np.linspace(0, 100, 257)
    matched = np.empty_like(aligned)
    for c in range(3):
        src_q = np.maximum.accumulate(np.percentile(frame_small[:, c], q))
        dst_q = np.percentile(stack_small[:, c], q)
        matched[..., c] = np.interp(aligned[..., c], src_q, dst_q)
    mid = np.percentile(frame_small, 50, axis=0), np.percentile(stack_small, 50, axis=0)
    ratio = [round(float(to_linear(d) / max(to_linear(f), 1e-6)), 3) for f, d in zip(*mid)]
    info = {"shift_px": [round(dx, 1), round(dy, 1)], "ecc": round(float(cc), 3), "median_gain": ratio, "tone": tone}
    if tone == "lift" and sky_level:
        base, bright = to_linear(matched), to_linear(aligned)
        lift = np.clip(1 - _gray(base) / sky_level, 0, 1)[..., None]
        info["sky_level"] = round(float(sky_level), 4)
        return to_srgb(base + (bright - base) * lift), info
    return matched, info


def sky_gradient(img: np.ndarray, sky: np.ndarray, degree: int) -> tuple[np.ndarray, dict]:
    """Fit a smooth per-channel background over open sky, in linear light.

    Returns the absolute background and the level to keep: subtracting `model - level` flattens the
    glow while the sky keeps its overall brightness.

    Block medians suppress stars; iterative clipping of blocks far above the fit keeps the Milky
    Way's own glow out of the model, so it survives the subtraction.
    """
    b = 64
    h, w = img.shape[:2]
    lin = to_linear(img)
    gh, gw = h // b, w // b
    med = np.median(lin[: gh * b, : gw * b].reshape(gh, b, gw, b, 3), axis=(1, 3))
    sky_blocks = cv2.resize(_erode(sky, 16).astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA) > 0.99
    yy, xx = np.mgrid[0:gh, 0:gw]
    ys, xs = (yy + 0.5) / gh * 2 - 1, (xx + 0.5) / gw * 2 - 1
    terms = lambda y, x: np.stack([y ** i * x ** j for i in range(degree + 1) for j in range(degree + 1 - i)], -1)
    design = terms(ys, xs)

    keep = sky_blocks.copy()
    for _ in range(4):
        A = design[keep]
        coef = [np.linalg.lstsq(A, med[..., c][keep], rcond=None)[0] for c in range(3)]
        fit = np.stack([design @ k for k in coef], -1)
        resid = _gray(med - fit)
        mad = 1.4826 * np.median(np.abs(resid[keep] - np.median(resid[keep])))
        keep = sky_blocks & (resid < 2.0 * mad)
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    full = terms(Y / h * 2 - 1, X / w * 2 - 1)
    model = np.stack([full @ k for k in coef], -1).astype(np.float32)
    sky_full = sky > 0.5
    level = np.array([np.median(model[..., c][sky_full]) for c in range(3)], dtype=np.float32)
    info = {"degree": degree, "blocks_used": int(keep.sum()), "blocks_sky": int(sky_blocks.sum()),
            "range": round(float(np.ptp(_gray(model)[sky_full])), 4)}
    return model, level, info
