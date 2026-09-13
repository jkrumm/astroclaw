"""Blend the star-aligned sky stack with the unaligned foreground stack along a detected mask.

Modes:
  horizon  a skyline: per-column horizon line, everything below it is foreground
  objects  lit structures against the sky (towers, buildings): deviation-from-background mask
"""
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from astropy.io import fits

FEATHER_PX = 12
DEFAULT_SHIFT_PX = 75.0

# EXIF Orientation -> rotation that turns sensor pixels (what Siril stacks) into display pixels
ROTATIONS = {3: cv2.ROTATE_180, 6: cv2.ROTATE_90_CLOCKWISE, 8: cv2.ROTATE_90_COUNTERCLOCKWISE}


def load_fits(path: Path) -> np.ndarray:
    data = fits.getdata(path).astype(np.float32)   # (3, H, W), bottom-up
    return np.flipud(np.moveaxis(data, 0, -1))      # (H, W, 3) RGB, top-down


def stretch8(img: np.ndarray, lo: float = 0.5, hi: float = 99.8) -> np.ndarray:
    a, b = np.percentile(img, [lo, hi])
    return (np.clip((img - a) / (b - a), 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def orientation(workdir: Path) -> int:
    first = sorted((workdir / "src").glob("*.[jJ][pP][gG]"))[0]
    out = subprocess.run(["exiftool", "-s3", "-n", "-Orientation", str(first)], capture_output=True, text=True)
    return int(out.stdout.strip() or 1)


def lum(img: np.ndarray) -> np.ndarray:
    return img @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


HORIZON_SEARCH = (0.55, 0.97)   # fraction of image height the skyline may sit in
HORIZON_EDGE = 0.2              # edge strength, relative to the strongest edges in the search band


def horizon_mask(fg: np.ndarray, shift: float) -> np.ndarray:
    """The skyline is the FIRST significant bright-above/dark-below edge from the top.

    Brightness thresholds fail both ways: a dark dust lane at the Milky Way's base reads as ground,
    and a bright cloud sea below a faint far ridge reads as sky. An edge in log luminance is sharp at
    a ridge and soft at a dust lane, and taking the first one keeps the ridge above the cloud sea.
    """
    h, w = fg.shape[:2]
    s = 4
    small = cv2.resize(lum(fg), (w // s, h // s), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape
    top, bottom = int(sh * HORIZON_SEARCH[0]), int(sh * HORIZON_SEARCH[1])
    g = cv2.GaussianBlur(np.log(np.clip(small, 1e-4, None)), (0, 0), 1.2)
    edge = np.zeros_like(g)
    edge[1:-1] = np.maximum(g[:-2] - g[2:], 0)
    edge /= np.percentile(edge[top:bottom], 99.5) + 1e-9
    strong = edge[top:bottom] > HORIZON_EDGE
    horizon = np.where(strong.any(0), top + strong.argmax(0), bottom).astype(np.float32)
    padded = np.pad(horizon, 4, mode="edge")
    horizon = np.array([np.median(padded[i:i + 9]) for i in range(sw)], dtype=np.float32)
    horizon = np.interp(np.arange(w), np.arange(sw) * s + s / 2, horizon * s)
    rows = np.arange(h)[:, None]
    return (rows >= horizon[None, :] - shift).astype(np.float32)  # ghosts reach the full registration shift


def objects_mask(fg: np.ndarray, shift: float) -> np.ndarray:
    h, w = fg.shape[:2]
    s = 4
    small = cv2.resize(fg, (w // s, h // s), interpolation=cv2.INTER_AREA)
    bg = np.stack([cv2.medianBlur((small[..., c] * 255).astype(np.uint8), 101) for c in range(3)], -1) / 255.0
    dev = np.abs(small - bg).max(-1)
    sky_ref = dev[: small.shape[0] // 3, small.shape[1] // 2:]
    sigma = 1.4826 * np.median(np.abs(sky_ref - np.median(sky_ref)))
    m = (dev > np.median(sky_ref) + 8 * sigma).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, np.uint8)
    area = stats[1:, cv2.CC_STAT_AREA]
    x0, y0 = stats[1:, cv2.CC_STAT_LEFT], stats[1:, cv2.CC_STAT_TOP]
    x1, y1 = x0 + stats[1:, cv2.CC_STAT_WIDTH], y0 + stats[1:, cv2.CC_STAT_HEIGHT]
    at_edge = (x0 == 0) | (y0 == 0) | (x1 == m.shape[1])  # an object cut by the frame shows only a sliver
    keep[1:] = (area >= 1500) | (at_edge & (area >= 150))   # drop star and trail specks
    m = keep[labels]
    near = int(shift / 3 / s) * 2 + 1
    m = cv2.dilate(m, np.ones((near, near), np.uint8))
    # bright structures (lit dishes, walls) ghost visibly across the full registration shift
    L = lum(small)
    bright = ((L > np.median(L[: small.shape[0] // 3]) + 0.25) & (m > 0)).astype(np.uint8)
    far = int(shift / s) * 2 + 1
    m |= cv2.dilate(bright, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (far, far)))
    lower = int(m.shape[0] * 0.75)
    for x in range(m.shape[1]):               # below the first structure in the lower quarter is ground
        hits = np.nonzero(m[lower:, x])[0]
        if hits.size:
            m[lower + hits[0]:, x] = 1
    return cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def blend(workdir: Path, mode: str, foreground: Path | None = None, gradient: int | None = None) -> dict:
    reg = workdir / "registration.json"
    shift = json.loads(reg.read_text())["max_shift_px"] if reg.exists() else DEFAULT_SHIFT_PX
    rotation = ROTATIONS.get(orientation(workdir))
    rotate = (lambda im: cv2.rotate(im, rotation)) if rotation is not None else (lambda im: im)
    make_mask = horizon_mask if mode == "horizon" else objects_mask

    sky = rotate(load_fits(workdir / "sky_stack.fit"))
    fg = rotate(load_fits(workdir / "fg_stack.fit"))
    hard = make_mask(fg, shift)
    ground = make_mask(fg, 0.0)          # the foreground itself, without the ghost margin
    report: dict = {"mode": mode, "max_shift_px": shift}

    mask = cv2.GaussianBlur(hard, (0, 0), FEATHER_PX)[..., None]
    out = np.clip(sky * (1 - mask) + fg * mask, 0, 1)

    if foreground is not None:
        # the long exposure replaces the ground only: the ghost margin above it is sky, and its trailed,
        # differently exposed sky must not show there
        from astroclaw.enhance import align_and_match, load_foreground_frame
        frame, report["foreground"] = align_and_match(load_foreground_frame(foreground, fg.shape[:2]), fg, ground)
        report["foreground"]["frame"] = foreground.name
        inner = cv2.GaussianBlur(cv2.erode(ground, np.ones((9, 9), np.uint8)), (0, 0), FEATHER_PX / 2)[..., None]
        out = np.clip(out * (1 - inner) + frame * inner, 0, 1)

    if gradient is not None:
        from astroclaw.enhance import sky_gradient, to_linear, to_srgb
        open_sky = 1.0 - ground
        model, level, report["gradient"] = sky_gradient(sky, open_sky, gradient)
        lin = to_linear(out)
        # Near the skyline, how much a pixel is sky is read off its brightness against the fitted
        # background: glowing sky ~1, dark ridge ~0. The detected line sits a few px off the true edge,
        # and a geometric weight alone left a lit rim above the ridge or a colour-shifted one on it.
        skyness = np.clip(lum(lin) / np.maximum(lum(model), 1e-6), 0, 1)[..., None] ** 2
        # The zone must end softly: a hard edge repeats the skyline's contour inside bright haze below it.
        core = cv2.GaussianBlur(cv2.erode(open_sky, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (97, 97))), (0, 0), 16)[..., None]
        zone = cv2.GaussianBlur(cv2.dilate(open_sky, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))), (0, 0), 14)[..., None]
        weight = np.maximum(core, zone * skyness)
        out = to_srgb(lin - weight * (model - level))
        glow = model - model.min()
        cv2.imwrite(str(workdir / "preview_gradient.jpg"),
                    cv2.cvtColor(stretch8(cv2.resize(glow, None, fx=0.2, fy=0.2), 0, 100), cv2.COLOR_RGB2BGR))

    np.save(workdir / "blend.npy", out)
    small = lambda im: cv2.resize(im, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
    preview = cv2.cvtColor(small(stretch8(out)), cv2.COLOR_RGB2BGR)
    overlay = preview.copy()
    sel = small(mask[..., 0]) > 0.5
    overlay[sel] = (0.5 * overlay[sel] + np.array([0, 0, 128])).astype(np.uint8)
    cv2.imwrite(str(workdir / "preview_blend.jpg"), preview)
    cv2.imwrite(str(workdir / "preview_mask.jpg"), overlay)
    (workdir / "blend.json").write_text(json.dumps(report, indent=2))
    return report


def crop_preview(workdir: Path, y: int, x: int, size: int = 1000) -> Path:
    """A 1:1 crop of the blend, for checking a seam at full resolution."""
    out = np.load(workdir / "blend.npy")
    tile = out[y:y + size, x:x + size]
    path = workdir / f"crop_{y}_{x}.jpg"
    cv2.imwrite(str(path), cv2.cvtColor(stretch8(tile), cv2.COLOR_RGB2BGR))
    return path
