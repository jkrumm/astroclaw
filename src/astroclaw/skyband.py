"""Restack the sky near the skyline with the ground cut out of every aligned frame.

Aligning on the stars moves the ground from frame to frame, so the ordinary sky stack carries a
smeared copy of the ridge up to `max_shift_px` above it. Hiding that ghost behind a margin taken from
the unaligned stack leaves a starless band between ridge and sky. Masking the ground per frame
removes the ghost at its source: every sky pixel is averaged only over frames in which it was sky,
so stars reach the ridge.
"""
from pathlib import Path

import cv2
import numpy as np
from astropy.io import fits

EDGE_PX = 60          # ECC looks at this band around the skyline, where the ridge edge carries the signal
MASK_GROW_PX = 6      # safety margin around each frame's ground; the ridge edge is soft in a single frame
MIN_FRAMES = 5        # a restack needs at least this many aligned frames overall
FULL_COVERAGE = 10    # a sky pixel seen by this many frames uses the restack alone; fewer blend towards the unaligned stack
MIN_ECC = 0.5


def _gray(img: np.ndarray) -> np.ndarray:
    return img @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def skyline(ground: np.ndarray) -> np.ndarray:
    """Row of the first ground pixel per column (display orientation)."""
    g = ground > 0.5
    return np.where(g.any(0), g.argmax(0), ground.shape[0]).astype(np.float32)


def restack_band(workdir: Path, rotate, fg: np.ndarray, sky: np.ndarray, ground: np.ndarray,
                 shift: float) -> tuple[np.ndarray, dict]:
    frames = sorted((workdir / "work").glob("r_f_*.fit"))
    if not frames:
        raise SystemExit("no aligned frames: re-run `astroclaw stack` without --discard-work")
    h, w = fg.shape[:2]
    line = skyline(ground)
    r0 = max(int(line.min() - 2 * shift - 120), 0)
    r1 = min(int(line.max() + 40), h)
    band_ground = ground[r0:r1]
    band_line = line - r0
    rows = np.arange(r1 - r0)[:, None]

    s = 2
    small = lambda im: cv2.resize(im, ((w // s), (r1 - r0) // s), interpolation=cv2.INTER_AREA)
    template = cv2.GaussianBlur(small(_gray(fg[r0:r1])), (0, 0), 2)
    ecc_region = (np.abs(rows - band_line[None, :]) < EDGE_PX).astype(np.uint8)
    ecc_region = cv2.resize(ecc_region, (w // s, (r1 - r0) // s), interpolation=cv2.INTER_NEAREST)
    grown = cv2.dilate(band_ground.astype(np.uint8), np.ones((2 * MASK_GROW_PX + 1,) * 2, np.uint8))

    stack, per_frame = [], []
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    for path in frames:
        data = fits.getdata(path).astype(np.float32)
        band = rotate(np.flipud(np.moveaxis(data, 0, -1)))[r0:r1]
        del data
        moving = cv2.GaussianBlur(small(_gray(band)), (0, 0), 2)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            cc, warp = cv2.findTransformECC(template, moving, warp, cv2.MOTION_EUCLIDEAN, criteria, ecc_region, 5)
        except cv2.error:
            cc = 0.0
        if cc < MIN_ECC:
            per_frame.append({"frame": path.name, "ecc": round(float(cc), 3), "used": False})
            continue
        full = warp.copy()
        full[:, 2] *= s
        # ground of this frame: the reference ground moved the way ECC says the ridge moved
        frame_ground = cv2.warpAffine(grown, full, (w, r1 - r0), flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=1)
        invalid = (frame_ground > 0) | (band.max(-1) <= 0)
        band[invalid] = np.nan
        stack.append(band)
        per_frame.append({"frame": path.name, "ecc": round(float(cc), 3), "used": True,
                          "shift_px": [round(float(full[0, 2]), 1), round(float(full[1, 2]), 1)]})

    if len(stack) < MIN_FRAMES:
        raise SystemExit(f"only {len(stack)} frames aligned on the skyline; ECC failed: {per_frame}")
    cube = np.stack(stack)                      # (n, rows, w, 3), NaN = ground or no data
    del stack

    # normalise each frame to the first on its own sky pixels (Siril stacked with addscale)
    ref_med = np.nanmedian(cube[0], axis=(0, 1))
    ref_mad = np.nanmedian(np.abs(cube[0] - ref_med), axis=(0, 1))
    for k in range(1, len(cube)):
        med = np.nanmedian(cube[k], axis=(0, 1))
        mad = np.nanmedian(np.abs(cube[k] - med), axis=(0, 1))
        cube[k] = (cube[k] - med) * (ref_mad / np.maximum(mad, 1e-6)) + ref_med

    # sigma-clipped mean, ignoring masked samples
    centre = np.nanmedian(cube, axis=0)
    spread = 1.4826 * np.nanmedian(np.abs(cube - centre), axis=0)
    cube[np.abs(cube - centre) > 3 * np.maximum(spread, 1e-4)] = np.nan
    count = np.sum(~np.isnan(cube[..., 0]), axis=0)
    band_sky = np.nanmean(cube, axis=0)
    del cube

    # bring the restack onto the ordinary sky stack's scale, on rows far above any ghost
    clean_rows = int(max(band_line.min() - 2 * shift - 40, 20))
    ref = sky[r0:r0 + clean_rows].reshape(-1, 3)
    new = band_sky[:clean_rows].reshape(-1, 3)
    ok = ~np.isnan(new).any(1)
    for c in range(3):
        gain, offset = np.polyfit(new[ok, c], ref[ok, c], 1)
        band_sky[..., c] = gain * band_sky[..., c] + offset

    # Right above the ridge few frames saw sky, and the pixel next to the edge may have none: fill those
    # from the unaligned stack, level-matched on well-covered sky, which has no ghost there by definition.
    sky_px = (band_ground < 0.5)
    well = sky_px & (count >= len(frames) // 2)
    fill = np.nan_to_num(band_sky)
    # local, not global: near the horizon the glow changes how the two stacks relate, and one global
    # fit left a lit rim along the ridge. The offset is measured on well-covered sky nearby and carried
    # down to the edge by normalised convolution.
    fg_band = fg[r0:r1]
    wf = well.astype(np.float32)
    den = cv2.GaussianBlur(wf, (0, 0), 40) + 1e-6
    offset = np.stack([cv2.GaussianBlur((fill[..., c] - fg_band[..., c]) * wf, (0, 0), 40) / den
                       for c in range(3)], -1)
    fg_band = fg_band + offset
    # a pixel seen by 1-3 frames is noise plus whatever of the ridge edge leaked into them; blend by coverage
    trust = np.clip(count / FULL_COVERAGE, 0, 1)[..., None]
    band_sky = fill * trust + fg_band * (1 - trust)
    empty = (count == 0)[..., None]

    fade = np.clip((rows - clean_rows / 2) / max(clean_rows / 2, 1), 0, 1)[..., None]  # blend in from the top
    out = sky.copy()
    out[r0:r1] = band_sky * fade + sky[r0:r1] * (1 - fade)
    report = {"rows": [r0, r1], "frames_used": sum(f["used"] for f in per_frame),
              "empty_sky_px": int((empty[..., 0] & sky_px).sum()), "frames": per_frame}
    return out, report
