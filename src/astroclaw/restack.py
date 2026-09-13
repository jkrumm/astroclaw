"""Restack the sky with the foreground cut out of every aligned frame.

Aligning on the stars moves the foreground from frame to frame, so the ordinary sky stack carries a
smeared copy of every ridge, tower and roof up to `max_shift_px` around it. Hiding that ghost behind
a margin taken from the unaligned stack leaves a starless halo. Masking the foreground per frame
removes the ghost at its source: each sky pixel is averaged only over the frames in which it was sky.

Works for any foreground shape. `rows` limits the work to where the foreground can have ghosted.
"""
from pathlib import Path

import cv2
import numpy as np
from astropy.io import fits

EDGE_PX = 60          # ECC looks this far around the foreground outline, where its edge carries the signal
MASK_GROW_PX = 6      # safety margin around each frame's foreground; edges are soft in a single frame
MIN_FRAMES = 5        # a restack needs at least this many aligned frames
MIN_ECC = 0.8         # the foreground outline matches at 0.95+; a frame far below aligned on something else
FULL_COVERAGE = 10    # a sky pixel seen by this many frames uses the restack alone; fewer blend towards fg
STAR_COVERAGE = 3     # frames needed before a low-coverage pixel carries star detail at full strength
CHUNK_ROWS = 192


def _gray(img: np.ndarray) -> np.ndarray:
    return img[..., :3].astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def display_frames(workdir: Path, rotate) -> list[Path]:
    """Aligned frames in display orientation as float16 .npy (memory-mappable); the FITS are deleted."""
    work = workdir / "work"
    for fit in sorted(work.glob("r_f_*.fit")):
        target = fit.with_suffix(".npy")
        if not target.exists():
            data = fits.getdata(fit).astype(np.float32)
            np.save(target, rotate(np.flipud(np.moveaxis(data, 0, -1))).astype(np.float16))
            del data
        fit.unlink()
    frames = sorted(work.glob("r_f_*.npy"))
    if not frames:
        raise SystemExit("no aligned frames: re-run `astroclaw stack` without --discard-work")
    return frames


def restack(workdir: Path, rotate, fg: np.ndarray, sky: np.ndarray, ground: np.ndarray,
            rows: tuple[int, int], fade_top: bool) -> tuple[np.ndarray, dict]:
    frames = display_frames(workdir, rotate)
    h, w = fg.shape[:2]
    r0, r1 = rows
    region_h = r1 - r0
    s = 2 if region_h < h // 3 else 4
    small = lambda im: cv2.resize(im, (w // s, region_h // s), interpolation=cv2.INTER_AREA)

    region_ground = (ground[r0:r1] > 0.5).astype(np.uint8)
    edge_zone = cv2.dilate(region_ground, np.ones((2 * EDGE_PX + 1,) * 2, np.uint8)) - \
        cv2.erode(region_ground, np.ones((2 * EDGE_PX + 1,) * 2, np.uint8))
    ecc_region = cv2.resize(edge_zone, (w // s, region_h // s), interpolation=cv2.INTER_NEAREST)
    template = cv2.GaussianBlur(small(_gray(fg[r0:r1])), (0, 0), 2)
    grown = cv2.dilate(region_ground, np.ones((2 * MASK_GROW_PX + 1,) * 2, np.uint8))
    sky_sample = (grown[::8, ::8] == 0)

    # pass 1, per frame: where did the foreground move to, and how bright is this frame's sky
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    used, per_frame = [], []
    for path in frames:
        frame = np.load(path, mmap_mode="r")[r0:r1]
        moving = cv2.GaussianBlur(small(_gray(frame)), (0, 0), 2)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            cc, warp = cv2.findTransformECC(template, moving, warp, cv2.MOTION_EUCLIDEAN, criteria, ecc_region, 5)
        except cv2.error:
            cc = 0.0
        info = {"frame": path.name, "ecc": round(float(cc), 3), "used": bool(cc >= MIN_ECC)}
        per_frame.append(info)
        if cc < MIN_ECC:
            continue
        full = warp.copy()
        full[:, 2] *= s
        info["shift_px"] = [round(float(full[0, 2]), 1), round(float(full[1, 2]), 1)]
        moved = cv2.warpAffine(grown, full, (w, region_h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=1)
        sample = np.asarray(frame[::8, ::8], dtype=np.float32)[sky_sample & (moved[::8, ::8] == 0)]
        med = np.median(sample, axis=0)
        mad = np.median(np.abs(sample - med), axis=0)
        used.append((path, np.packbits(moved, axis=1), med, mad))
    if len(used) < MIN_FRAMES:
        raise SystemExit(f"only {len(used)} frames aligned on the foreground outline: {per_frame}")

    # pass 2, in row chunks: clipped mean over the frames in which each pixel was sky
    ref_med, ref_mad = used[0][2], used[0][3]
    band_sky = np.empty((region_h, w, 3), np.float32)
    count = np.empty((region_h, w), np.int16)
    for c0 in range(0, region_h, CHUNK_ROWS):
        c1 = min(c0 + CHUNK_ROWS, region_h)
        cube = np.empty((len(used), c1 - c0, w, 3), np.float32)
        for k, (path, packed, med, mad) in enumerate(used):
            chunk = np.asarray(np.load(path, mmap_mode="r")[r0 + c0:r0 + c1], dtype=np.float32)
            invalid = np.unpackbits(packed[c0:c1], axis=1, count=w).astype(bool) | (chunk.max(-1) <= 0)
            chunk = (chunk - med) * (ref_mad / np.maximum(mad, 1e-6)) + ref_med
            chunk[invalid] = np.nan
            cube[k] = chunk
        for _ in range(2):
            centre = np.nanmean(cube, axis=0)
            spread = np.nanstd(cube, axis=0)
            cube[np.abs(cube - centre) > 3 * np.maximum(spread, 1e-4)] = np.nan
        count[c0:c1] = np.sum(~np.isnan(cube[..., 0]), axis=0)
        band_sky[c0:c1] = np.nanmean(cube, axis=0)
        del cube

    # onto the ordinary sky stack's scale, on pixels every frame saw as sky (no ghost can reach them)
    everyone = (count == len(used)) & (grown == 0)
    pick = np.flatnonzero(everyone.ravel())[:: max(1, int(everyone.sum() // 200_000))]
    new, ref = band_sky.reshape(-1, 3)[pick], sky[r0:r1].reshape(-1, 3)[pick]
    for c in range(3):
        gain, offset = np.polyfit(new[:, c], ref[:, c], 1)
        band_sky[..., c] = gain * band_sky[..., c] + offset

    # Next to the foreground few frames saw sky, and some pixels none: blend towards the unaligned stack,
    # which has no ghost, with a local brightness offset carried in from well-covered sky nearby.
    sky_px = region_ground == 0
    well = (sky_px & (count >= len(used) // 2)).astype(np.float32)
    fill = np.nan_to_num(band_sky)
    # Filling from the unaligned stack left a starless, lighter strip along the outline. Instead the
    # sky's own smooth background is carried down to the outline (normalised convolution from
    # well-covered sky), and star detail is added wherever at least a few frames saw sky.
    den = cv2.GaussianBlur(well, (0, 0), 25) + 1e-6
    background = np.stack([cv2.GaussianBlur(cv2.medianBlur(fill[..., c], 5) * well, (0, 0), 25) / den
                           for c in range(3)], -1)
    seen = (count > 0).astype(np.float32)
    local_bg = np.stack([cv2.GaussianBlur(cv2.medianBlur(fill[..., c], 5) * seen, (0, 0), 25)
                         / (cv2.GaussianBlur(seen, (0, 0), 25) + 1e-6) for c in range(3)], -1)
    detail = (fill - local_bg) * seen[..., None]
    trust = np.clip(count / FULL_COVERAGE, 0, 1)[..., None]
    stars = np.clip(count / STAR_COVERAGE, 0, 1)[..., None]
    band_sky = fill * trust + (background + detail * stars) * (1 - trust)

    out = sky.copy()
    if fade_top:   # a band: fade in from its top, where the ordinary stack was already ghost-free
        fade_rows = min(80, region_h // 4)
        fade = np.clip(np.arange(region_h) / fade_rows - 0.5, 0, 1)[:, None, None]
        out[r0:r1] = band_sky * fade + sky[r0:r1] * (1 - fade)
    else:
        out[r0:r1] = band_sky
    report = {"rows": [r0, r1], "frames_used": len(used), "empty_sky_px": int(((count == 0) & sky_px).sum()),
              "frames": per_frame}
    return out, report
