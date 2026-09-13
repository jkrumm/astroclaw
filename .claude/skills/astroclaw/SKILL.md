---
name: astroclaw
description: Stack nightscape / Milky Way / astro photo runs with Siril and blend a star-aligned sky with an unaligned foreground into a 16-bit TIFF, via the astroclaw CLI. Use when the user wants to stack astro shots, "stack the Milky Way frames", reduce noise on a night series, find which frames belong together, or asks how to edit a stacked night sky (gradient removal, stretch, star reduction).
---

# astroclaw

CLI in this repo: `uv run astroclaw <cmd>`. Needs `siril-cli` 1.4+ and `exiftool` on PATH. Read
`README.md` for the model (two stacks, one mask) and `docs/editing.md` for everything after the TIFF.

## Workflow

1. **Find runs** - `uv run astroclaw runs <jpg-dir>`. A run = identical ISO/aperture/shutter, frames
   back to back. Only stack a run that is one composition: open the first and last frame of a run
   (downscale with `sips -Z 1200 <jpg> --out /tmp/x.jpg`) and confirm the framing did not change.
   Default filters: shutter >= 5 s, >= 4 frames. 15+ frames is where stacking clearly pays off.
2. **Prepare** - `uv run astroclaw prepare <jpg-dir> <out-root> --run N`. Copies; never move or edit
   the source JPGs, they may be the only copy of the night.
3. **Stack** - `uv run astroclaw stack <workdir>` (~2-4 min per 25 frames). Check the printed
   `registered` equals `frames`; fewer means Siril dropped frames (clouds, a car's headlights).
   `max_shift_px` is how far the sky moved; it sizes the mask margin automatically.
4. **Pick the mode** from the first frame:
   - `horizon`: the foreground is everything below one skyline (mountains, sea, ridge).
   - `objects`: structures stick up into the sky (tower, tree, building, person).
5. **Blend** - `uv run astroclaw blend <workdir> <mode> [--foreground <jpg>] [--gradient 2]`, then
   **Read `preview_mask.jpg`** (red = taken
   from the unaligned foreground stack). This check is not optional; a green exit code says nothing
   about the mask.
   - Red must cover every foreground object, including thin parts and anything at the frame edges.
   - Red must not cover open sky beyond a small margin; red blobs in the Milky Way are false positives.
   - **`--foreground`**: look for a 1-5 min frame right before or after the run with the SAME framing
     (`runs` hides it: different settings, one frame). Compare it with the run's first frame visually.
     `blend.json` must show `ecc` > 0.9 and a shift of a few px; the command refuses worse.
   - **`--gradient 2`** is the default choice; `3` only for strong, uneven light domes. Read
     `preview_gradient.jpg`: it must be a smooth glow, brightest towards the horizon or a town.
6. **Seam check at 1:1** - `uv run astroclaw crop <workdir> <y> <x>` on the hardest seam (bright
   object edges against the sky, the skyline where it is faintest). Coordinates are display
   orientation, full resolution. Look for doubled outlines (ghosts) next to bright objects and for
   smeared ground just below the horizon. With `--gradient`, also check where bright haze or a cloud
   sea sits below the skyline: a lit rim above the ridge, a colour-shifted rim on it, or a second copy
   of the ridge contour below it are all gradient-weight artefacts, not scene content. Compare against
   the same crop of `fg_stack.fit` before believing any structure the blend shows near the skyline.
7. **Finish** - `uv run astroclaw finish <workdir>` → `<first-frame>_stackN.tif` (16-bit, for editing)
   and `.jpg` (q95, 4:4:4, for JPEG-only libraries) next to the workdir, EXIF copied, orientation
   normal. No stretch or colour work is applied.

## When the mask is wrong

Tune in `src/astroclaw/blend.py`, re-run `blend` (seconds, no re-stack needed):

| Symptom | Where |
|-|-|
| `horizon` line misses a faint far ridge | `HORIZON_EDGE` lower (edge strength relative to the strongest edges) |
| `horizon` line snags on a dust lane or cloud edge in the sky | `HORIZON_EDGE` higher, or `HORIZON_SEARCH` start lower in the frame |
| Horizon sits outside 55-97 % of the frame height | `HORIZON_SEARCH` |
| Lit rim above the ridge, or a copied ridge contour in haze below it | the `core` / `zone` kernels in `blend()`'s gradient block |
| `objects` flags star specks in the sky | `area >= 1500` threshold higher |
| `objects` skips an object cut off by the frame edge | the `at_edge & (area >= 150)` floor lower |
| `objects` misses thin members (antenna rods, branches) | the `8 * sigma` threshold lower |
| Doubled outlines next to bright objects | the `far` dilation, or the `+ 0.25` brightness cutoff lower |

Report which knob moved and why; do not silently change thresholds that were tuned on other data.

## Gotchas

- Siril ignores EXIF orientation; `blend` rotates using the first frame's orientation. Portrait
  frames are the norm for Milky Way shots, so never assume landscape.
- `cv2.imread` DOES apply EXIF orientation, unlike Siril. Compare a single frame with the blend
  without rotating the frame again.
- `exiftool -Orientation=1` writes "Rotate 180"; use `-Orientation#=1`.
- Siril intermediates are ~0.4 GB per frame and deleted after `stack` unless `--keep-work`.
- A JPG stack is non-linear: no photometric colour calibration, and light stretches only
  (`docs/editing.md`).
