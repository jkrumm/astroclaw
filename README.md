# astroclaw

Nightscape stacking that a coding agent can drive end to end. Point it at a folder of camera JPGs;
it finds the runs you shot back to back, stacks each run twice with [Siril](https://siril.org), and
blends a sharp star-aligned sky with a sharp unaligned foreground into one 16-bit TIFF.

Built for a Milky Way night shot in JPG only, where stacking is the one way left to pull the noise
down. Works the same for RAW-converted JPGs.

## Why two stacks

The sky rotates. Aligning frames on the stars makes the sky sharp and smears the ground; not aligning
keeps the ground sharp and trails the stars. astroclaw makes both stacks and joins them along a mask:

| Mode | For | Mask |
|-|-|-|
| `horizon` | a skyline: mountains, sea, a ridge | per-column horizon line |
| `objects` | structures against the sky: towers, trees, buildings | deviation from the smooth sky background |

The mask margin comes from the registration itself: Siril reports how far it moved each frame, and
that distance is exactly how far foreground ghosts reach into the aligned sky.

## Requirements

- macOS or Linux, Python 3.14, [uv](https://docs.astral.sh/uv/)
- `siril-cli` 1.4+ (`brew install siril`)
- `exiftool` (`brew install exiftool`)

## Usage

```bash
uv sync
uv run astroclaw runs ~/Pictures/night                 # 1. list stackable runs
uv run astroclaw prepare ~/Pictures/night ~/astro --run 3
uv run astroclaw stack ~/astro/<first-frame>           # 2. Siril: fg_stack.fit + sky_stack.fit
uv run astroclaw blend ~/astro/<first-frame> horizon \
    --foreground ~/Pictures/night/<long-exposure>.JPG \
    --gradient 2                                       # 3. writes preview_mask.jpg - LOOK at it
uv run astroclaw crop  ~/astro/<first-frame> 5000 1500 #    1:1 seam check
uv run astroclaw finish ~/astro/<first-frame>          # 4. <first-frame>_stackN.tif + .jpg next to the workdir
```

Both `blend` options are optional:

| Option | Does | Why |
|-|-|-|
| `--foreground <jpg>` | uses one long exposure of the same composition for the ground: aligned on the ground (ECC, refuses a shift over 30 px). `--foreground-tone lift` (default) keeps its brightness in dark ground and eases to the stack's tone where the ground is as bright as the de-glowed horizon sky; `frame` / `stack` are the two extremes | a 2-4 min frame has far less noise in dark ground than any stack of short frames; its trailed sky is never used |
| `--segment` | builds the foreground from segmentation (Replicate): BiRefNet for the main object down to single rods, Grounded-SAM for ground level (fence, building, grass, hills), plus threshold blobs cut by the top edge; snapped to the image's edges at full resolution | lattice towers, antenna rods, branches: thinner than any safe dilation and inside the sky's brightness range. Needs `REPLICATE_API_TOKEN`; `REPLICATE_BASE_URL` for a compatible gateway. The mask is cached in the workdir |
| `--gradient 1\|2\|3` | fits the sky glow (light pollution, airglow) as a polynomial in linear light over open sky and subtracts it | flattens the horizon glow before any editing; the Milky Way's own glow is clipped out of the fit |

A run is consecutive frames with identical ISO, aperture and shutter, each starting within
`shutter + 20 s` of the previous one. `prepare` copies; originals are never touched.

`blend` also restacks the sky from the aligned frames with the foreground cut out of each one (in
`horizon` mode only the band above the skyline, in `objects` mode the whole frame), so stars reach the
ridge or the tower instead of stopping at a starless ghost margin. That needs the aligned frames
`stack` keeps (~0.35 GB per frame, converted to ~0.15 GB float16 on first use) until `finish`
deletes them. Without them `blend` falls back to the margin.

## Editing the result

The TIFF is deliberately unedited. What to do next, and which tool does what:
[docs/editing.md](docs/editing.md).

## Agent use

`.claude/skills/astroclaw/SKILL.md` teaches Claude Code the workflow, including the visual checks
that decide whether a blend is done.
