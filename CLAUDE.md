# astroclaw

Python 3.14 CLI (uv) that stacks nightscape JPG runs with `siril-cli` and blends a star-aligned sky
stack with an unaligned foreground stack. Public repo: no personal paths, hostnames or image data.

- Workflow and verification steps: `.claude/skills/astroclaw/SKILL.md`. Editing concepts: `docs/editing.md`.
- `src/astroclaw/`: `runs.py` (EXIF → runs), `stack.ssf` + `stack.py` (Siril), `blend.py` (masks, layering), `enhance.py` (long-exposure foreground, sky gradient), `restack.py` (ghost-free sky: foreground cut out per aligned frame), `segment.py` (BiRefNet + Grounded-SAM via Replicate), `usage.py` (cost of every paid call → local log + usage ledger),
  `finish.py` (TIFF + EXIF), `cli.py`.
- Every layer (sky, ground, gradient) must share ONE skyline. Separate boundaries a few px apart
  draw parallel contours along a ridge; that bug was fixed three times before this rule existed.
- A mask change is verified by LOOKING at `preview_mask.jpg` and a 1:1 `crop`, never by exit code.
- Source JPGs are only ever copied. Never write to the directory they came from.
- Any new paid model call goes through `usage.record_prediction`, with its hardware and rate pinned in
  `usage.py`. A call that is not costed is a bug.
- Dependencies are pinned exactly in `pyproject.toml`; upgrade one at a time.
