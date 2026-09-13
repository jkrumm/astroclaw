"""astroclaw - stack nightscape JPG runs with Siril and blend sky and foreground.

  astroclaw runs <jpg-dir>                        list stackable runs
  astroclaw prepare <jpg-dir> <out-root> [--run N] copy runs into <out-root>/<first-frame>/src
  astroclaw stack <workdir>                        Siril: fg_stack.fit + sky_stack.fit
  astroclaw blend <workdir> horizon|objects        blend + preview_mask.jpg / preview_blend.jpg
  astroclaw crop <workdir> <y> <x>                 1:1 crop of the blend for seam checks
  astroclaw finish <workdir> [--dest DIR]          16-bit TIFF with EXIF
"""
import argparse
import shutil
from pathlib import Path

from astroclaw.runs import find_runs, read_frames


def main() -> None:
    p = argparse.ArgumentParser(prog="astroclaw", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("runs")
    r.add_argument("jpg_dir", type=Path)
    r.add_argument("--min-shutter", type=float, default=5.0)
    r.add_argument("--min-frames", type=int, default=4)

    pr = sub.add_parser("prepare")
    pr.add_argument("jpg_dir", type=Path)
    pr.add_argument("out_root", type=Path)
    pr.add_argument("--run", type=int, action="append", help="run number from `runs` (repeatable); default all")
    pr.add_argument("--min-shutter", type=float, default=5.0)
    pr.add_argument("--min-frames", type=int, default=4)

    s = sub.add_parser("stack")
    s.add_argument("workdir", type=Path)
    s.add_argument("--keep-work", action="store_true")

    b = sub.add_parser("blend")
    b.add_argument("workdir", type=Path)
    b.add_argument("mode", choices=["horizon", "objects"])

    c = sub.add_parser("crop")
    c.add_argument("workdir", type=Path)
    c.add_argument("y", type=int)
    c.add_argument("x", type=int)
    c.add_argument("--size", type=int, default=1000)

    f = sub.add_parser("finish")
    f.add_argument("workdir", type=Path)
    f.add_argument("--dest", type=Path)

    a = p.parse_args()
    if a.cmd in ("runs", "prepare"):
        runs = find_runs(read_frames(a.jpg_dir), a.min_shutter, a.min_frames)
        if a.cmd == "runs":
            for i, run in enumerate(runs, 1):
                print(f"{i}: {len(run.frames):3d} frames  {run.settings:24s} {run.span}  "
                      f"{run.frames[0].path.name} .. {run.frames[-1].path.name}")
            return
        for i, run in enumerate(runs, 1):
            if a.run and i not in a.run:
                continue
            src = a.out_root / run.frames[0].path.stem / "src"
            src.mkdir(parents=True, exist_ok=True)
            for frame in run.frames:
                shutil.copy2(frame.path, src / frame.path.name)   # copies: originals are never touched
            print(src.parent)
    elif a.cmd == "stack":
        from astroclaw.stack import stack
        print(stack(a.workdir, a.keep_work))
    elif a.cmd == "blend":
        from astroclaw.blend import blend
        print(blend(a.workdir, a.mode))
    elif a.cmd == "crop":
        from astroclaw.blend import crop_preview
        print(crop_preview(a.workdir, a.y, a.x, a.size))
    elif a.cmd == "finish":
        from astroclaw.finish import finish
        print(finish(a.workdir, a.dest or a.workdir.parent))
