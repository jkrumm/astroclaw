"""Write the blend as a 16-bit TIFF carrying the first frame's EXIF."""
import subprocess
from pathlib import Path

import numpy as np
import tifffile

TRIM_PX = 12  # stack edges carry interpolation and rejection artefacts


def finish(workdir: Path, dest: Path) -> Path:
    out = np.load(workdir / "blend.npy")
    frames = sorted((workdir / "src").glob("*.[jJ][pP][gG]"))
    first = frames[0]
    target = dest / f"{first.stem}_stack{len(frames)}.tif"

    pixels = (np.clip(out[TRIM_PX:-TRIM_PX, TRIM_PX:-TRIM_PX], 0, 1) * 65535).astype(np.uint16)
    tifffile.imwrite(target, np.ascontiguousarray(pixels), photometric="rgb", compression="zlib")

    description = f"astroclaw stack of {len(frames)} frames {frames[0].stem} .. {frames[-1].stem}"
    subprocess.run(["exiftool", "-q", "-overwrite_original", "-TagsFromFile", str(first), "-all:all",
                    f"-ImageDescription={description}", str(target)], check=True)
    # pixels are already rotated; the numeric form matters, "Orientation=1" writes "Rotate 180"
    subprocess.run(["exiftool", "-q", "-overwrite_original", "-Orientation#=1", "-IFD1:Orientation#=",
                    str(target)], check=True)
    (workdir / "blend.npy").unlink()
    return target
