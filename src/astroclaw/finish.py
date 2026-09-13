"""Write the blend as a 16-bit TIFF and a JPEG, both carrying the first frame's EXIF."""
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import tifffile

TRIM_PX = 12  # stack edges carry interpolation and rejection artefacts


def finish(workdir: Path, dest: Path, jpeg: bool = True) -> list[Path]:
    """16-bit TIFF for editing with headroom, plus a JPEG master for JPEG-only libraries."""
    out = np.clip(np.load(workdir / "blend.npy")[TRIM_PX:-TRIM_PX, TRIM_PX:-TRIM_PX], 0, 1)
    frames = sorted((workdir / "src").glob("*.[jJ][pP][gG]"))
    first = frames[0]
    stem = f"{first.stem}_stack{len(frames)}"
    written = [dest / f"{stem}.tif"]
    tifffile.imwrite(written[0], np.ascontiguousarray((out * 65535).astype(np.uint16)),
                     photometric="rgb", compression="zlib")
    if jpeg:
        written.append(dest / f"{stem}.jpg")
        bgr = cv2.cvtColor((out * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(written[1]), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
                                           cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])

    description = f"astroclaw stack of {len(frames)} frames {frames[0].stem} .. {frames[-1].stem}"
    for target in written:
        subprocess.run(["exiftool", "-q", "-overwrite_original", "-TagsFromFile", str(first), "-all:all",
                        f"-ImageDescription={description}", str(target)], check=True)
        # pixels are already rotated; the numeric form matters, "Orientation=1" writes "Rotate 180"
        subprocess.run(["exiftool", "-q", "-overwrite_original", "-Orientation#=1", "-IFD1:Orientation#=",
                        "-ThumbnailImage=", str(target)], check=True)
    (workdir / "blend.npy").unlink()
    shutil.rmtree(workdir / "work", ignore_errors=True)   # aligned frames kept for the skyline restack
    return written
