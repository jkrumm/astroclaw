"""Run the Siril script: one unaligned foreground stack, one star-aligned sky stack."""
import json
import shutil
import subprocess
from importlib.resources import files
from pathlib import Path


def stack(workdir: Path, keep_work: bool = True) -> dict:
    src = workdir / "src"
    if not any(src.glob("*.[jJ][pP][gG]")):
        raise SystemExit(f"no JPGs in {src}")
    if shutil.which("siril-cli") is None:
        raise SystemExit("siril-cli not found - brew install siril")

    script = workdir / "stack.ssf"
    script.write_text(files("astroclaw").joinpath("stack.ssf").read_text())
    log = workdir / "siril.log"
    with log.open("w") as fh:
        proc = subprocess.run(["siril-cli", "-d", str(workdir), "-s", str(script)], stdout=fh, stderr=subprocess.STDOUT)
    script.unlink()

    text = log.read_text()
    registered = _last_int(text, "registered.")
    if proc.returncode != 0 or not (workdir / "sky_stack.fit").exists() or not (workdir / "fg_stack.fit").exists():
        raise SystemExit(f"siril failed (exit {proc.returncode}), see {log}")

    info = {
        "frames": len(list(src.glob("*.[jJ][pP][gG]"))),
        "registered": registered,
        "max_shift_px": _max_shift(workdir / "work" / "f_.seq"),
    }
    (workdir / "registration.json").write_text(json.dumps(info, indent=2))
    if keep_work:
        for f in (workdir / "work").glob("f_*.fit"):   # only the aligned frames are needed later
            f.unlink()
    else:
        shutil.rmtree(workdir / "work")  # ~0.4 GB per frame, fully regenerable
    return info


def _max_shift(seq: Path) -> float:
    """Largest translation Siril applied to any frame: sets how far foreground ghosts reach in the sky stack."""
    shift = 0.0
    for line in seq.read_text().splitlines():
        parts = line.split()
        if line.startswith("R") and "H" in parts:
            h = [float(v) for v in parts[parts.index("H") + 1:]]
            shift = max(shift, (h[2] ** 2 + h[5] ** 2) ** 0.5)
    return round(shift, 1)


def _last_int(text: str, suffix: str) -> int | None:
    for line in reversed(text.splitlines()):
        if line.rstrip().endswith(suffix):
            digits = [w for w in line.replace(",", " ").split() if w.isdigit()]
            return int(digits[-1]) if digits else None
    return None
