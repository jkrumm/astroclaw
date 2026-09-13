"""Find stackable runs: consecutive frames with identical exposure settings, shot back to back."""
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

GAP_SLACK_S = 20  # interval between frames may exceed the exposure by this much (write time, interval timer)


@dataclass
class Frame:
    path: Path
    taken: datetime
    iso: int
    aperture: float
    shutter: float


@dataclass
class Run:
    frames: list[Frame]

    @property
    def settings(self) -> str:
        f = self.frames[0]
        return f"ISO {f.iso}, f/{f.aperture:g}, {f.shutter:g}s"

    @property
    def span(self) -> str:
        return f"{self.frames[0].taken:%Y-%m-%d %H:%M:%S} - {self.frames[-1].taken:%H:%M:%S}"


def read_frames(directory: Path) -> list[Frame]:
    out = subprocess.run(
        ["exiftool", "-j", "-n", "-ext", "jpg", "-DateTimeOriginal", "-ISO", "-FNumber", "-ExposureTime", str(directory)],
        capture_output=True, text=True, check=False,
    )
    if not out.stdout.strip():
        return []
    frames = []
    for row in json.loads(out.stdout):
        if not all(k in row for k in ("DateTimeOriginal", "ISO", "FNumber", "ExposureTime")):
            continue
        frames.append(Frame(
            path=Path(row["SourceFile"]),
            taken=datetime.strptime(str(row["DateTimeOriginal"]), "%Y:%m:%d %H:%M:%S"),
            iso=int(row["ISO"]),
            aperture=float(row["FNumber"]),
            shutter=float(row["ExposureTime"]),
        ))
    return sorted(frames, key=lambda f: (f.taken, f.path.name))


def find_runs(frames: list[Frame], min_shutter: float, min_frames: int) -> list[Run]:
    runs: list[list[Frame]] = []
    for frame in (f for f in frames if f.shutter >= min_shutter):
        prev = runs[-1][-1] if runs else None
        same = prev and (prev.iso, prev.aperture, prev.shutter) == (frame.iso, frame.aperture, frame.shutter)
        close = prev and (frame.taken - prev.taken).total_seconds() <= frame.shutter + GAP_SLACK_S
        if same and close:
            runs[-1].append(frame)
        else:
            runs.append([frame])
    return [Run(r) for r in runs if len(r) >= min_frames]
