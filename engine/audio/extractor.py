import subprocess
import os
from pathlib import Path
from ..config import get_config


def extract_audio(video_path: str, output_path: str = None) -> str:
    cfg = get_config().audio

    if output_path is None:
        stem = Path(video_path).stem
        output_path = os.path.join(
            os.path.dirname(video_path),
            f"{stem}.{cfg.FORMAT}"
        )

    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-q:a", "0", "-map", "a",
        "-ar", str(cfg.SAMPLE_RATE), "-ac", str(cfg.CHANNELS),
        output_path
    ], check=True, capture_output=True)

    return output_path
