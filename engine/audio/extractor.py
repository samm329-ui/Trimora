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
        "-map", "a",
        "-ar", str(cfg.SAMPLE_RATE), "-ac", str(cfg.CHANNELS),
        "-c:a", cfg.FFMPEG_AUDIO_CODEC, "-b:a", cfg.AUDIO_BITRATE,
        output_path
    ], check=True, capture_output=True)

    return output_path
