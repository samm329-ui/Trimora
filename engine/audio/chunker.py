import os
from pydub import AudioSegment
from ..data.models import AudioQuality, Chunk
from ..config import get_config


def overlap_chunk(audio_path: str, quality: AudioQuality = None) -> list[Chunk]:
    cfg = get_config().chunking
    qcfg = get_config().quality

    if quality and quality.snr_db < qcfg.SNR_STRICT_THRESHOLD:
        chunk_size = cfg.CHUNK_SIZE_STRICT
        overlap = cfg.CHUNK_OVERLAP_STRICT
    else:
        chunk_size = cfg.CHUNK_SIZE_DEFAULT
        overlap = cfg.CHUNK_OVERLAP_DEFAULT

    audio = AudioSegment.from_file(audio_path)
    total_ms = len(audio)
    chunks = []
    start_ms = 0
    step_ms = (chunk_size - overlap) * 1000
    index = 0

    while start_ms < total_ms:
        end_ms = min(start_ms + chunk_size * 1000, total_ms)
        segment = audio[start_ms:end_ms]
        segment = segment.fade_in(cfg.FADE_IN_MS).fade_out(cfg.FADE_OUT_MS)

        chunk_path = f"{audio_path}.chunk.{index}.{cfg.EXPORT_FORMAT}"
        segment.export(chunk_path, format=cfg.EXPORT_FORMAT)

        chunks.append(Chunk(
            path=chunk_path,
            start_time=start_ms / 1000.0,
            end_time=end_ms / 1000.0,
            index=index
        ))

        start_ms += step_ms
        index += 1

    return chunks
