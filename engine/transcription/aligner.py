import whisperx
from ..config import get_config


def get_model_for_language(language: str) -> str:
    cfg = get_config().alignment
    return cfg.ALIGN_MODEL_LANGUAGE_MAP.get(language, "en")


def align_segments(segments: list[dict], audio_path: str, language: str) -> list[dict]:
    cfg = get_config().alignment
    model_id = get_model_for_language(language)

    try:
        model, metadata = whisperx.load_align_model(
            language_code=model_id,
            device=cfg.WHISPERX_DEVICE
        )
    except Exception:
        fallback_lang = "en"
        model, metadata = whisperx.load_align_model(
            language_code=fallback_lang,
            device=cfg.WHISPERX_DEVICE
        )

    audio = whisperx.load_audio(audio_path)

    prompt = [
        {"text": s["text"], "start": round(s["start"], 3), "end": round(s["end"], 3)}
        for s in segments
    ]

    result = whisperx.align(
        prompt, model, metadata, audio,
        device=cfg.WHISPERX_DEVICE
    )

    aligned = []
    for seg in result.get("segments", []):
        words = []
        for w in seg.get("words", []):
            words.append({
                "text": w.get("text", ""),
                "start": w.get("start", 0.0),
                "end": w.get("end", 0.0),
                "score": w.get("score", cfg.UNALIGNED_FALLBACK_SCORE)
            })

        aligned.append({
            "text": seg.get("text", ""),
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "words": words
        })

    return aligned
