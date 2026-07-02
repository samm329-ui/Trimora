import os
from groq import Groq
from ..config import get_config


_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get(get_config().transcription.GROQ_API_KEY_ENV)
        if not api_key:
            raise ValueError(f"{get_config().transcription.GROQ_API_KEY_ENV} not set")
        _client = Groq(api_key=api_key)
    return _client


def transcribe_chunk(audio_path: str) -> str:
    cfg = get_config().transcription
    client = _get_client()

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), f.read()),
            model=cfg.GROQ_MODEL,
            response_format=cfg.RESPONSE_FORMAT,
            temperature=cfg.TEMPERATURE,
        )

    return response.strip() if hasattr(response, 'strip') else str(response).strip()
