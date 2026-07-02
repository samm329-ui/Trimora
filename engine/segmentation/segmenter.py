from uuid import uuid4
from ..data.models import Segment
from ..config import get_config


def split_into_atomic_segments(aligned_segments: list[dict]) -> list[Segment]:
    cfg = get_config().segmentation
    atomic = []

    for seg in aligned_segments:
        words = seg.get("words", [])
        if not words:
            text = seg.get("text", "")
            if text:
                atomic.append(Segment(
                    id=str(uuid4()),
                    text=text,
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    duration=seg.get("end", 0.0) - seg.get("start", 0.0),
                    speaker="unknown",
                    words=[]
                ))
            continue

        sentences = split_by_punctuation(words)

        for sentence in sentences:
            duration = sentence["end"] - sentence["start"]
            if duration > cfg.MAX_SEGMENT_DURATION:
                sub_sentences = split_by_time(sentence["words"], max_duration=cfg.MAX_SEGMENT_DURATION)
                for sub in sub_sentences:
                    atomic.append(_make_segment(sub))
            else:
                atomic.append(_make_segment(sentence))

    return merge_short_segments(atomic, min_duration=cfg.MIN_SEGMENT_DURATION)


def _make_segment(sentence: dict) -> Segment:
    return Segment(
        id=str(uuid4()),
        text=sentence["text"],
        start=sentence["start"],
        end=sentence["end"],
        duration=sentence["end"] - sentence["start"],
        speaker="unknown",
        words=sentence.get("words", [])
    )


def split_by_punctuation(words: list[dict]) -> list[dict]:
    sentences = []
    current = []
    punc_set = get_config().segmentation.PUNCTUATION_BOUNDARIES

    for w in words:
        current.append(w)
        text = w.get("text", "")
        if text and text[-1] in punc_set:
            sentence = _build_sentence(current)
            sentences.append(sentence)
            current = []

    if current:
        sentence = _build_sentence(current)
        sentences.append(sentence)

    return sentences


def _build_sentence(words: list[dict]) -> dict:
    if not words:
        return {"text": "", "start": 0.0, "end": 0.0, "words": []}

    return {
        "text": " ".join(w.get("text", "") for w in words),
        "start": words[0].get("start", 0.0),
        "end": words[-1].get("end", 0.0),
        "words": words
    }


def split_by_time(words: list[dict], max_duration: float) -> list[dict]:
    if not words:
        return []

    sentences = []
    current = []
    start_time = words[0].get("start", 0.0)

    for w in words:
        w_end = w.get("end", 0.0)
        if w_end - start_time > max_duration and current:
            sentences.append(_build_sentence(current))
            current = [w]
            start_time = w.get("start", 0.0)
        else:
            current.append(w)

    if current:
        sentences.append(_build_sentence(current))

    return sentences


def merge_short_segments(segments: list[Segment], min_duration: float = 2.0) -> list[Segment]:
    if not segments:
        return []

    merged = []
    buffer = None

    for seg in segments:
        if buffer is not None:
            buffer = Segment(
                id=str(uuid4()),
                text=buffer.text + " " + seg.text,
                start=buffer.start,
                end=seg.end,
                duration=seg.end - buffer.start,
                speaker=buffer.speaker or seg.speaker,
                words=buffer.words + seg.words
            )
            if buffer.duration >= min_duration:
                merged.append(buffer)
                buffer = None
        elif seg.duration < min_duration:
            buffer = seg
        else:
            merged.append(seg)

    if buffer is not None:
        if merged:
            last = merged[-1]
            merged[-1] = Segment(
                id=str(uuid4()),
                text=last.text + " " + buffer.text,
                start=last.start,
                end=buffer.end,
                duration=buffer.end - last.start,
                speaker=last.speaker or buffer.speaker,
                words=last.words + buffer.words
            )
        else:
            merged.append(buffer)

    return merged
