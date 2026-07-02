from ..data.models import ProcessedChunk


def merge_chunks(processed_chunks: list[ProcessedChunk], overlap_word_count: int = 10) -> list[dict]:
    sorted_chunks = sorted(processed_chunks, key=lambda c: c.index)
    merged = []

    for chunk in sorted_chunks:
        text = chunk.text.strip()
        if not text:
            continue

        if merged:
            last_text = merged[-1]["text"]
            last_words = last_text.split()[-overlap_word_count:]
            first_words = text.split()[:overlap_word_count]

            for i in range(len(first_words), 0, -1):
                candidate = " ".join(first_words[:i]).lower()
                last_slice = " ".join(last_words[-i:]).lower()
                if candidate == last_slice and len(candidate) > 2:
                    text = " ".join(text.split()[i:])
                    break

        if text.strip():
            merged.append({
                "text": text,
                "start": chunk.start_time,
                "end": chunk.end_time
            })

    return merged
