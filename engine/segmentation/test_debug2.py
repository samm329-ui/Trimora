import sys; sys.path.insert(0, "d:\\openshorts")
from engine.segmentation.segmenter import split_into_atomic_segments, split_by_punctuation

aligned = [
    {
        "text": "What if I told you? Then this happened.",
        "start": 0.0, "end": 5.0,
        "words": [
            {"text": "What", "start": 0.0, "end": 0.3},
            {"text": "if", "start": 0.3, "end": 0.5},
            {"text": "I", "start": 0.5, "end": 0.7},
            {"text": "told", "start": 0.7, "end": 0.9},
            {"text": "you", "start": 0.9, "end": 1.1},
            {"text": "?", "start": 1.1, "end": 1.2},
            {"text": "Then", "start": 1.3, "end": 1.5},
            {"text": "this", "start": 1.5, "end": 1.7},
            {"text": "happened", "start": 1.7, "end": 2.0},
            {"text": ".", "start": 2.0, "end": 2.1},
        ]
    }
]

# Test split_by_punctuation directly on the words
print("Direct split_by_punctuation:")
punc_result = split_by_punctuation(aligned[0]["words"])
for p in punc_result:
    print("  text=%r" % p["text"])

# Test full pipeline
print("Full split_into_atomic_segments:")
atoms = split_into_atomic_segments(aligned)
print("Count:", len(atoms))
for a in atoms:
    print("  id=%s text=%r dur=%.2f" % (a.id[:8], a.text, a.duration))
