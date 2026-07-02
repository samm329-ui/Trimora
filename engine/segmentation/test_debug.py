import sys; sys.path.insert(0, 'd:\\openshorts')
from engine.segmentation.segmenter import split_by_punctuation
from engine.config import get_config

cfg = get_config()
print("PUNCTUATION:", cfg.segmentation.PUNCTUATION_BOUNDARIES)

words = [
    {"text": "What", "start": 0.0, "end": 0.3},
    {"text": "?", "start": 1.1, "end": 1.2},
    {"text": "Then", "start": 1.3, "end": 1.5},
    {"text": ".", "start": 2.0, "end": 2.1},
]

for w in words:
    text = w["text"]
    last = text[-1] if text else ""
    in_set = last in cfg.segmentation.PUNCTUATION_BOUNDARIES
    print("  word=%r last=%r in_set=%s" % (text, last, in_set))

result = split_by_punctuation(words)
print("Result count:", len(result))
for s in result:
    print("  text=%r" % s["text"])
