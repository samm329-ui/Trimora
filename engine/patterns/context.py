from ..knowledge.context_db import ContextDatabase


class PatternContext:
    def __init__(self, context_db: ContextDatabase):
        self._db = context_db
        self._node_contexts: dict[str, dict] = {}

    CONTEXT_TRIGGERS = {
        "needs_context": [
            "as i said", "as i mentioned", "as i told", "going back to",
            "as we discussed", "like i said", "like i was saying",
            "to go back", "recall that", "coming back to",
        ],
        "creates_context": [
            "what if", "imagine if", "have you ever", "let me tell you",
            "here's the thing", "the thing is", "so here's",
        ],
        "standalone_ok": [
            "the key takeaway", "here's what i", "the point is",
            "so the lesson", "in summary", "to summarize",
            "bottom line", "my point is",
        ],
    }

    def register_segment_context(self, segment_id: str, text: str) -> dict:
        return self._db.register_segment(segment_id, text)

    def get_context(self, segment_id: str):
        return self._db.get_context(segment_id)

    def record_pattern_context(self, pattern_type: str, text: str) -> None:
        ctx = self._db.analyze_context_needs(text)
        self._db.record_pattern_context(pattern_type, ctx)

    def get_pattern_context(self, pattern_type: str):
        return self._db.get_pattern_context(pattern_type)

    def type_count(self) -> int:
        return len(self._node_contexts)

    def get_standalone_probability(self, segment_id: str) -> float:
        return self._db.standalone_probability(segment_id)
