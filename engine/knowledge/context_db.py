import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional


class ContextDatabase:
    def __init__(self):
        self._patterns: dict[str, dict] = {}
        self._segment_context: dict[str, dict] = {}

    CONTEXT_KEYWORDS = {
        "needs_context": [
            "as i said", "as i mentioned", "as i told", "going back to",
            "as we discussed", "as we talked", "earlier i", "earlier we",
            "like i said", "like i was saying", "to go back", "recall that",
            "recall when", "coming back to", "back to what i was",
        ],
        "creates_context": [
            "what if", "imagine if", "have you ever", "let me tell you",
            "here's the thing", "the thing is", "so here's",
        ],
        "standalone_ok": [
            "the key takeaway", "here's what i", "the point is",
            "so the lesson", "in summary", "to summarize",
            "bottom line", "my point is", "here's the bottom line",
        ],
    }

    def analyze_context_needs(self, text: str) -> dict:
        text_lower = text.lower()
        needs_context = any(kw in text_lower for kw in self.CONTEXT_KEYWORDS["needs_context"])
        creates_context = any(kw in text_lower for kw in self.CONTEXT_KEYWORDS["creates_context"])
        standalone_ok = any(kw in text_lower for kw in self.CONTEXT_KEYWORDS["standalone_ok"])

        if standalone_ok:
            return {"requires_previous_context": 0.1, "standalone_probability": 0.9, "creates_new_context": 0.5}
        if creates_context:
            return {"requires_previous_context": 0.2, "standalone_probability": 0.8, "creates_new_context": 0.8}
        if needs_context:
            return {"requires_previous_context": 0.9, "standalone_probability": 0.2, "creates_new_context": 0.1}
        return {"requires_previous_context": 0.3, "standalone_probability": 0.7, "creates_new_context": 0.4}

    def register_segment(self, segment_id: str, text: str,
                          patterns: Optional[list[str]] = None) -> dict:
        ctx = self.analyze_context_needs(text)
        ctx["segment_id"] = segment_id
        ctx["text_snippet"] = text[:100]
        ctx["patterns"] = patterns or []
        self._segment_context[segment_id] = ctx
        return ctx

    def requires_context(self, segment_id: str) -> float:
        ctx = self._segment_context.get(segment_id)
        if ctx is None:
            return 0.5
        return ctx.get("requires_previous_context", 0.5)

    def standalone_probability(self, segment_id: str) -> float:
        ctx = self._segment_context.get(segment_id)
        if ctx is None:
            return 0.5
        return ctx.get("standalone_probability", 0.5)

    def get_context(self, segment_id: str) -> Optional[dict]:
        return self._segment_context.get(segment_id)

    def record_pattern_context(self, pattern_type: str, context_info: dict) -> None:
        if pattern_type in self._patterns:
            p = self._patterns[pattern_type]
            n = p["observations"] + 1
            p["observations"] = n
            for key in ["requires_previous_context", "standalone_probability", "creates_new_context"]:
                if key in context_info:
                    p[key] = (p.get(key, 0) * (n - 1) + context_info[key]) / n
        else:
            self._patterns[pattern_type] = {
                "pattern_type": pattern_type,
                "observations": 1,
                "requires_previous_context": context_info.get("requires_previous_context", 0.5),
                "standalone_probability": context_info.get("standalone_probability", 0.5),
                "creates_new_context": context_info.get("creates_new_context", 0.5),
            }

    def get_pattern_context(self, pattern_type: str) -> Optional[dict]:
        return self._patterns.get(pattern_type)

    def get_all_segments_needing_context(self, threshold: float = 0.7) -> list[str]:
        return [
            sid for sid, ctx in self._segment_context.items()
            if ctx.get("requires_previous_context", 0) >= threshold
        ]

    def summary(self) -> dict:
        return {
            "segments_tracked": len(self._segment_context),
            "patterns_learned": len(self._patterns),
            "segments_needing_context": len(self.get_all_segments_needing_context()),
        }
