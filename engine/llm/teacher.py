import json
import os
from typing import Optional
from datetime import datetime

from ..config import get_config
from ..data.models import Segment, CompleteClip, ValidClip

from .label_schemas import SegmentLabel, CandidateLabel, RejectionLabel
from .prompts import SEGMENT_LABEL_PROMPT, CANDIDATE_LABEL_PROMPT, REJECTION_PROMPT


def _parse_json_response(text: str) -> Optional[dict]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        start = 1 if text.startswith("```json") else 1
        text = "\n".join(lines[start:])
        if text.endswith("```"):
            text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _query_groq(prompt: str) -> Optional[str]:
    from groq import Groq
    api_key = os.environ.get(get_config().llm.GROQ_API_KEY_ENV if hasattr(get_config().llm, 'GROQ_API_KEY_ENV') else "GROQ_API_KEY") or \
               os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=get_config().llm.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=get_config().llm.GROQ_TEMPERATURE,
        max_tokens=get_config().llm.GROQ_MAX_TOKENS,
    )
    return response.choices[0].message.content if response.choices else None


def _query_gemini(prompt: str) -> Optional[str]:
    import google.generativeai as genai
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text


def _query_llm(prompt: str, provider: str = "groq") -> Optional[str]:
    try:
        if provider == "groq":
            return _query_groq(prompt)
        elif provider == "gemini":
            return _query_gemini(prompt)
        return _query_groq(prompt)
    except Exception:
        return None


class LLMTeacher:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._provider = get_config().llm.PROVIDER

    def _call_llm(self, prompt: str) -> Optional[str]:
        return _query_llm(prompt, provider=self._provider)

    def label_segment(self, segment: Segment) -> Optional[SegmentLabel]:
        cfg = get_config().llm
        if not cfg.USE_LLM:
            return None
        if not self.api_key:
            if self._provider == "groq":
                self.api_key = os.environ.get("GROQ_API_KEY")
            else:
                self.api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not self.api_key:
                return None

        prompt = SEGMENT_LABEL_PROMPT.format(
            text=segment.text,
            duration=segment.duration,
            patterns=", ".join(segment.patterns) if segment.patterns else "none",
            sentiment=f"{segment.sentiment:.2f}",
        )

        try:
            response = self._call_llm(prompt)
            if not response:
                return None
            data = _parse_json_response(response)
            if data is None:
                return None
            return SegmentLabel(
                segment_id=segment.id,
                is_hook=float(data.get("is_hook", 0)),
                hook_strength=float(data.get("hook_strength", 0)),
                is_context=float(data.get("is_context", 0)),
                is_takeaway=float(data.get("is_takeaway", 0)),
                emotion=str(data.get("emotion", "neutral")),
                requires_previous_context=float(data.get("requires_previous_context", 0)),
                creates_new_context=float(data.get("creates_new_context", 0)),
                is_story=float(data.get("is_story", 0)),
                is_opinion=float(data.get("is_opinion", 0)),
                is_fact=float(data.get("is_fact", 0)),
                speaker_confidence=float(data.get("speaker_confidence", 0)),
                saveability=float(data.get("saveability", 0)),
                shareability=float(data.get("shareability", 0)),
                confidence=float(data.get("confidence", 0)),
            )
        except Exception:
            return None

    def label_candidate(self, clip: ValidClip) -> Optional[CandidateLabel]:
        cfg = get_config().llm
        if not cfg.USE_LLM:
            return None
        if not self.api_key:
            if self._provider == "groq":
                self.api_key = os.environ.get("GROQ_API_KEY")
            else:
                self.api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not self.api_key:
                return None

        prompt = CANDIDATE_LABEL_PROMPT.format(
            hook_text=clip.hook_segment.text,
            body_text=" ".join(s.text for s in clip.body_segments) if clip.body_segments else "(no body)",
            ending_text=clip.ending_segment.text,
            duration=clip.total_duration,
            diversity=clip.diversity_score,
        )

        try:
            response = self._call_llm(prompt)
            if not response:
                return None
            data = _parse_json_response(response)
            if data is None:
                return None
            return CandidateLabel(
                candidate_id=f"{clip.hook_segment.id}_{clip.ending_segment.id}",
                story_complete=float(data.get("story_complete", 0)),
                transition_quality=float(data.get("transition_quality", 0)),
                context_missing=float(data.get("context_missing", 0)),
                shareability=float(data.get("shareability", 0)),
                saveability=float(data.get("saveability", 0)),
                hook_strength=float(data.get("hook_strength", 0)),
                ending_strength=float(data.get("ending_strength", 0)),
                emotional_arc_build_up=float(data.get("emotional_arc_build_up", 0)),
                naturalness=float(data.get("naturalness", 0)),
                curiosity_gap=float(data.get("curiosity_gap", 0)),
                confidence=float(data.get("confidence", 0)),
            )
        except Exception:
            return None

    def label_rejection(self, clip: CompleteClip, rules_passed: list[str],
                        rules_failed: list[str], total_score: float) -> Optional[RejectionLabel]:
        cfg = get_config().llm
        if not cfg.USE_LLM:
            return None
        if not self.api_key:
            if self._provider == "groq":
                self.api_key = os.environ.get("GROQ_API_KEY")
            else:
                self.api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not self.api_key:
                return None

        prompt = REJECTION_PROMPT.format(
            hook_text=clip.segments[0].text if clip.segments else "",
            body_text=" ".join(s.text for s in clip.segments[1:-1]) if len(clip.segments) > 2 else "",
            ending_text=clip.segments[-1].text if clip.segments else "",
            rules_passed=", ".join(rules_passed),
            rules_failed=", ".join(rules_failed),
            total_score=total_score,
        )

        try:
            response = self._call_llm(prompt)
            if not response:
                return None
            data = _parse_json_response(response)
            if data is None:
                return None
            return RejectionLabel(
                candidate_id=f"{clip.hook_id}_{clip.ending_id}",
                reason=str(data.get("reason", "missing_context")),
                confidence=float(data.get("confidence", 0)),
            )
        except Exception:
            return None
