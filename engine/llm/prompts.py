SEGMENT_LABEL_PROMPT = """You are an expert video editor analyzing a transcript segment.
Rate the following segment on a scale of 0.0 to 1.0 for each dimension.

Segment Text: "{text}"
Duration: {duration}s
Patterns Detected: {patterns}
Sentiment: {sentiment}

Dimensions to rate:
- is_hook: Does this sound like an opening hook that grabs attention?
- hook_strength: How strong is it as a hook?
- is_context: Is this segment providing background context?
- is_takeaway: Does this contain a key takeaway or lesson?
- emotion: Categorize the emotion (neutral, curiosity, urgency, surprise, humor, inspiration, anger, sadness, excitement, fear, disgust, trust, anticipation, joy)
- requires_previous_context: Does the viewer need prior context to understand this?
- creates_new_context: Does this introduce a new idea or topic?
- is_story: Is this part of a narrative/story?
- is_opinion: Is this the speaker's opinion?
- is_fact: Is this a factual statement?
- speaker_confidence: How confident does the speaker sound?
- saveability: Would viewers save/bookmark this?
- shareability: Would viewers share this with others?
- confidence: How confident are you in these ratings?

Return ONLY a JSON object with these 14 fields (13 numeric, 1 string for emotion)."""

CANDIDATE_LABEL_PROMPT = """You are an expert video editor evaluating a short video clip.
Rate the following clip on a scale of 0.0 to 1.0 for each dimension.

Hook Text: "{hook_text}"
Body Text: "{body_text}"
Ending Text: "{ending_text}"
Total Duration: {duration}s
Diversity Score: {diversity:.2f}

Dimensions to rate:
- story_complete: Does this feel like a complete story/thought?
- transition_quality: How smooth are transitions between segments?
- context_missing: Is important context missing for understanding?
- shareability: Would viewers share this clip?
- saveability: Would viewers save this clip?
- hook_strength: How strong is the opening hook?
- ending_strength: How strong is the closing/ending?
- emotional_arc_build_up: Is there an emotional journey or build up?
- naturalness: Does the clip flow naturally?
- curiosity_gap: Does this create a curiosity gap for viewers?
- confidence: How confident are you in these ratings?

Return ONLY a JSON object with these 11 numeric fields."""

REJECTION_PROMPT = """You are an expert video editor reviewing a rejected clip.
Analyze why this clip was rejected and confirm the rejection reason.

Hook Text: "{hook_text}"
Body Text: "{body_text}"
Ending Text: "{ending_text}"
Rules Passed: {rules_passed}
Rules Failed: {rules_failed}
Total Score: {total_score:.2f}

Valid rejection reasons:
- missing_context: Clip assumes viewer knows prior information
- weak_hook: Opening fails to grab attention
- weak_ending: Closing is unsatisfying
- no_emotional_arc: No emotional journey through the clip
- transition_too_abrupt: Segments don't flow smoothly
- story_incomplete: Clip feels like it cuts off mid-thought
- low_naturalness: Clip feels forced or unnatural
- low_shareability: Content is not shareable

Return ONLY a JSON object with fields:
- reason (string, one of the valid reasons)
- confidence (float, 0.0-1.0)
- explanation (string, brief explanation)"""
