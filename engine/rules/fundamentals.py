FUNDAMENTALS = {
    "hook_grabs_3_seconds": "Hook must grab attention in first 3 seconds",
    "story_complete": "Story must have beginning, middle, end",
    "viewer_gets_value": "Knowledge, emotion, or entertainment delivered",
    "context_feels_complete": "Not a random cut — context preserved",
    "emotional_journey": "Tension to resolution arc",
    "authenticity": "Feels real, not forced",
}


HOOK_RULES = {
    "hook_in_3_seconds": {"description": "Duration <= 8 seconds", "weight": 1.0},
    "creates_curiosity_gap": {"description": "Has question, cliffhanger, contrast", "weight": 0.9},
    "feels_energetic": {"description": "High speech rate + volume", "weight": 0.7},
    "problem_statement": {"description": "Negative sentiment + personal story", "weight": 0.8},
    "promises_value": {"description": "Solution/lesson/secret implied", "weight": 0.8},
}


BODY_RULES = {
    "explains_hook": {"description": "Shares keywords with hook", "weight": 0.9},
    "builds_tension": {"description": "Has emotional progression", "weight": 0.7},
    "appropriate_length": {"description": "15-30 seconds", "weight": 0.8},
    "context_preservation": {"description": "No unresolved references", "weight": 0.9},
}


ENDING_RULES = {
    "resolves_tension": {"description": "Positive sentiment", "weight": 0.9},
    "delivers_takeaway": {"description": "Lesson/practical/summary", "weight": 0.8},
    "appropriate_length": {"description": "5-10 seconds", "weight": 0.7},
    "strong_takeaway": {"description": "Practical + personal + positive (bonus)", "weight": 0.3},
}


STITCHING_RULES = {
    "natural_transition": {"description": "Hook -> Body -> Ending feels smooth", "penalty": "reject"},
    "no_context_gap": {"description": "No missing context between segments", "penalty": "reject"},
    "emotional_arc": {"description": "Peak -> Tension -> Resolution", "bonus": 20},
    "practical_value": {"description": "Contains actionable content", "bonus": 15},
    "relatable": {"description": "Personal pronouns, universal appeal", "bonus": 10},
}


HARD_FILTERS = {
    "duration_45_to_90": "45-90 seconds total",
    "hook_position": "Hook must be first 5 seconds",
    "curiosity_required": "At least one curiosity segment required",
    "value_required": "At least one practicality or strong emotion",
    "speaker_limit": "Max 2 speaker changes",
    "context_safety": "No unresolved context references",
}
