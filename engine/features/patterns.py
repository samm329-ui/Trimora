import re

PATTERNS = {
    "curiosity": [
        (r"(?i)what if", "what_if"),
        (r"(?i)no one (knows|talks|told)", "unknown"),
        (r"(?i)the biggest (mistake|problem|secret)", "biggest"),
        (r"(?i)I didn't expect", "unexpected"),
        (r"(?i)imagine if", "imagine"),
        (r"(?i)I learned?", "learned"),
        (r"(?i)\?\s*$", "question"),
    ],
    "story": [
        (r"(?i)\bthen\b", "then"),
        (r"(?i)\bafter\b", "after"),
        (r"(?i)\bbefore\b", "before"),
        (r"(?i)\bsuddenly\b", "suddenly"),
        (r"(?i)\bfinally\b", "finally"),
        (r"(?i)\bone day\b", "one_day"),
        (r"(?i)\blater\b", "later"),
        (r"(?i)\bbecause\b", "because"),
    ],
    "practicality": [
        (r"(?i)(steps?|step \d)", "steps"),
        (r"(?i)(framework|system)", "framework"),
        (r"(?i)(tips?|trick)", "tip"),
        (r"(?i)(rules?|rule \d)", "rule"),
        (r"(?i)(checklist|template)", "checklist"),
        (r"(?i)(lesson|takeaway)", "lesson"),
        (r"(?i)(method|formula)", "method"),
    ],
    "shareability": [
        (r"\d+(\.\d+)?%", "percentage"),
        (r"\$\d+(\.\d+)?[kKmMbB]?", "money"),
        (r"\d+\s*(days?|weeks?|months?|years?)", "timeframe"),
        (r"(?i)(surprising|shocking|unbelievable)", "surprising"),
        (r"(?i)(contrarian|unpopular|hot take)", "contrarian"),
        (r"(?i)(the best|the worst|the only)", "extreme"),
        (r"(?i)(according to|study shows|research says)", "authority"),
    ],
    "contrast": [
        (r"(?i)\bbut\b", "but"),
        (r"(?i)\bhowever\b", "however"),
        (r"(?i)\bactually\b", "actually"),
        (r"(?i)\bsurprisingly\b", "surprisingly"),
        (r"(?i)\byet\b", "yet"),
        (r"(?i)\b(lekin|par|magar)\b", "contrast_hi"),
    ],
    "relatability": [
        (r"\b(I|we|my|our)\b", "personal"),
        (r"(?i)(everyone|nobody|anyone)", "universal"),
        (r"(?i)(you know|think about|imagine)", "engaging"),
        (r"(?i)(we've all|all of us|everyone has)", "shared_experience"),
        (r"(?i)(it's like|feels like|that moment)", "empathy"),
    ],
    "takeaway": [
        (r"(?i)(key lesson|main takeaway)", "key_lesson"),
        (r"(?i)(do this|try this|start with)", "action"),
        (r"(?i)(remember|never forget)", "remember"),
        (r"(?i)(the point is|what matters)", "point"),
        (r"(?i)(if you (take|learn|get) (anything|something))", "call_to_action"),
        (r"(?i)(here's (the thing|what I|what you))", "here_is"),
        (r"(?i)(that's why|that's how|that's what)", "conclusion"),
    ],
    "power_words": [
        (r"(?i)\bsecret\b", "secret"),
        (r"(?i)\bshocking\b", "shocking"),
        (r"(?i)\bnever\b", "never"),
        (r"(?i)\balways\b", "always"),
        (r"(?i)\bimpossible\b", "impossible"),
        (r"(?i)\bguaranteed?\b", "guaranteed"),
        (r"(?i)\bchanged?\s+my\s+life\b", "life_changing"),
    ]
}


def match_patterns(text: str) -> list[str]:
    if not text:
        return []
    matched = []
    for category, pattern_list in PATTERNS.items():
        for pattern, name in pattern_list:
            if re.search(pattern, text):
                matched.append(name)
    return matched
