from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = None


def _get_analyzer() -> SentimentIntensityAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def compute_sentiment(text: str) -> float:
    if not text or not text.strip():
        return 0.0
    scores = _get_analyzer().polarity_scores(text)
    return scores["compound"]
