def detect_language(words: list[str]) -> str:
    if not words:
        return "english"

    devanagari = sum(
        1 for w in words
        if any(0x0900 <= ord(c) <= 0x097F for c in w)
    )
    bengali = sum(
        1 for w in words
        if any(0x0980 <= ord(c) <= 0x09FF for c in w)
    )
    latin = sum(
        1 for w in words
        if all(ord(c) < 0x80 for c in w)
    )

    if devanagari > len(words) * 0.1 and latin > len(words) * 0.1:
        return "hinglish"
    elif devanagari > len(words) * 0.3:
        return "hindi"
    elif bengali > len(words) * 0.3:
        return "bengali"
    else:
        return "english"
