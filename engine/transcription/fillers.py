_FILLERS = {
    "um", "uh", "er", "ah", "like", "you know", "actually", "basically",
    "literally", "honestly", "sort of", "kind of", "i mean", "you see",
    "hai", "hain", "tha", "thi", "ho", "hua", "hue",
    "ka", "ki", "ke", "ko", "se", "me", "pe", "aur", "ya"
}


def remove_fillers(text: str) -> str:
    if not text:
        return text

    words = text.split()
    result = []
    skip_next = False

    for i, w in enumerate(words):
        if skip_next:
            skip_next = False
            continue

        lower = w.lower().strip(".,!?;:\"'")

        if lower in _FILLERS:
            continue

        if i + 1 < len(words):
            bigram = f"{lower} {words[i+1].lower().strip('.,!?;:\"\'')}"
            if bigram in _FILLERS:
                skip_next = True
                continue

        result.append(w)

    return " ".join(result)
