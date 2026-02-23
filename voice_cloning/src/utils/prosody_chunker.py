import re


def prosody_split(text: str, max_chars: int = 150, first_chunk_max_chars: int = 120) -> list[str]:
    """
    Split text on natural prosody boundaries.
    Priority:
      1) Strong punctuation (.?!)
    """
    if not text:
        return []

    text = re.sub(r"\s+", " ", text).strip()
    words = text.split(" ")
    chunks: list[str] = []
    current = ""

    for idx, word in enumerate(words):
        candidate = f"{current} {word}".strip() if current else word

        current = candidate

        # Force a short first chunk for faster time-to-first-audio.
        if not chunks and len(current) >= first_chunk_max_chars:
            chunks.append(current.strip())
            current = ""
            continue

        # Strong punctuation: split immediately.
        if re.search(r"[.?!]$", word):
            chunks.append(current.strip())
            current = ""
            continue

    if current:
        chunks.append(current.strip())

    return chunks
