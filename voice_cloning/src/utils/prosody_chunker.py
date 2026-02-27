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
    current_words: list[str] = []

    first_limit = max(20, int(first_chunk_max_chars))
    regular_limit = max(first_limit, int(max_chars))

    def _emit(count: int | None = None) -> None:
        nonlocal current_words
        if not current_words:
            return
        if count is None:
            part = current_words
            current_words = []
        else:
            part = current_words[:count]
            current_words = current_words[count:]
        text_part = " ".join(part).strip()
        if text_part:
            chunks.append(text_part)

    def _find_soft_break(max_len: int) -> int:
        if not current_words:
            return 0
        total = 0
        last_soft = -1
        for idx, w in enumerate(current_words):
            add = len(w) if idx == 0 else len(w) + 1
            total += add
            if total > max_len:
                break
            if re.search(r"[,;:\-]$", w):
                last_soft = idx
        if last_soft >= 0:
            return last_soft + 1
        # Fallback: split near limit on word boundary.
        total = 0
        for idx, w in enumerate(current_words):
            add = len(w) if idx == 0 else len(w) + 1
            if total + add > max_len:
                return max(1, idx)
            total += add
        return len(current_words)

    for word in words:
        current_words.append(word)
        current_text = " ".join(current_words)
        is_first_chunk = len(chunks) == 0
        limit = first_limit if is_first_chunk else regular_limit

        if re.search(r"[.?!]$", word):
            _emit()
            continue

        if len(current_text) >= limit:
            split_at = _find_soft_break(limit)
            _emit(split_at)

    _emit()
    return chunks
