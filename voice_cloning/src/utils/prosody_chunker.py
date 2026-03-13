import re


_NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "million",
    "billion",
    "trillion",
}

_GLUE_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "than",
    "that",
    "the",
    "to",
    "with",
}


def _normalize_word(word: str) -> str:
    return re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", word).lower()


def _is_numericish(word: str) -> bool:
    normalized = _normalize_word(word)
    if not normalized:
        return False
    if normalized in _NUMBER_WORDS:
        return True
    return bool(re.fullmatch(r"\d+(?:[.,:/-]\d+)*", normalized))


def _has_strong_break(word: str) -> bool:
    return bool(re.search(r"[.?!][\"')\]]*$", word))


def _has_soft_break(word: str) -> bool:
    return bool(re.search(r"[,;:][\"')\]]*$", word))


def prosody_split(
    text: str,
    max_chars: int = 150,
    first_chunk_max_chars: int = 120,
    max_words: int | None = None,
) -> list[str]:
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

    def _find_soft_break(max_len: int, word_limit: int | None) -> int:
        if not current_words:
            return 0
        total = 0
        best_idx = -1
        best_score = float("-inf")
        for idx, w in enumerate(current_words):
            if word_limit is not None and idx + 1 > word_limit:
                break
            add = len(w) if idx == 0 else len(w) + 1
            total += add
            if total > max_len:
                break
            score = float(idx + 1) * 0.12
            prev_norm = _normalize_word(w)
            next_word = current_words[idx + 1] if idx + 1 < len(current_words) else ""
            next_norm = _normalize_word(next_word)
            if _has_strong_break(w):
                score += 100.0
            elif _has_soft_break(w):
                score += 60.0
            if not (_has_strong_break(w) or _has_soft_break(w)) and (_is_numericish(w) or _is_numericish(next_word)):
                score -= 90.0
            if next_norm in _GLUE_WORDS:
                score -= 24.0
            if prev_norm in _GLUE_WORDS:
                score -= 10.0
            if next_word and next_word[:1].islower() and not (_has_strong_break(w) or _has_soft_break(w)):
                score -= 6.0
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx >= 0:
            return best_idx + 1

        # Fallback: split near limit on a word boundary.
        total = 0
        for idx, w in enumerate(current_words):
            if word_limit is not None and idx + 1 > word_limit:
                return max(1, idx)
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
        word_limit = max_words if max_words and max_words > 0 else None

        if _has_strong_break(word):
            _emit()
            continue

        if len(current_text) >= limit or (word_limit is not None and len(current_words) >= word_limit):
            split_at = _find_soft_break(limit, word_limit)
            _emit(split_at)

    _emit()
    return chunks
