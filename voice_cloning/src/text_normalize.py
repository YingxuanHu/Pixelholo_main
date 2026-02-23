import re
from typing import Callable

from num2words import num2words

try:
    from nemo_text_processing.text_normalization.normalize import Normalizer  # type: ignore
except Exception:
    Normalizer = None


_CURRENCY_RE = re.compile(r"(?P<sign>[$£€])\s?(?P<amount>\d[\d,]*)(?:\.(?P<cents>\d{1,2}))?")
_NUMBER_RE = re.compile(r"\d[\d,]*")

_ACRONYMS = {
    "LLM": "L L M",
    "API": "A P I",
    "AWS": "A W S",
    "GPU": "G P U",
    "AI": "A I",
    "SQL": "Sequel",
}

_ABBREVIATIONS = {
    "Mr.": "Mister",
    "Mrs.": "Missus",
    "Dr.": "Doctor",
    "St.": "Street",
    "vs.": "versus",
}

_nemo_normalizer = None


def _get_nemo_normalizer():
    global _nemo_normalizer
    if _nemo_normalizer is None and Normalizer is not None:
        _nemo_normalizer = Normalizer(input_case="cased", lang="en")
    return _nemo_normalizer


def warmup_text_normalizer() -> None:
    _get_nemo_normalizer()


def _strip_commas(value: str) -> str:
    return value.replace(",", "")


def _to_words(value: str, *, to_year: bool = False) -> str:
    number = int(_strip_commas(value))
    if to_year:
        return num2words(number, to="year")
    return num2words(number)


def _replace_currency(match: re.Match) -> str:
    sign = match.group("sign")
    amount = match.group("amount") or "0"
    cents = match.group("cents")
    currency_word = {
        "$": "dollars",
        "£": "pounds",
        "€": "euros",
    }.get(sign, "dollars")
    main = _to_words(amount)
    if cents:
        cent_value = int(cents.ljust(2, "0"))
        cents_words = num2words(cent_value)
        return f"{main} {currency_word} and {cents_words} cents"
    return f"{main} {currency_word}"


def _replace_number(match: re.Match, year_hint: Callable[[int], bool]) -> str:
    raw = match.group()
    number = int(_strip_commas(raw))
    return _to_words(raw, to_year=year_hint(number))


def clean_text_for_tts(text: str) -> str:
    """
    Normalizes numbers to words so TTS pronounces them naturally.
    - $100 -> "one hundred dollars"
    - 1995 -> "nineteen ninety-five" (year style)
    """
    if not text:
        return text

    # Convert numbered list items to ordinals: "1." -> "first"
    def _list_num_to_ordinal(match: re.Match) -> str:
        num = int(match.group(1))
        return f"{num2words(num, to='ordinal')} "

    text = re.sub(r"(?m)^\s*(\d+)\.\s+", _list_num_to_ordinal, text)

    # Normalize smart quotes/dashes to avoid phonemizer confusion.
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("—", ", ")
        .replace("–", "-")
    )

    # Expand common contractions for clearer pronunciation.
    contractions = {
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "can't": "cannot",
        "won't": "will not",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
        "i'm": "i am",
        "i've": "i have",
        "i'd": "i would",
        "i'll": "i will",
        "we're": "we are",
        "we've": "we have",
        "we'll": "we will",
        "they're": "they are",
        "they've": "they have",
        "they'll": "they will",
        "it's": "it is",
        "there's": "there is",
        "that's": "that is",
        "what's": "what is",
        "who's": "who is",
        "could've": "could have",
        "should've": "should have",
        "would've": "would have",
    }
    for contraction, expansion in contractions.items():
        text = re.sub(rf"(?i)\b{re.escape(contraction)}\b", expansion, text)

    # Remove repeated punctuation that can trigger artifacts.
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"!{2,}", "!", text)

    # Strip markdown bullets/asterisks so TTS doesn't read them aloud.
    text = re.sub(r"(^|\n)\s*[*+-]\s+", r"\1", text)
    text = text.replace("*", " ")

    # Address-style "St" -> "Street" (avoid "Saint" in addresses)
    text = re.sub(r"\b(\d+)\s+St\.?\b", r"\1 Street", text)
    text = re.sub(r"\bSt\.?(?=\s*,|\s*$)", " Street", text)

    # 1) Tech jargon first (so the normalizer doesn't alter them)
    for key, value in _ACRONYMS.items():
        text = re.sub(rf"\b{re.escape(key)}\b", value, text)

    # 2) If NeMo is available, let it handle general TN/ITN
    normalizer = _get_nemo_normalizer()
    if normalizer is not None:
        text = normalizer.normalize(text)
    else:
        # 3) Fallback: lightweight rules
        for key, value in _ABBREVIATIONS.items():
            text = re.sub(rf"\b{re.escape(key)}\b", value, text)

        text = (
            text.replace("%", " percent")
            .replace("&", " and ")
            .replace("+", " plus ")
            .replace("@", " at ")
            .replace("#", " number ")
        )

        # Decimals: "1.3" -> "1 point 3" (before currency/number rules)
        text = re.sub(r"(\d+)\.(\d+)", r"\1 point \2", text)

        text = _CURRENCY_RE.sub(_replace_currency, text)

        def year_hint(value: int) -> bool:
            return 1000 <= value <= 2099

        text = _NUMBER_RE.sub(lambda m: _replace_number(m, year_hint), text)

    text = _glue_connectors(text)
    return text


def _glue_connectors(text: str) -> str:
    connectors = [
        "What's",
        "How's",
        "Where's",
        "Who's",
        "What",
        "How",
        "Where",
        "When",
        "Why",
        "It's",
        "That's",
    ]
    no_glue_after = {"about"}
    connectors.sort(key=len, reverse=True)
    for word in connectors:
        pattern = re.compile(rf"\b({re.escape(word)})\s+([A-Za-z]+)", re.IGNORECASE)
        def _repl(match: re.Match) -> str:
            next_word = match.group(2)
            if next_word.lower() in no_glue_after:
                return f"{match.group(1)} {next_word}"
            return f"{match.group(1)}{next_word}"
        text = pattern.sub(_repl, text)
    return text
