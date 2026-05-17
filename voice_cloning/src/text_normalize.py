import re
from typing import Callable

from num2words import num2words

try:
    from nemo_text_processing.text_normalization.normalize import Normalizer  # type: ignore
except Exception:
    Normalizer = None


_CURRENCY_RE = re.compile(r"(?P<sign>[$£€])\s?(?P<amount>\d[\d,]*)(?:\.(?P<cents>\d{1,2}))?")
_NUMBER_RE = re.compile(r"\d[\d,]*")
_DOTTED_INITIALISM_RE = re.compile(r"\b(?:[A-Za-z]\.){2,}[A-Za-z]?\.?")
_TEMP_RE = re.compile(
    r"(?P<value>[+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:°\s*)?(?P<unit>[CF])\b",
    re.IGNORECASE,
)
_STANDALONE_TEMP_UNIT_RE = re.compile(r"°\s*(?P<unit>[CF])\b", re.IGNORECASE)
_TEMP_CONTEXT_RE = re.compile(
    r"\b(?:temperature|weather|forecast|currently|current|feels?\s+like|high|low|degrees?)\b",
    re.IGNORECASE,
)
_TEMP_UNIT_PAIR_RE = re.compile(
    r"\b(?P<first>[CF])\s*(?P<joiner>/|and|or)\s*(?P<second>[CF])\b",
    re.IGNORECASE,
)
_DEGREES_SINGLE_UNIT_RE = re.compile(r"\bdegrees?\s+(?P<unit>[CF])\b", re.IGNORECASE)
_UNIT_REPLACEMENTS = (
    (re.compile(r"\bkm\s*/\s*h\b", re.IGNORECASE), "kilometers per hour"),
    (re.compile(r"\bkmh\b", re.IGNORECASE), "kilometers per hour"),
    (re.compile(r"\bkph\b", re.IGNORECASE), "kilometers per hour"),
    (re.compile(r"\bmph\b", re.IGNORECASE), "miles per hour"),
    (re.compile(r"\bm\s*/\s*s\b", re.IGNORECASE), "meters per second"),
    (re.compile(r"\bhpa\b", re.IGNORECASE), "hectopascals"),
    (re.compile(r"\bmb\b"), "millibars"),
    (re.compile(r"\buv\b", re.IGNORECASE), "U V"),
    (re.compile(r"\baqi\b", re.IGNORECASE), "A Q I"),
    (re.compile(r"\bpm\s*2\s*\.?\s*5\b", re.IGNORECASE), "P M two point five"),
    (re.compile(r"(?:µ|μ|u)g\s*/\s*m(?:³|\^3|3)\b", re.IGNORECASE), "micrograms per cubic meter"),
)
_WIND_DIRECTIONS = {
    "N": "north",
    "NNE": "north northeast",
    "NE": "northeast",
    "ENE": "east northeast",
    "E": "east",
    "ESE": "east southeast",
    "SE": "southeast",
    "SSE": "south southeast",
    "S": "south",
    "SSW": "south southwest",
    "SW": "southwest",
    "WSW": "west southwest",
    "W": "west",
    "WNW": "west northwest",
    "NW": "northwest",
    "NNW": "north northwest",
}
_WIND_DIRECTION_PATTERN = re.compile(
    r"\b(?P<prefix>winds?|wind\s+(?:is|are|from|out\s+of)?|from|towards?|gusts?)"
    r"(?P<space>\s+(?:the\s+)?)"
    r"(?P<direction>NNE|NNW|ENE|ESE|SSE|SSW|WSW|WNW|NE|NW|SE|SW|N|E|S|W)\b",
    re.IGNORECASE,
)
_DIRECTION_WIND_PATTERN = re.compile(
    r"\b(?P<direction>NNE|NNW|ENE|ESE|SSE|SSW|WSW|WNW|NE|NW|SE|SW|N|E|S|W)"
    r"(?P<space>\s+)"
    r"(?P<suffix>winds?|gusts?)\b",
    re.IGNORECASE,
)

_ACRONYMS = {
    "US": "U S",
    "USA": "U S A",
    "UK": "U K",
    "UAE": "U A E",
    "EU": "E U",
    "UN": "U N",
    "NYC": "N Y C",
    "DC": "D C",
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
    normalizer = _get_nemo_normalizer()
    # Instantiating the normalizer doesn't compile the OpenFST grammars —
    # only the first normalize() call does. Run it here so first inference is fast.
    sample = (
        "The temperature is 72 degrees F with winds at 15 mph. "
        "There are 3 items costing $12.50 each."
    )
    try:
        if normalizer is not None:
            normalizer.normalize(sample)
    except Exception:
        pass


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


def _expand_dotted_initialism(match: re.Match) -> str:
    letters = re.findall(r"[A-Za-z]", match.group(0))
    return " ".join(letter.upper() for letter in letters)


def _replace_temperature(match: re.Match) -> str:
    unit = match.group("unit").upper()
    unit_word = "Fahrenheit" if unit == "F" else "Celsius"
    return f"{match.group('value')} degrees {unit_word}"


def _replace_standalone_temperature_unit(match: re.Match) -> str:
    unit = match.group("unit").upper()
    unit_word = "Fahrenheit" if unit == "F" else "Celsius"
    return f"degrees {unit_word}"


def _temperature_unit_word(unit: str) -> str:
    return "Fahrenheit" if unit.upper() == "F" else "Celsius"


def _replace_temperature_unit_pair(match: re.Match) -> str:
    first = _temperature_unit_word(match.group("first"))
    second = _temperature_unit_word(match.group("second"))
    joiner = match.group("joiner").lower()
    spoken_joiner = "or" if joiner == "or" else "and"
    return f"{first} {spoken_joiner} {second}"


def _normalize_bare_temperature_units(text: str) -> str:
    text = _DEGREES_SINGLE_UNIT_RE.sub(
        lambda match: f"degrees {_temperature_unit_word(match.group('unit'))}",
        text,
    )
    if _TEMP_CONTEXT_RE.search(text):
        text = _TEMP_UNIT_PAIR_RE.sub(_replace_temperature_unit_pair, text)
    return text


def _normalize_spoken_units(text: str) -> str:
    text = _TEMP_RE.sub(_replace_temperature, text)
    text = _STANDALONE_TEMP_UNIT_RE.sub(_replace_standalone_temperature_unit, text)
    text = _normalize_bare_temperature_units(text)
    text = re.sub(r"%", " percent", text)
    for pattern, replacement in _UNIT_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    text = _WIND_DIRECTION_PATTERN.sub(
        lambda match: (
            f"{match.group('prefix')}{match.group('space')}"
            f"{_WIND_DIRECTIONS[match.group('direction').upper()]}"
        ),
        text,
    )
    text = _DIRECTION_WIND_PATTERN.sub(
        lambda match: (
            f"{_WIND_DIRECTIONS[match.group('direction').upper()]}"
            f"{match.group('space')}{match.group('suffix')}"
        ),
        text,
    )
    return text


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
        .replace("–", ", ")
    )

    # Turn dotted letter abbreviations into connected spoken letters:
    # "B.C." -> "B C", "U.S.A." -> "U S A".
    text = _DOTTED_INITIALISM_RE.sub(_expand_dotted_initialism, text)

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

    text = _normalize_spoken_units(text)

    # Remove repeated punctuation that can trigger artifacts.
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"!{2,}", "!", text)

    # Strip markdown bullets/asterisks so TTS doesn't read them aloud.
    text = re.sub(r"(^|\n)\s*[*+-]\s+", r"\1", text)
    text = text.replace("*", " ")
    text = re.sub(r"[()\[\]{}<>|]", " ", text)
    text = re.sub(r"[\\/]", " ", text)
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)
    text = re.sub(r"-{2,}", ", ", text)

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
