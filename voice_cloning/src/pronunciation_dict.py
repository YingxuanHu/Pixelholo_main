from __future__ import annotations

from functools import lru_cache


_ARPABET_BASE_TO_IPA = {
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ʌ",
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "EH": "ɛ",
    "ER": "ɝ",
    "EY": "eɪ",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "IH": "ɪ",
    "IY": "i",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "P": "p",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "UH": "ʊ",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}

_WORD_OVERRIDES = {
    "are": "ɑːɹ",
    "r": "ˈɑːɹ",
    "er": "ˈɝ",
}


def _try_load_cmudict() -> dict[str, list[list[str]]]:
    try:
        import cmudict  # type: ignore

        return cmudict.dict()
    except Exception:
        pass

    try:
        from nltk.corpus import cmudict as nltk_cmudict  # type: ignore

        return nltk_cmudict.dict()
    except Exception:
        return {}


@lru_cache(maxsize=1)
def load_cmudict_entries() -> dict[str, list[list[str]]]:
    return _try_load_cmudict()


def has_base_pronunciation_dictionary() -> bool:
    return bool(load_cmudict_entries())


def _arpabet_token_to_ipa(token: str) -> str:
    stress = ""
    base = token
    if token and token[-1].isdigit():
        base = token[:-1]
        if base == "AH" and token[-1] == "0":
            return "ə"
        if base == "ER":
            if token[-1] == "0":
                return "ɚ"
            if token[-1] == "1":
                return "ˈɝ"
            if token[-1] == "2":
                return "ˌɝ"
        if token[-1] == "1":
            stress = "ˈ"
        elif token[-1] == "2":
            stress = "ˌ"

    ipa = _ARPABET_BASE_TO_IPA.get(base)
    if ipa is None:
        return ""
    return f"{stress}{ipa}"


def arpabet_to_ipa(symbols: list[str]) -> str:
    return "".join(part for part in (_arpabet_token_to_ipa(symbol) for symbol in symbols) if part)


def lookup_base_pronunciation(word: str) -> str | None:
    key = word.strip().lower()
    if not key:
        return None

    override = _WORD_OVERRIDES.get(key)
    if override:
        return override

    entries = load_cmudict_entries()
    if not entries:
        return None

    pronunciations = entries.get(key)
    if not pronunciations:
        return None

    for pronunciation in pronunciations:
        ipa = arpabet_to_ipa(pronunciation)
        if ipa:
            return ipa
    return None
