import argparse
import json
import re
from collections import Counter
from pathlib import Path
import sys

import phonemizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROFILE_TYPE_AVATAR, PROFILE_TYPE_VOICE, resolve_dataset_root
from src.pronunciation_dict import has_base_pronunciation_dictionary, lookup_base_pronunciation

WORD_RE = re.compile(r"[A-Za-z']+")


def _extract_words(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def _load_metadata(metadata_path: Path) -> list[str]:
    lines = metadata_path.read_text().splitlines()
    texts = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        texts.append(parts[1].strip())
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a default lexicon.json for a profile.")
    parser.add_argument("--profile", help="Profile name (uses data/<type>/<profile>/metadata.csv).")
    parser.add_argument(
        "--profile_type",
        choices=[PROFILE_TYPE_VOICE, PROFILE_TYPE_AVATAR],
        default=PROFILE_TYPE_VOICE,
        help="Profile type to read data from.",
    )
    parser.add_argument("--metadata", type=Path, help="Path to metadata.csv.")
    parser.add_argument("--lang", default="en-ca", help="Phonemizer language (default: en-ca).")
    parser.add_argument("--min_count", type=int, default=1, help="Only include words seen N times.")
    parser.add_argument("--output", type=Path, help="Output lexicon.json path.")
    args = parser.parse_args()

    if args.profile:
        profile_root = resolve_dataset_root(args.profile, args.profile_type)
        metadata_path = profile_root / "metadata.csv"
        output_path = profile_root / "lexicon.json"
    elif args.metadata:
        metadata_path = args.metadata
        output_path = metadata_path.parent / "lexicon.json"
    else:
        raise SystemExit("Provide --profile or --metadata.")

    if args.output:
        output_path = args.output

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")

    texts = _load_metadata(metadata_path)
    counts = Counter()
    for text in texts:
        counts.update(_extract_words(text))

    words = [word for word, count in counts.items() if count >= args.min_count]
    words.sort()

    try:
        backend = phonemizer.backend.EspeakBackend(
            language=args.lang,
            preserve_punctuation=False,
            with_stress=True,
        )
    except RuntimeError:
        backend = phonemizer.backend.EspeakBackend(
            language="en-us",
            preserve_punctuation=False,
            with_stress=True,
        )

    lexicon = {}
    base_dict_count = 0
    phonemizer_count = 0
    for word in words:
        phoneme = lookup_base_pronunciation(word)
        if phoneme:
            base_dict_count += 1
        else:
            phoneme = backend.phonemize([word])[0].strip()
            phonemizer_count += 1
        lexicon[word] = phoneme

    output_path.write_text(json.dumps(lexicon, indent=2, ensure_ascii=True))
    print(f"Wrote {output_path} ({len(lexicon)} words)")
    print(
        f"Sources: base_dict={base_dict_count}, phonemizer={phonemizer_count}, "
        f"cmudict_available={has_base_pronunciation_dictionary()}"
    )


if __name__ == "__main__":
    main()
