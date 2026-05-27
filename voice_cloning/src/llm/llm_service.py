import contextlib
import logging
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List

from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI as OpenAIClient


logger = logging.getLogger("pixelholo.llm")


def _env_int(name: str, default: int, *, minimum: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning(
            "component=llm op=env_config status=invalid name=%s value=%s default=%s",
            name,
            raw_value,
            default,
        )
        return default
    return max(minimum, value)


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}


def _load_env_files() -> None:
    load_dotenv()
    for parent in Path(__file__).resolve().parents:
        env_path = parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        if (parent / ".git").exists():
            break


LLM_MODE_LEGACY_FAST = "legacy_fast"
LLM_MODE_LIVE_SEARCH = "live_search"
LLM_MODE_GEMINI_FLASH = "gemini_flash"
LLM_MODE_GEMINI_SEARCH = "gemini_search"
LLM_MODE_AUTO = "auto"
DEFAULT_LLM_MODE = LLM_MODE_LEGACY_FAST

LEGACY_FAST_MODEL = "llama-3.1-8b-instant"
LIVE_SEARCH_MODEL = "gpt-4o-mini"
GEMINI_FLASH_MODEL = "gemini-2.5-flash-lite"
OPENAI_RESPONSES_SEARCH_MODELS: frozenset[str] = frozenset(
    {
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-mini",
        "gpt-5.4-nano",
        "gpt-5.4-mini",
        "gpt-5-nano",
        "gpt-5-mini",
    }
)
OPENAI_CHAT_SEARCH_MODELS: frozenset[str] = frozenset(
    {
        "gpt-5-search-api",
        "gpt-4o-mini-search-preview",
        "gpt-4o-search-preview",
    }
)
OPENAI_MODELS: frozenset[str] = OPENAI_RESPONSES_SEARCH_MODELS | OPENAI_CHAT_SEARCH_MODELS
OPENAI_REASONING_EFFORTS: frozenset[str] = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})
WARMUP_MAX_COMPLETION_TOKENS = 16
REALTIME_MAX_COMPLETION_TOKENS = 320
OPENAI_REALTIME_MAX_OUTPUT_TOKENS = 512
GEMINI_MAX_OUTPUT_TOKENS = 320
GEMINI_SEARCH_TIMEOUT_SECONDS = 15
MAX_HISTORY_MESSAGES = 8
DEFAULT_REALTIME_MAX_COMPLETION_TOKENS = REALTIME_MAX_COMPLETION_TOKENS
DEFAULT_OPENAI_REALTIME_MAX_OUTPUT_TOKENS = OPENAI_REALTIME_MAX_OUTPUT_TOKENS
DEFAULT_GEMINI_MAX_OUTPUT_TOKENS = GEMINI_MAX_OUTPUT_TOKENS
DEFAULT_GEMINI_SEARCH_TIMEOUT_SECONDS = GEMINI_SEARCH_TIMEOUT_SECONDS
DEFAULT_GEMINI_THINKING_LEVEL = "minimal"
DEFAULT_OPENAI_REASONING_EFFORT = "none"
DEFAULT_REALTIME_HISTORY_MESSAGES = 2
DEFAULT_MAX_MESSAGE_CHARS = 1200
DEFAULT_MAX_HISTORY_CHARS = 6000
DEFAULT_REALTIME_MAX_MESSAGE_CHARS = 800
DEFAULT_REALTIME_MAX_HISTORY_CHARS = 2600
DEFAULT_AUTO_LIVE_FOLLOWUP_TURNS = 2
DEFAULT_AUTO_LIVE_FOLLOWUP_TTL_SECONDS = 600
DEFAULT_AUTO_LIVE_TOPIC_TURNS = 4
DEFAULT_FIRST_CHUNK_SOFT_MAX_CHARS = 72
DEFAULT_AUTO_GEMINI_SEARCH_ENABLED = True
RETRY_AFTER_PATTERN = re.compile(r"try again in ([0-9.]+)s", re.IGNORECASE)
REQUEST_TOO_LARGE_PATTERN = re.compile(r"request entity too large", re.IGNORECASE)
# Matches OpenAI inline citation annotations like ([domain.com](https://...))
_OPENAI_CITATION_PATTERN = re.compile(r"\s*\(\[[^\]]*\]\([^)]*\)\)")
_TRAILING_DOTTED_INITIALISM_PATTERN = re.compile(
    r"(?:^|[\s(])(?:[A-Za-z]\.\s*)+$"
)
_TRAILING_COMMON_ABBREVIATION_PATTERN = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e)\.\s*$",
    re.IGNORECASE,
)


def _strip_openai_citations(text: str) -> str:
    return _OPENAI_CITATION_PATTERN.sub("", text)


def _ends_with_protected_abbreviation(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    return bool(
        _TRAILING_DOTTED_INITIALISM_PATTERN.search(stripped)
        or _TRAILING_COMMON_ABBREVIATION_PATTERN.search(stripped)
    )


def _ends_with_safe_soft_boundary(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped or stripped[-1] not in ",;:":
        return False
    return not bool(re.search(r"\d,\s*$", stripped))

SUPPORTED_LLM_MODES = {
    LLM_MODE_LEGACY_FAST,
    LLM_MODE_LIVE_SEARCH,
    LLM_MODE_GEMINI_FLASH,
    LLM_MODE_GEMINI_SEARCH,
    LLM_MODE_AUTO,
}

MODEL_BY_MODE = {
    LLM_MODE_LEGACY_FAST: LEGACY_FAST_MODEL,
    LLM_MODE_LIVE_SEARCH: LIVE_SEARCH_MODEL,
    LLM_MODE_GEMINI_FLASH: GEMINI_FLASH_MODEL,
    LLM_MODE_GEMINI_SEARCH: GEMINI_FLASH_MODEL,
}

# Tier 1 — strong live signals: always live_search, cannot be overridden.
_LIVE_STRONG: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bweather\s+(in|for|at|near|today|tomorrow|tonight|right\s+now|now)\b", re.I),
    re.compile(r"\bforecast\b", re.I),
    re.compile(r"\bbreaking\b", re.I),
    re.compile(r"\bright\s+now\b", re.I),
    re.compile(r"\bat\s+the\s+moment\b", re.I),
    re.compile(r"\bexchange\s+rate\b", re.I),
    re.compile(r"\btrending\b", re.I),
    re.compile(r"\btoday'?s\b", re.I),
    re.compile(r"\bwhat\s+('s\s*|is\s+)?(happening|going\s+on)\b", re.I),
    re.compile(r"\b(latest|current|recent)\s+(news|headlines?|updates?|events?|developments?)\b", re.I),
    re.compile(r"\b(any|the|what'?s?)\s+(news|updates?|headlines?)\b", re.I),
    re.compile(r"\b(stock|share)\s+price\b", re.I),
    re.compile(r"\b(bitcoin|ethereum|crypto|nft)\b", re.I),
    re.compile(r"\b20(2[4-9]|3[0-9])\b"),
    re.compile(r"\b(live|real[\s-]?time)\s+(score|update|result|feed)\b", re.I),
    re.compile(r"\bthis\s+(morning|afternoon|evening|week|month)\b", re.I),
    re.compile(r"\bjust\s+(happen\w*|announc\w+|releas\w+|launch\w+|came\s+out)\b", re.I),
    re.compile(r"\bwhat'?s?\s+new\b", re.I),
    re.compile(r"\bwho\s+is\s+(?:the\s+)?(?:current|sitting)\s+", re.I),
    re.compile(r"\bwho\s+is\s+(?:the\s+)?(?:u\.?\s*s\.?|united\s+states|american)\s+president\b", re.I),
    re.compile(r"\bwho\s+is\s+(?:the\s+)?president\s+of\s+(?:the\s+)?(?:u\.?\s*s\.?|united\s+states|america)\b", re.I),
    re.compile(r"\b(?:current|sitting)\s+(?:president|prime\s+minister|mayor|governor|senator|ceo|leader)\b", re.I),
    re.compile(r"\bwho\s+is\s+(?:likely|expected|favou?red|projected)\s+to\s+(?:be|become|win)\b", re.I),
    re.compile(
        r"\b(?:next|future|upcoming)\s+(?:u\.?\s*s\.?|united\s+states|american)?\s*"
        r"(?:president|presidential|election|administration|government|leader)\b",
        re.I,
    ),
    re.compile(r"\b(?:presidential|election)\s+(?:polls?|odds|forecast|prediction|candidate|race)\b", re.I),
)

# Tier 2 — weak live signals: live_search unless a tier-3 negative fires.
_LIVE_WEAK: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bnews\b", re.I),
    re.compile(r"\brecent(ly)?\b", re.I),
    re.compile(r"\byesterday\b", re.I),
    re.compile(r"\btoday\b", re.I),
    re.compile(r"\btomorrow\b", re.I),
    re.compile(r"\btonight\b", re.I),
    re.compile(r"\bcurrent(ly)?\b", re.I),
    re.compile(r"\blatest\b", re.I),
    re.compile(r"\bnow\b", re.I),
    re.compile(r"\bmarket\b", re.I),
    re.compile(r"\bscore\b", re.I),
    re.compile(r"\bstock\b", re.I),
    re.compile(r"\bprice\b", re.I),
    re.compile(r"\btemperature\b", re.I),
    re.compile(r"\btraffic\b", re.I),
    re.compile(r"\blive\b", re.I),
    re.compile(r"\bafter\s+2023\b", re.I),
    re.compile(r"\bup[\s-]to[\s-]date\b", re.I),
    re.compile(r"\b(rain(ing)?|snow(ing)?|humid(ity)?|sunny|cloudy|windy)\b", re.I),
    re.compile(r"\bweather\b", re.I),
)

# Tier 3 — negative overrides: suppress tier-2 signals (never suppress tier-1).
_LEGACY_PREFER: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bhow\s+to\b", re.I),
    re.compile(r"\bexplain\b", re.I),
    re.compile(r"\bdefin(e|ition)\b", re.I),
    re.compile(r"\bwhat\s+is\s+(a|an)\b", re.I),
    re.compile(r"\bwhat\s+does\b", re.I),
    re.compile(r"\bwrite\s+(me\s+)?(a|an|the)\b", re.I),
    re.compile(r"\b(create|generate|compose|draft)\b", re.I),
    re.compile(r"\b(recipe|cooking|bak(e|ing)|grill(ing)?)\b", re.I),
    re.compile(r"\b(supermarket|grocery|shopping)\b", re.I),
    re.compile(r"\bstock\s+(photo|image|footage|illustration)\b", re.I),
    re.compile(r"\b(code|program|script|algorithm|function|formula)\b", re.I),
    re.compile(r"\b(joke|poem|story|lyric|fiction|novel)\b", re.I),
    re.compile(r"\btranslat(e|ion)\b", re.I),
    re.compile(r"\b(math|calculate|compute|equation)\b", re.I),
    re.compile(r"\bhistor(y|ical|ically)\b|\bin\s+(history|the\s+past)\b", re.I),
    re.compile(r"\b(best\s+practice|approach|methodology|technique|paradigm|pattern)\b", re.I),
)

_LIVE_FOLLOWUP: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(what|how)\s+about\b", re.I),
    re.compile(r"^\s*(and|also|then|so)\b", re.I),
    re.compile(r"^\s*(why|how\s+so)\??\s*$", re.I),
    re.compile(r"\b(tell\s+me\s+more|more\s+detail|what\s+else|anything\s+else)\b", re.I),
    re.compile(r"\b(that|it|this|they|them|there|those|these|same|he|him|his|she|her)\b", re.I),
    re.compile(r"\b(should|would|could|can)\s+(i|we)\b", re.I),
)

_LIVE_CONTEXT_FOLLOWUP: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(likely|expected|favou?red|projected|odds|chances?|polls?|prediction|forecast)\b", re.I),
    re.compile(r"\b(who\s+will|who'?s\s+going\s+to|going\s+to\s+win|could\s+win|would\s+win)\b", re.I),
    re.compile(r"\b(next|future|upcoming|successor|replace|after)\b", re.I),
    re.compile(
        r"\b(president|prime\s+minister|mayor|governor|senator|congress|parliament|"
        r"election|campaign|candidate|nominee|incumbent|administration|government|leader)\b",
        re.I,
    ),
)

_QUESTION_LIKE_PATTERN = re.compile(
    r"^\s*(who|what|when|where|why|how|which|do|does|did|is|are|can|could|would|should|will)\b",
    re.I,
)

_LIVE_CONTEXT_STOPWORDS = {
    "about",
    "after",
    "again",
    "answer",
    "around",
    "because",
    "before",
    "being",
    "current",
    "currently",
    "could",
    "degrees",
    "detail",
    "does",
    "from",
    "have",
    "into",
    "just",
    "likely",
    "more",
    "next",
    "only",
    "please",
    "right",
    "same",
    "should",
    "tell",
    "than",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "today",
    "tomorrow",
    "tonight",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "your",
}

_AUTO_GPT_REQUIRED: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(president|prime\s+minister|mayor|governor|senator|congress|parliament)\b", re.I),
    re.compile(r"\b(election|campaign|candidate|polls?|odds|forecast|projected|likely\s+to\s+win)\b", re.I),
    re.compile(r"\b(news|headline|breaking|war|conflict|court|lawsuit|legal|medical|health)\b", re.I),
    re.compile(r"\b(stock|share|market|crypto|bitcoin|ethereum|price|earnings|exchange\s+rate)\b", re.I),
    re.compile(r"\b(ceo|leader|government|administration|policy|law|regulation)\b", re.I),
)

_AUTO_GEMINI_LIGHT_LIVE: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(weather|temperature|forecast|rain|snow|humidity|wind)\b", re.I),
    re.compile(r"\b(what\s+day|what\s+date|today'?s?\s+date|current\s+time|time\s+in)\b", re.I),
    re.compile(r"\b(open\s+(now|today)|hours?|schedule|near\s+me)\b", re.I),
    re.compile(r"\b(local|nearby|traffic|directions?)\b", re.I),
)

LOCATION_PREPOSITION_PATTERN = re.compile(
    r"\b(?:in|near|for|at)\s+"
    r"(?!today\b|tomorrow\b|tonight\b|now\b|right\b|current\b|the\b|a\b|an\b)"
    r"[a-z][a-z .,'-]{2,}",
    re.IGNORECASE,
)
WEATHER_QUERY_PATTERN = re.compile(
    r"\b(?:weather|forecast|temperature|feels?\s+like|rain|snow|wind|humidity)\b",
    re.IGNORECASE,
)
WEATHER_LOCATION_PATTERN = re.compile(
    r"\b(?:in|for|at|near)\s+"
    r"(?P<location>[A-Za-z][A-Za-z .,'-]{1,80}?)"
    r"(?=$|[?.!,]|\s+(?:today|right\s+now|now|currently|tonight|tomorrow|"
    r"tell|give|show|please|with|and|I)\b)",
    re.IGNORECASE,
)
WEATHER_DETAIL_PATTERN = re.compile(
    r"\b(?:detail|detailed|more|humidity|wind|rain|snow|precipitation|forecast)\b",
    re.IGNORECASE,
)
WEATHER_CODE_DESCRIPTIONS = {
    0: "clear skies",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast skies",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


class LLMStreamCancelled(Exception):
    pass


@dataclass(frozen=True)
class LLMRoute:
    requested_mode: str
    resolved_mode: str
    model: str

    @property
    def uses_realtime_tools(self) -> bool:
        return self.model in OPENAI_MODELS

    @property
    def uses_openai_responses_search(self) -> bool:
        return self.model in OPENAI_RESPONSES_SEARCH_MODELS

    @property
    def uses_openai_chat_search(self) -> bool:
        return self.model in OPENAI_CHAT_SEARCH_MODELS

    @property
    def uses_gemini(self) -> bool:
        return self.resolved_mode in {LLM_MODE_GEMINI_FLASH, LLM_MODE_GEMINI_SEARCH} or self.model.startswith(
            ("gemini-", "models/gemini-")
        )

    @property
    def uses_gemini_search(self) -> bool:
        return self.resolved_mode == LLM_MODE_GEMINI_SEARCH

    @property
    def uses_current_tools(self) -> bool:
        return self.uses_realtime_tools or self.uses_gemini_search

    @property
    def cache_key(self) -> str:
        return f"{self.resolved_mode}:{self.model}"


class LLMService:
    def __init__(self, system_prompt: str = "You are a helpful, concise AI assistant."):
        _load_env_files()
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("WARNING: GROQ_API_KEY not found in environment variables.")

        self.client = Groq(api_key=api_key)
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            print("WARNING: OPENAI_API_KEY not found — live_search mode will fall back to legacy_fast.")
        self.openai_client: OpenAIClient | None = OpenAIClient(api_key=openai_api_key) if openai_api_key else None
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not self.gemini_api_key:
            print("WARNING: GEMINI_API_KEY not found — Gemini modes will be unavailable.")
        self.gemini_model = os.environ.get("GEMINI_MODEL", GEMINI_FLASH_MODEL).strip() or GEMINI_FLASH_MODEL
        self.gemini_search_model = (
            os.environ.get("GEMINI_SEARCH_MODEL", self.gemini_model).strip() or self.gemini_model
        )
        self.live_search_model = os.environ.get("OPENAI_LIVE_SEARCH_MODEL", LIVE_SEARCH_MODEL).strip() or LIVE_SEARCH_MODEL
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        self._warmed_routes: set[str] = set()
        self._realtime_rate_limit_until = 0.0
        self.service_tier = os.environ.get("GROQ_SERVICE_TIER", "").strip().lower()
        self.realtime_max_completion_tokens = _env_int(
            "GROQ_REALTIME_MAX_COMPLETION_TOKENS",
            DEFAULT_REALTIME_MAX_COMPLETION_TOKENS,
            minimum=32,
        )
        self.openai_realtime_max_output_tokens = _env_int(
            "OPENAI_REALTIME_MAX_OUTPUT_TOKENS",
            DEFAULT_OPENAI_REALTIME_MAX_OUTPUT_TOKENS,
            minimum=256,
        )
        self.gemini_max_output_tokens = _env_int(
            "GEMINI_MAX_OUTPUT_TOKENS",
            DEFAULT_GEMINI_MAX_OUTPUT_TOKENS,
            minimum=32,
        )
        self.gemini_search_timeout_seconds = _env_int(
            "GEMINI_SEARCH_TIMEOUT_SEC",
            DEFAULT_GEMINI_SEARCH_TIMEOUT_SECONDS,
            minimum=5,
        )
        self.auto_gemini_search_enabled = _env_bool(
            "LLM_AUTO_GEMINI_SEARCH",
            DEFAULT_AUTO_GEMINI_SEARCH_ENABLED,
        )
        self.gemini_thinking_level = os.environ.get("GEMINI_THINKING_LEVEL", DEFAULT_GEMINI_THINKING_LEVEL).strip().lower()
        self.openai_reasoning_effort = self._normalize_openai_reasoning_effort(
            os.environ.get("OPENAI_REASONING_EFFORT", DEFAULT_OPENAI_REASONING_EFFORT)
        )
        self.realtime_history_messages = _env_int(
            "GROQ_REALTIME_HISTORY_MESSAGES",
            DEFAULT_REALTIME_HISTORY_MESSAGES,
            minimum=1,
        )
        self.max_message_chars = _env_int(
            "GROQ_MAX_MESSAGE_CHARS",
            DEFAULT_MAX_MESSAGE_CHARS,
            minimum=200,
        )
        self.max_history_chars = _env_int(
            "GROQ_MAX_HISTORY_CHARS",
            DEFAULT_MAX_HISTORY_CHARS,
            minimum=500,
        )
        self.realtime_max_message_chars = _env_int(
            "GROQ_REALTIME_MAX_MESSAGE_CHARS",
            DEFAULT_REALTIME_MAX_MESSAGE_CHARS,
            minimum=200,
        )
        self.realtime_max_history_chars = _env_int(
            "GROQ_REALTIME_MAX_HISTORY_CHARS",
            DEFAULT_REALTIME_MAX_HISTORY_CHARS,
            minimum=500,
        )
        self.auto_live_followup_turns = _env_int(
            "LLM_AUTO_LIVE_FOLLOWUP_TURNS",
            DEFAULT_AUTO_LIVE_FOLLOWUP_TURNS,
            minimum=0,
        )
        self.auto_live_followup_ttl_seconds = _env_int(
            "LLM_AUTO_LIVE_FOLLOWUP_TTL_SEC",
            DEFAULT_AUTO_LIVE_FOLLOWUP_TTL_SECONDS,
            minimum=1,
        )
        self.auto_live_topic_turns = _env_int(
            "LLM_AUTO_LIVE_TOPIC_TURNS",
            DEFAULT_AUTO_LIVE_TOPIC_TURNS,
            minimum=0,
        )
        self.first_chunk_soft_max_chars = _env_int(
            "LLM_FIRST_CHUNK_SOFT_MAX_CHARS",
            DEFAULT_FIRST_CHUNK_SOFT_MAX_CHARS,
            minimum=32,
        )
        self._turn_index = 0
        self._last_live_turn_index: int | None = None
        self._last_live_until = 0.0
        self._last_live_context_terms: set[str] = set()
        self._last_live_context_summary = ""

    @property
    def stream_warmed(self) -> bool:
        return self.is_warmed()

    def is_warmed(self, mode: str | None = None, model: str | None = None) -> bool:
        return all(route.cache_key in self._warmed_routes for route in self._warmup_routes(mode, model))

    def warmup(self, mode: str | None = None, model: str | None = None) -> bool:
        warmed_any = False
        for route in self._warmup_routes(mode, model):
            if route.cache_key in self._warmed_routes:
                continue
            if route.model in OPENAI_MODELS:
                self._warmed_routes.add(route.cache_key)
                warmed_any = True
                logger.info(
                    "component=llm op=warmup status=ok mode=%s resolved_mode=%s model=%s",
                    route.requested_mode,
                    route.resolved_mode,
                    route.model,
                )
                continue
            if route.uses_gemini:
                try:
                    for token in self._stream_gemini_tokens(
                        [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": "Reply with exactly one short word."},
                        ],
                        route.model,
                        enable_search=route.uses_gemini_search,
                    ):
                        if token:
                            break
                    self._warmed_routes.add(route.cache_key)
                    warmed_any = True
                    logger.info(
                        "component=llm op=warmup status=ok mode=%s resolved_mode=%s model=%s",
                        route.requested_mode,
                        route.resolved_mode,
                        route.model,
                    )
                    continue
                except Exception:
                    logger.exception(
                        "component=llm op=warmup status=error mode=%s resolved_mode=%s model=%s",
                        route.requested_mode,
                        route.resolved_mode,
                        route.model,
                    )
                    raise
            try:
                stream = self.client.chat.completions.create(
                    model=route.model,
                    messages=self._messages_for_route(
                        route,
                        [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": "Reply with exactly one short word."},
                        ],
                    ),
                    stream=True,
                    temperature=0.7,
                    max_completion_tokens=WARMUP_MAX_COMPLETION_TOKENS,
                    **self._service_tier_param(),
                )
                for chunk in stream:
                    token = chunk.choices[0].delta.content
                    if token:
                        break
                self._warmed_routes.add(route.cache_key)
                warmed_any = True
                logger.info(
                    "component=llm op=warmup status=ok mode=%s resolved_mode=%s model=%s",
                    route.requested_mode,
                    route.resolved_mode,
                    route.model,
                )
            except Exception:
                logger.exception(
                    "component=llm op=warmup status=error mode=%s resolved_mode=%s model=%s",
                    route.requested_mode,
                    route.resolved_mode,
                    route.model,
                )
                raise
        return warmed_any

    def stream_response(
        self,
        user_input: str,
        min_words: int = 8,
        min_chars: int = 40,
        max_chars: int = 180,
        mode: str | None = None,
        model: str | None = None,
        cancel_event=None,
    ) -> Generator[str, None, None]:
        """
        Sends text to Groq and yields complete spoken-friendly chunks as they are generated.
        """
        if self._is_cancelled(cancel_event):
            return
        self._turn_index += 1
        self.history.append({"role": "user", "content": user_input})
        route = self.resolve_route(user_input=user_input, mode=mode, model=model)

        if route.uses_current_tools:
            if self._is_cancelled(cancel_event):
                return
            direct_weather = self._direct_weather_answer(user_input)
            if self._is_cancelled(cancel_event):
                return
            if direct_weather is not None:
                self.history.append({"role": "assistant", "content": direct_weather})
                self._mark_live_context("direct_weather", user_input=user_input, assistant_text=direct_weather)
                yield direct_weather
                return

        rate_limit_wait = self._realtime_rate_limit_wait_seconds()
        if route.uses_realtime_tools and rate_limit_wait > 0:
            if self._requires_live_route(user_input):
                message = (
                    "The live search model is rate-limited right now. "
                    f"Try again in about {rate_limit_wait} seconds."
                )
                logger.info(
                    "component=llm op=llm_chat status=skipped_rate_limit mode=%s resolved_mode=%s model=%s retry_after_seconds=%s",
                    route.requested_mode,
                    route.resolved_mode,
                    route.model,
                    rate_limit_wait,
                )
                self.history.append({"role": "assistant", "content": message})
                yield message
                return
            route = LLMRoute(
                requested_mode=route.requested_mode,
                resolved_mode=LLM_MODE_LEGACY_FAST,
                model=LEGACY_FAST_MODEL,
            )

        candidates = self._candidate_routes(route, user_input)

        full_response_text = ""
        completed_route: LLMRoute | None = None
        for idx, candidate in enumerate(candidates):
            yielded_chunks = 0
            try:
                route_stream = self._stream_from_route(
                    candidate,
                    min_words=min_words,
                    min_chars=min_chars,
                    max_chars=max_chars,
                    cancel_event=cancel_event,
                )
                while True:
                    if self._is_cancelled(cancel_event):
                        with contextlib.suppress(Exception):
                            route_stream.close()
                        return
                    try:
                        sentence = next(route_stream)
                    except StopIteration as stop:
                        full_response_text = stop.value or ""
                        if not full_response_text.strip() and yielded_chunks == 0:
                            raise RuntimeError(f"{candidate.model} returned no text")
                        self._warmed_routes.add(candidate.cache_key)
                        completed_route = candidate
                        break
                    yielded_chunks += 1
                    yield sentence
                break
            except LLMStreamCancelled:
                return
            except Exception as exc:
                self._remember_realtime_rate_limit(candidate, exc)
                next_candidate = candidates[idx + 1] if idx + 1 < len(candidates) else None
                if yielded_chunks or next_candidate is None:
                    retry_after_seconds = self._retry_after_seconds(exc)
                    logger.exception(
                        "component=llm op=llm_chat status=error mode=%s resolved_mode=%s model=%s input_chars=%s request_too_large=%s retry_after_seconds=%s",
                        candidate.requested_mode,
                        candidate.resolved_mode,
                        candidate.model,
                        len(user_input),
                        self._is_request_too_large(exc),
                        retry_after_seconds,
                    )
                    yield self._friendly_error_message(user_input, exc)
                    return
                logger.warning(
                    "component=llm op=llm_chat_fallback status=start from_model=%s to_model=%s error=%s",
                    candidate.model,
                    next_candidate.model,
                    self._short_error_message(exc),
                )
                continue

        if full_response_text:
            if self._is_cancelled(cancel_event):
                return
            self.history.append({"role": "assistant", "content": full_response_text})
            if completed_route and completed_route.uses_current_tools:
                self._mark_live_context("live_route", user_input=user_input, assistant_text=full_response_text)

    def resolve_route(
        self,
        user_input: str,
        mode: str | None = None,
        model: str | None = None,
    ) -> LLMRoute:
        requested_mode = self._normalize_mode(mode)
        resolved_mode = requested_mode
        if requested_mode == LLM_MODE_AUTO:
            resolved_mode = self._auto_mode_for_prompt(user_input)
        if model and model.strip():
            route_model = model.strip()
        elif resolved_mode == LLM_MODE_GEMINI_FLASH:
            route_model = self.gemini_model
        elif resolved_mode == LLM_MODE_GEMINI_SEARCH:
            route_model = self.gemini_search_model
        elif resolved_mode == LLM_MODE_LIVE_SEARCH:
            route_model = self.live_search_model
        else:
            route_model = MODEL_BY_MODE[resolved_mode]
        return LLMRoute(
            requested_mode=requested_mode,
            resolved_mode=resolved_mode,
            model=route_model,
        )

    def _warmup_routes(self, mode: str | None, model: str | None) -> list[LLMRoute]:
        requested_mode = self._normalize_mode(mode)
        if model and model.strip():
            return [self.resolve_route(user_input="", mode=requested_mode, model=model)]
        if requested_mode == LLM_MODE_AUTO:
            return [
                self.resolve_route(user_input="", mode=LLM_MODE_LEGACY_FAST),
                self.resolve_route(user_input="today", mode=LLM_MODE_AUTO),
            ]
        return [self.resolve_route(user_input="", mode=requested_mode)]

    def _stream_from_route(
        self,
        route: LLMRoute,
        min_words: int,
        min_chars: int,
        max_chars: int,
        cancel_event=None,
    ) -> Generator[str, None, str]:
        if self._is_cancelled(cancel_event):
            raise LLMStreamCancelled()
        messages = self._messages_for_route(route, self._conversation_window(route))
        request_chars = self._messages_char_count(messages)
        logger.info(
            "component=llm op=chat_request mode=%s resolved_mode=%s model=%s messages=%s request_chars=%s",
            route.requested_mode,
            route.resolved_mode,
            route.model,
            len(messages),
            request_chars,
        )
        full_response_text = ""
        buffer = ""
        min_chunk_size = max(min_chars, min_words * 2)
        # First sentence uses a lower threshold so TTS starts sooner.
        first_chunk_min = max(20, min_chunk_size // 2)
        first_soft_max = max(first_chunk_min, int(self.first_chunk_soft_max_chars))
        first_yielded = False
        is_openai = route.uses_realtime_tools
        is_openai_responses = route.uses_openai_responses_search
        is_gemini = route.uses_gemini

        print(f"{route.model} Thinking...", end="", flush=True)

        stream = None
        try:
            if is_gemini:
                stream = self._stream_gemini_tokens(
                    messages,
                    route.model,
                    cancel_event=cancel_event,
                    enable_search=route.uses_gemini_search,
                )
            elif route.uses_openai_responses_search:
                stream = self._stream_openai_responses_tokens(
                    messages,
                    route.model,
                    cancel_event=cancel_event,
                )
            elif route.uses_openai_chat_search:
                if self.openai_client is None:
                    raise RuntimeError("OPENAI_API_KEY not set — cannot use live search model")
                stream = self.openai_client.chat.completions.create(
                    model=route.model,
                    messages=messages,
                    stream=True,
                    max_completion_tokens=self.realtime_max_completion_tokens,
                    web_search_options={"search_context_size": "low"},
                )
            else:
                create_params = {
                    "model": route.model,
                    "messages": messages,
                    "stream": True,
                    "temperature": 0.7,
                    **self._service_tier_param(),
                }
                if route.uses_realtime_tools:
                    create_params["max_completion_tokens"] = self.realtime_max_completion_tokens
                stream = self.client.chat.completions.create(**create_params)

            for chunk in stream:
                if self._is_cancelled(cancel_event):
                    raise LLMStreamCancelled()
                token = chunk if (is_gemini or is_openai_responses) else chunk.choices[0].delta.content
                if not token:
                    continue
                if is_openai:
                    token = _strip_openai_citations(token)
                    if not token:
                        continue
                full_response_text += token
                buffer += token

                # Split on spoken boundaries once there is enough context.
                has_boundary = any(punct in token for punct in (".", "?", "!", "\n")) or _ends_with_safe_soft_boundary(buffer)
                if has_boundary:
                    threshold = first_chunk_min if not first_yielded else min_chunk_size
                    if len(buffer) >= threshold and not _ends_with_protected_abbreviation(buffer):
                        chunk_text = _strip_openai_citations(buffer).strip() if is_openai else buffer.strip()
                        if chunk_text:
                            yield chunk_text
                        buffer = ""
                        first_yielded = True
                        continue

                # Soft-split the first answer fragment even without punctuation. This
                # starts TTS earlier on long LLM first sentences while later chunks
                # remain boundary-driven for natural prosody.
                if not first_yielded and len(buffer) >= first_soft_max:
                    split_at = buffer.rfind(" ")
                    if split_at >= first_chunk_min:
                        head = buffer[:split_at].strip()
                        tail = buffer[split_at:].strip()
                        if head and not _ends_with_protected_abbreviation(head):
                            chunk_text = _strip_openai_citations(head).strip() if is_openai else head
                            if chunk_text:
                                yield chunk_text
                            buffer = tail
                            first_yielded = True
                            continue

                # Emergency split if buffer gets too long (avoid latency spikes).
                if len(buffer) >= max_chars:
                    split_at = buffer.rfind(" ")
                    while split_at != -1:
                        head = buffer[:split_at].strip()
                        if head and not _ends_with_protected_abbreviation(head):
                            break
                        split_at = buffer.rfind(" ", 0, split_at)
                    if split_at != -1:
                        head = buffer[:split_at].strip()
                        tail = buffer[split_at:].strip()
                        if head:
                            chunk_text = _strip_openai_citations(head).strip() if is_openai else head
                            if chunk_text:
                                yield chunk_text
                            first_yielded = True
                        buffer = tail

            if buffer.strip() and not self._is_cancelled(cancel_event):
                chunk_text = _strip_openai_citations(buffer).strip() if is_openai else buffer.strip()
                if chunk_text:
                    yield chunk_text
            if self._is_cancelled(cancel_event):
                raise LLMStreamCancelled()
        finally:
            if stream is not None and self._is_cancelled(cancel_event):
                close_stream = getattr(stream, "close", None)
                if callable(close_stream):
                    with contextlib.suppress(Exception):
                        close_stream()

        print(" Done.")
        return _strip_openai_citations(full_response_text).strip() if is_openai else full_response_text

    def _stream_openai_responses_tokens(
        self,
        messages: List[Dict[str, str]],
        model_name: str,
        cancel_event=None,
    ) -> Generator[str, None, None]:
        if self.openai_client is None:
            raise RuntimeError("OPENAI_API_KEY not set — cannot use live search model")

        instructions, input_items = self._openai_responses_input(messages)
        create_params: dict[str, object] = {
            "model": model_name,
            "instructions": instructions or None,
            "input": input_items,
            "stream": True,
            "max_output_tokens": self.openai_realtime_max_output_tokens,
            "tools": [{"type": "web_search", "search_context_size": "low"}],
            "tool_choice": "required",
            "max_tool_calls": 1,
            "truncation": "auto",
            "store": False,
        }
        text_config = self._openai_text_config_for_model(model_name)
        if text_config:
            create_params["text"] = text_config
        reasoning_effort = self._openai_reasoning_effort_for_model(model_name)
        if reasoning_effort:
            create_params["reasoning"] = {"effort": reasoning_effort}

        stream = self.openai_client.responses.create(**create_params)
        try:
            for event in stream:
                if self._is_cancelled(cancel_event):
                    raise LLMStreamCancelled()
                event_type = getattr(event, "type", "")
                if event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        yield delta
                    continue
                if event_type == "error":
                    message = getattr(event, "message", "unknown OpenAI Responses API error")
                    raise RuntimeError(f"OpenAI Responses API error: {message}")
                if event_type == "response.failed":
                    raise RuntimeError(self._openai_response_failure_message(event))
        finally:
            if self._is_cancelled(cancel_event):
                close_stream = getattr(stream, "close", None)
                if callable(close_stream):
                    with contextlib.suppress(Exception):
                        close_stream()

        if self._is_cancelled(cancel_event):
            raise LLMStreamCancelled()

    @staticmethod
    def _openai_responses_input(messages: List[Dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
        instructions: list[str] = []
        input_items: list[dict[str, str]] = []
        for message in messages:
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            role = str(message.get("role", "user")).lower()
            if role == "system":
                instructions.append(content)
                continue
            if role not in {"user", "assistant", "developer"}:
                role = "user"
            input_items.append({"role": role, "content": content})
        if not input_items:
            input_items.append({"role": "user", "content": ""})
        return "\n".join(instructions), input_items

    @staticmethod
    def _openai_response_failure_message(event: object) -> str:
        response = getattr(event, "response", None)
        error = getattr(response, "error", None)
        message = getattr(error, "message", None)
        code = getattr(error, "code", None)
        if message and code:
            return f"OpenAI Responses API error {code}: {message}"
        if message:
            return f"OpenAI Responses API error: {message}"
        return "OpenAI Responses API response failed"

    def _stream_gemini_tokens(
        self,
        messages: List[Dict[str, str]],
        model: str,
        cancel_event=None,
        enable_search: bool = False,
    ) -> Generator[str, None, None]:
        if not self.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY not set — cannot use Gemini mode")

        model_name = model.removeprefix("models/")
        if enable_search:
            grounded_text = self._generate_gemini_grounded_text(messages, model_name)
            for token in self._text_stream_pieces(grounded_text):
                if self._is_cancelled(cancel_event):
                    raise LLMStreamCancelled()
                yield token
            return

        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{urllib.parse.quote(model_name, safe='')}:streamGenerateContent?alt=sse"
        )
        payload = self._gemini_payload(messages, model_name, enable_search=enable_search)
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.gemini_api_key,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                while not self._is_cancelled(cancel_event):
                    raw_line = response.readline()
                    if not raw_line:
                        break
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if not data or data == "[DONE]":
                        continue
                    chunk = json.loads(data)
                    error = chunk.get("error") if isinstance(chunk, dict) else None
                    if isinstance(error, dict):
                        message = error.get("message") or error
                        raise RuntimeError(f"Gemini API error: {message}")
                    for token in self._gemini_text_from_chunk(chunk):
                        yield token
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API error {exc.code}: {body[:500]}") from exc

        if self._is_cancelled(cancel_event):
            raise LLMStreamCancelled()

    def _generate_gemini_grounded_text(self, messages: List[Dict[str, str]], model_name: str) -> str:
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{urllib.parse.quote(model_name, safe='')}:generateContent"
        )
        payload = self._gemini_payload(messages, model_name, enable_search=True)
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.gemini_api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.gemini_search_timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            raise RuntimeError(
                f"Gemini API grounded search timed out after {self.gemini_search_timeout_seconds} seconds"
            ) from exc
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API error {exc.code}: {body[:500]}") from exc
        text = "".join(self._gemini_text_from_chunk(data)).strip()
        if not text:
            raise RuntimeError("Gemini API returned no grounded text")
        queries = self._gemini_grounding_queries(data)
        logger.info(
            "component=llm op=gemini_search status=ok model=%s queries=%s",
            model_name,
            "|".join(queries[:3]),
        )
        return text

    def _gemini_payload(
        self,
        messages: List[Dict[str, str]],
        model_name: str,
        *,
        enable_search: bool = False,
    ) -> dict[str, object]:
        system_parts: list[str] = []
        contents: list[dict[str, object]] = []
        for message in messages:
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            role = str(message.get("role", "user")).lower()
            if role == "system":
                system_parts.append(content)
                continue
            gemini_role = "model" if role == "assistant" else "user"
            part = {"text": content}
            if contents and contents[-1].get("role") == gemini_role:
                parts = contents[-1].setdefault("parts", [])
                if isinstance(parts, list):
                    parts.append(part)
            else:
                contents.append({"role": gemini_role, "parts": [part]})

        payload: dict[str, object] = {
            "contents": contents or [{"role": "user", "parts": [{"text": ""}]}],
            "generationConfig": {
                "maxOutputTokens": self.gemini_max_output_tokens,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n".join(system_parts)}]}
        if enable_search:
            payload["tools"] = [{"google_search": {}}]
        if model_name.startswith("gemini-3") and self.gemini_thinking_level:
            generation_config = payload["generationConfig"]
            if isinstance(generation_config, dict):
                generation_config["thinkingConfig"] = {"thinkingLevel": self.gemini_thinking_level}
        return payload

    @staticmethod
    def _gemini_text_from_chunk(chunk: object) -> Generator[str, None, None]:
        if not isinstance(chunk, dict):
            return
        candidates = chunk.get("candidates")
        if not isinstance(candidates, list):
            return
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text:
                    yield text

    @staticmethod
    def _gemini_grounding_queries(response: object) -> list[str]:
        if not isinstance(response, dict):
            return []
        queries: list[str] = []
        candidates = response.get("candidates")
        if not isinstance(candidates, list):
            return queries
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            metadata = candidate.get("groundingMetadata") or candidate.get("grounding_metadata")
            if not isinstance(metadata, dict):
                continue
            raw_queries = metadata.get("webSearchQueries") or metadata.get("web_search_queries") or []
            if not isinstance(raw_queries, list):
                continue
            queries.extend(str(query) for query in raw_queries if query)
        return queries

    @staticmethod
    def _text_stream_pieces(text: str) -> Generator[str, None, None]:
        for match in re.finditer(r"\S+\s*", text):
            yield match.group(0)

    def _messages_for_route(
        self,
        route: LLMRoute,
        messages: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        if route.uses_gemini_search:
            return self._messages_for_gemini_search(messages)
        if not route.uses_realtime_tools:
            return self._messages_with_live_context_hint(messages)
        realtime_prompt = (
            "You are running in a live search mode with web search/tools available. "
            "For current facts, weather, news, prices, schedules, or post-2023 information, "
            "use those tools instead of answering from memory. If a local answer needs a city "
            "or location and none is provided, ask for the location in one short sentence. "
            "If the live tool is unavailable or rate-limited, say that clearly. "
            "Start with the direct answer in the first sentence. "
            "Keep the answer concise and spoken-friendly. For temperatures, always include "
            "a number and write full unit words like 'eleven degrees Celsius' or "
            "'fifty two degrees Fahrenheit'; never write degree symbols, C, F, or C/F. "
            "Do not include citations, source links, markdown formatting, headings, bold text, bullets, tables, code blocks, hashtags, or raw URLs unless asked."
        )
        routed_messages = list(messages)
        if routed_messages and routed_messages[0].get("role") == "system":
            routed_messages[0] = {
                "role": "system",
                "content": f"{routed_messages[0]['content']} {realtime_prompt}",
            }
        else:
            routed_messages.insert(0, {"role": "system", "content": realtime_prompt})
        return routed_messages

    def _messages_for_gemini_search(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        grounded_prompt = (
            "You are running in Gemini Search mode with Google Search grounding enabled. "
            "For current facts, weather, news, prices, schedules, or post-2025 information, "
            "use Google Search grounding instead of answering from memory. "
            "Start with the direct answer in the first sentence. "
            "Keep the answer concise and spoken-friendly. "
            "Do not use markdown headings, bullets, tables, code blocks, hashtags, or raw URLs unless asked."
        )
        routed_messages = list(messages)
        if routed_messages and routed_messages[0].get("role") == "system":
            routed_messages[0] = {
                "role": "system",
                "content": f"{routed_messages[0]['content']} {grounded_prompt}",
            }
        else:
            routed_messages.insert(0, {"role": "system", "content": grounded_prompt})
        return routed_messages

    def _normalize_mode(self, mode: str | None) -> str:
        if not mode:
            return DEFAULT_LLM_MODE
        normalized = mode.strip().lower().replace("-", "_")
        aliases = {
            "legacy": LLM_MODE_LEGACY_FAST,
            "fast": LLM_MODE_LEGACY_FAST,
            "fresh": LLM_MODE_LIVE_SEARCH,
            "fresh_fast": LLM_MODE_LIVE_SEARCH,
            "research": LLM_MODE_LIVE_SEARCH,
            "web": LLM_MODE_LIVE_SEARCH,
            "search": LLM_MODE_LIVE_SEARCH,
            "live": LLM_MODE_LIVE_SEARCH,
            "live_search": LLM_MODE_LIVE_SEARCH,
            "current": LLM_MODE_LIVE_SEARCH,
            "current_info": LLM_MODE_LIVE_SEARCH,
            "gemini": LLM_MODE_GEMINI_FLASH,
            "gemini_flash": LLM_MODE_GEMINI_FLASH,
            "gemini_3": LLM_MODE_GEMINI_FLASH,
            "gemini_3_flash": LLM_MODE_GEMINI_FLASH,
            "gemini_3_flash_live": LLM_MODE_GEMINI_FLASH,
            "gemini_search": LLM_MODE_GEMINI_SEARCH,
            "gemini_grounded": LLM_MODE_GEMINI_SEARCH,
            "gemini_grounding": LLM_MODE_GEMINI_SEARCH,
            "gemini_live_search": LLM_MODE_GEMINI_SEARCH,
            "gemini_3_search": LLM_MODE_GEMINI_SEARCH,
            "gemini_3_flash_search": LLM_MODE_GEMINI_SEARCH,
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in SUPPORTED_LLM_MODES:
            logger.warning("component=llm op=normalize_mode status=unknown mode=%s", mode)
            return DEFAULT_LLM_MODE
        return normalized

    @staticmethod
    def _normalize_openai_reasoning_effort(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if normalized in {"", "off", "false", "0", "default"}:
            return None
        if normalized in OPENAI_REASONING_EFFORTS:
            return normalized
        logger.warning(
            "component=llm op=env_config status=invalid name=OPENAI_REASONING_EFFORT value=%s default=%s",
            value,
            DEFAULT_OPENAI_REASONING_EFFORT,
        )
        return DEFAULT_OPENAI_REASONING_EFFORT

    def _openai_reasoning_effort_for_model(self, model_name: str) -> str | None:
        if not model_name.startswith("gpt-5"):
            return None
        effort = self.openai_reasoning_effort
        if effort == "none" and not model_name.startswith("gpt-5.4"):
            return "low"
        return effort

    @staticmethod
    def _openai_text_config_for_model(model_name: str) -> dict[str, str] | None:
        if model_name.startswith("gpt-5"):
            return {"verbosity": "low"}
        return None

    def _auto_mode_for_prompt(self, user_input: str) -> str:
        if self._needs_current_info(user_input):
            if self._should_use_gemini_search_in_auto(user_input):
                logger.info(
                    "component=llm op=auto_route status=gemini_light_live input_chars=%s",
                    len(user_input),
                )
                return LLM_MODE_GEMINI_SEARCH
            return LLM_MODE_LIVE_SEARCH
        if self._should_continue_live_context(user_input):
            logger.info(
                "component=llm op=auto_route status=live_context input_chars=%s overlap=%s",
                len(user_input),
                self._live_context_overlap(user_input),
            )
            return LLM_MODE_LIVE_SEARCH
        return LLM_MODE_LEGACY_FAST

    @staticmethod
    def _is_cancelled(cancel_event) -> bool:
        return bool(cancel_event is not None and cancel_event.is_set())

    def _candidate_routes(self, route: LLMRoute, user_input: str) -> list[LLMRoute]:
        candidates = [route]
        if route.uses_gemini_search and self.live_search_model:
            candidates.append(
                LLMRoute(
                    requested_mode=route.requested_mode,
                    resolved_mode=LLM_MODE_LIVE_SEARCH,
                    model=self.live_search_model,
                )
            )
        if route.model != LEGACY_FAST_MODEL and not route.uses_gemini and not self._requires_live_route(user_input):
            candidates.append(
                LLMRoute(
                    requested_mode=route.requested_mode,
                    resolved_mode=LLM_MODE_LEGACY_FAST,
                    model=LEGACY_FAST_MODEL,
                )
            )

        deduped: list[LLMRoute] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate.cache_key in seen:
                continue
            seen.add(candidate.cache_key)
            deduped.append(candidate)
        return deduped

    def _should_use_gemini_search_in_auto(self, user_input: str) -> bool:
        if not self.auto_gemini_search_enabled or not self.gemini_api_key:
            return False
        text = user_input.strip()
        if len(text) > 180:
            return False
        if any(pattern.search(text) for pattern in _AUTO_GPT_REQUIRED):
            return False
        return any(pattern.search(text) for pattern in _AUTO_GEMINI_LIGHT_LIVE)

    def _requires_live_route(self, user_input: str) -> bool:
        return self._needs_current_info(user_input) or self._should_continue_live_context(user_input)

    def _needs_current_info(self, user_input: str) -> bool:
        # Tier 1: strong signals — always live, no override.
        if any(p.search(user_input) for p in _LIVE_STRONG):
            return True
        # Tier 2: weak signals — live unless a negative (tier 3) fires.
        if any(p.search(user_input) for p in _LIVE_WEAK):
            if not any(p.search(user_input) for p in _LEGACY_PREFER):
                return True
        return False

    def _mark_live_context(self, reason: str, *, user_input: str = "", assistant_text: str = "") -> None:
        if max(self.auto_live_followup_turns, self.auto_live_topic_turns) <= 0:
            return
        self._last_live_turn_index = self._turn_index
        self._last_live_until = time.monotonic() + float(self.auto_live_followup_ttl_seconds)
        self._last_live_context_terms = self._extract_live_context_terms(user_input, assistant_text)
        self._last_live_context_summary = self._build_live_context_summary(user_input, assistant_text)
        logger.info(
            "component=llm op=live_context status=marked reason=%s turn=%s ttl_sec=%s terms=%s",
            reason,
            self._turn_index,
            self.auto_live_followup_ttl_seconds,
            ",".join(sorted(self._last_live_context_terms)),
        )

    def _should_continue_live_context(self, user_input: str) -> bool:
        if not self._has_recent_live_topic_context():
            return False

        text = user_input.strip()
        if not text:
            return False
        # New creative/explanatory/coding tasks can use the shared conversation history
        # in the fast model instead of paying live-search latency again.
        if any(p.search(text) for p in _LEGACY_PREFER):
            return False
        if len(text) > 220 and not any(p.search(text) for p in _LIVE_FOLLOWUP[:4]):
            return False

        overlap = self._live_context_overlap(text)
        has_live_signal = self._has_live_context_followup_signal(text)
        is_question_like = self._is_question_like(text)
        if any(p.search(text) for p in _LIVE_FOLLOWUP[:4]):
            return True
        if any(p.search(text) for p in _LIVE_FOLLOWUP[4:]) and (is_question_like or has_live_signal or overlap > 0):
            return True
        if has_live_signal and (is_question_like or overlap > 0):
            return True
        if is_question_like and overlap >= 2:
            return True
        if len(text.split()) <= 12 and overlap > 0 and has_live_signal:
            return True
        return False

    def _has_recent_live_context(self, max_turns: int | None = None) -> bool:
        if self._last_live_turn_index is None:
            return False
        if time.monotonic() > self._last_live_until:
            return False
        allowed_turns = self.auto_live_followup_turns if max_turns is None else max_turns
        if allowed_turns <= 0:
            return False
        return self._turn_index - self._last_live_turn_index <= allowed_turns

    def _has_recent_live_topic_context(self) -> bool:
        return self._has_recent_live_context(max(self.auto_live_followup_turns, self.auto_live_topic_turns))

    @staticmethod
    def _has_live_context_followup_signal(text: str) -> bool:
        return any(pattern.search(text) for pattern in _LIVE_CONTEXT_FOLLOWUP)

    @staticmethod
    def _is_question_like(text: str) -> bool:
        return "?" in text or bool(_QUESTION_LIKE_PATTERN.search(text))

    def _live_context_overlap(self, text: str) -> int:
        if not self._has_recent_live_topic_context() or not self._last_live_context_terms:
            return 0
        return len(self._extract_live_context_terms(text) & self._last_live_context_terms)

    @staticmethod
    def _extract_live_context_terms(*texts: str) -> set[str]:
        combined = " ".join(text for text in texts if text)
        combined = re.sub(r"\bU\.?\s*S\.?\b", "US", combined, flags=re.IGNORECASE)
        combined = re.sub(r"\bU\.?\s*K\.?\b", "UK", combined, flags=re.IGNORECASE)
        terms: set[str] = set()
        for token in re.findall(r"[A-Za-z][A-Za-z'-]*", combined):
            normalized = token.lower().removesuffix("'s")
            if normalized in {"us", "uk", "ai"}:
                terms.add(normalized)
            elif len(normalized) >= 3 and normalized not in _LIVE_CONTEXT_STOPWORDS:
                terms.add(normalized)
        return terms

    @staticmethod
    def _build_live_context_summary(user_input: str, assistant_text: str) -> str:
        user = re.sub(r"\s+", " ", user_input).strip()
        assistant = re.sub(r"\s+", " ", assistant_text).strip()
        summary = f"User asked: {user} Live answer: {assistant}".strip()
        return summary[:1000]

    def _messages_with_live_context_hint(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        routed_messages = list(messages)
        if not self._has_recent_live_topic_context() or not self._last_live_context_summary:
            return routed_messages
        live_context_prompt = (
            "Recent live-search information from this conversation is authoritative. "
            "Do not contradict or correct it using older training data. "
            "Use it when relevant, and if the user asks for new current facts, rely on live search routing. "
            f"Recent live context: {self._last_live_context_summary}"
        )
        if routed_messages and routed_messages[0].get("role") == "system":
            routed_messages[0] = {
                "role": "system",
                "content": f"{routed_messages[0]['content']} {live_context_prompt}",
            }
        else:
            routed_messages.insert(0, {"role": "system", "content": live_context_prompt})
        return routed_messages

    def _needs_location_for_current_query(self, user_input: str) -> bool:
        if not WEATHER_QUERY_PATTERN.search(user_input):
            return False
        if "near me" in user_input or "my location" in user_input or "here" in user_input:
            return True
        return LOCATION_PREPOSITION_PATTERN.search(user_input) is None

    def _direct_weather_answer(self, user_input: str) -> str | None:
        if not WEATHER_QUERY_PATTERN.search(user_input):
            return None
        location = self._extract_weather_location(user_input)
        if not location:
            return "Which city or location should I check?"
        try:
            place = self._geocode_location(location)
            if place is None:
                return f"I could not find a weather location for {location}."
            weather = self._fetch_current_weather(place["latitude"], place["longitude"])
            return self._format_weather_answer(user_input, place, weather)
        except Exception:
            logger.exception(
                "component=llm op=direct_weather status=error input_chars=%s",
                len(user_input),
            )
            return "I could not reach the live weather service right now. Try again in a moment."

    def _extract_weather_location(self, user_input: str) -> str | None:
        match = WEATHER_LOCATION_PATTERN.search(user_input)
        if not match:
            return None
        location = match.group("location")
        location = re.sub(
            r"\s+(?:"
            r"I(?=['\s]|$)"                                   # I'm, I'll, I've, I'd, I (alone)
            r"|we\b|you\b|they\b|he\b|she\b|it\b"            # other pronouns
            r"|and\b|but\b|so\b|or\b"                        # conjunctions
            r"|to\s+\w+"                                      # infinitive: "to go", "to walk"
            r"|for\s+(?:my|our|a|an|the)\b"                  # "for my walk", "for a run"
            r"|trying\b|planning\b|going\b|looking\b"        # common participles
            r"|want\b|need\b|should\b|would\b|could\b"       # modal verbs
            r"|tell\b|give\b|show\b|please\b"
            r"|right\s+now\b|now\b|today\b|currently\b|tonight\b|tomorrow\b"
            r"|detail(?:ed)?\b"
            r").*$",
            "",
            location,
            flags=re.IGNORECASE,
        )
        location = re.sub(r"\s+", " ", location).strip(" .,?!'")
        # Hard cap: real city names are at most 4 words (e.g. "Salt Lake City").
        words = location.split()
        if len(words) > 4:
            location = " ".join(words[:4])
        return location or None

    def _geocode_location(self, location: str) -> dict[str, object] | None:
        data = self._fetch_json(
            "https://geocoding-api.open-meteo.com/v1/search",
            {
                "name": location,
                "count": "1",
                "language": "en",
                "format": "json",
            },
        )
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list) or not results:
            return None
        result = results[0]
        if not isinstance(result, dict):
            return None
        return result

    def _fetch_current_weather(self, latitude: object, longitude: object) -> dict[str, object]:
        return self._fetch_json(
            "https://api.open-meteo.com/v1/forecast",
            {
                "latitude": str(latitude),
                "longitude": str(longitude),
                "current": (
                    "temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "precipitation,rain,showers,snowfall,weather_code,cloud_cover,"
                    "wind_speed_10m,wind_gusts_10m"
                ),
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "forecast_days": "1",
                "timezone": "auto",
            },
        )

    @staticmethod
    def _fetch_json(url: str, params: dict[str, str], timeout: float = 3.0) -> dict[str, object]:
        query = urllib.parse.urlencode(params)
        request = urllib.request.Request(
            f"{url}?{query}",
            headers={"User-Agent": "PixelHolo/1.0"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("Expected JSON object")
        return data

    def _format_weather_answer(
        self,
        user_input: str,
        place: dict[str, object],
        weather: dict[str, object],
    ) -> str:
        current = weather.get("current")
        daily = weather.get("daily")
        if not isinstance(current, dict):
            raise ValueError("Open-Meteo response missing current weather")
        if not isinstance(daily, dict):
            daily = {}

        location_name = self._format_place_name(place)
        temp_c = self._number_or_none(current.get("temperature_2m"))
        feels_c = self._number_or_none(current.get("apparent_temperature"))
        humidity = self._number_or_none(current.get("relative_humidity_2m"))
        wind = self._number_or_none(current.get("wind_speed_10m"))
        gusts = self._number_or_none(current.get("wind_gusts_10m"))
        precipitation = self._number_or_none(current.get("precipitation"))
        weather_code = self._int_or_none(current.get("weather_code"))
        high_c = self._first_number_or_none(daily.get("temperature_2m_max"))
        low_c = self._first_number_or_none(daily.get("temperature_2m_min"))
        precip_probability = self._first_number_or_none(daily.get("precipitation_probability_max"))

        condition = WEATHER_CODE_DESCRIPTIONS.get(weather_code, "current conditions")
        temp_phrase = self._temperature_phrase(temp_c)
        answer = f"In {location_name} right now, it is about {temp_phrase} with {condition}."

        details: list[str] = []
        if feels_c is not None:
            details.append(f"It feels like {self._temperature_phrase(feels_c)}")
        if humidity is not None:
            details.append(f"Humidity is {round(humidity)} percent")
        if wind is not None:
            wind_text = f"Wind is {round(wind)} kilometers per hour"
            if gusts is not None and gusts > wind:
                wind_text += f", with gusts up to {round(gusts)} kilometers per hour"
            details.append(wind_text)
        if precipitation is not None and precipitation > 0:
            details.append(f"Recent precipitation is {precipitation:g} millimeters")
        if high_c is not None and low_c is not None:
            details.append(
                f"Today's high is {self._temperature_phrase(high_c)} and the low is {self._temperature_phrase(low_c)}"
            )
        if precip_probability is not None:
            details.append(f"The chance of precipitation today is about {round(precip_probability)} percent")

        wants_detail = bool(WEATHER_DETAIL_PATTERN.search(user_input))
        if details:
            if wants_detail:
                answer = f"{answer} " + ". ".join(details) + "."
            else:
                answer = f"{answer} {details[0]}."
        return answer

    @staticmethod
    def _format_place_name(place: dict[str, object]) -> str:
        parts = [
            str(place.get("name") or "").strip(),
            str(place.get("admin1") or "").strip(),
        ]
        return ", ".join(part for part in parts if part) or "that location"

    @staticmethod
    def _number_or_none(value: object) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _first_number_or_none(self, value: object) -> float | None:
        if isinstance(value, list) and value:
            return self._number_or_none(value[0])
        return None

    @staticmethod
    def _int_or_none(value: object) -> int | None:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _temperature_phrase(self, celsius: float | None) -> str:
        if celsius is None:
            return "an unknown temperature"
        fahrenheit = celsius * 9 / 5 + 32
        return f"{round(celsius)} degrees Celsius, or {round(fahrenheit)} degrees Fahrenheit"

    def _friendly_error_message(self, user_input: str, exc: Exception) -> str:
        if self._is_gemini_error(exc):
            error_text = str(exc).lower()
            if "prepayment credits are depleted" in error_text or "resource_exhausted" in error_text:
                return "Gemini API billing credits are depleted. Add credits in Google AI Studio, then try Gemini again."
            if "grounded search timed out" in error_text:
                return "Gemini Search timed out. Try again, or use Live Search for this current question."
            if "api key not valid" in error_text or "permission_denied" in error_text:
                return "Gemini API rejected the key. Check GEMINI_API_KEY in the backend environment."
            return "Gemini API is unavailable right now. Try again in a moment."
        if self._is_request_too_large(exc):
            return (
                "The live search request became too large. "
                "Please ask again; I will use a smaller conversation context."
            )
        if self._requires_live_route(user_input):
            retry_seconds = self._retry_after_seconds(exc)
            if retry_seconds is not None:
                return (
                    "The live search model is rate-limited right now. "
                    f"Try again in about {retry_seconds} seconds."
                )
            return "The live search model is unavailable right now. Try again in a moment."
        return f"Error calling LLM provider: {str(exc)}"

    def _retry_after_seconds(self, exc: Exception) -> int | None:
        match = RETRY_AFTER_PATTERN.search(str(exc))
        if not match:
            return None
        try:
            return max(1, round(float(match.group(1))))
        except ValueError:
            return None

    def _remember_realtime_rate_limit(self, route: LLMRoute, exc: Exception) -> None:
        if not route.uses_realtime_tools:
            return
        retry_seconds = self._retry_after_seconds(exc)
        if retry_seconds is None:
            return
        self._realtime_rate_limit_until = max(
            self._realtime_rate_limit_until,
            time.monotonic() + retry_seconds,
        )

    def _realtime_rate_limit_wait_seconds(self) -> int:
        wait_seconds = self._realtime_rate_limit_until - time.monotonic()
        if wait_seconds <= 0:
            return 0
        return max(1, round(wait_seconds))

    def _short_error_message(self, exc: Exception) -> str:
        text = str(exc).strip().replace("\n", " ")
        if len(text) <= 240:
            return text
        return f"{text[:237]}..."

    def _is_request_too_large(self, exc: Exception) -> bool:
        return bool(REQUEST_TOO_LARGE_PATTERN.search(str(exc)))

    @staticmethod
    def _is_gemini_error(exc: Exception) -> bool:
        return "gemini api" in str(exc).lower() or "GEMINI_API_KEY" in str(exc)

    @staticmethod
    def _clip_content(content: str, max_chars: int) -> str:
        if len(content) <= max_chars:
            return content
        head = max(1, max_chars // 2)
        tail = max(1, max_chars - head)
        return f"{content[:head]}\n...[truncated]...\n{content[-tail:]}"

    @staticmethod
    def _messages_char_count(messages: List[Dict[str, str]]) -> int:
        return sum(len(str(message.get("content", ""))) for message in messages)

    def _bounded_messages(
        self,
        messages: List[Dict[str, str]],
        *,
        max_message_chars: int,
        max_total_chars: int,
    ) -> List[Dict[str, str]]:
        bounded = [
            {
                "role": str(message.get("role", "user")),
                "content": self._clip_content(str(message.get("content", "")), max_message_chars),
            }
            for message in messages
        ]
        while self._messages_char_count(bounded) > max_total_chars and len(bounded) > 2:
            remove_idx = 1 if bounded[0].get("role") == "system" else 0
            bounded.pop(remove_idx)
        if self._messages_char_count(bounded) > max_total_chars:
            bounded = [
                {
                    "role": message["role"],
                    "content": self._clip_content(message["content"], max(200, max_total_chars // len(bounded))),
                }
                for message in bounded
            ]
        return bounded

    def _conversation_window(self, route: LLMRoute | None = None) -> List[Dict[str, str]]:
        if not self.history:
            return []
        first = self.history[0]
        uses_realtime_tools = bool(route and route.uses_current_tools)
        max_messages = self.realtime_history_messages if uses_realtime_tools else MAX_HISTORY_MESSAGES
        max_message_chars = self.realtime_max_message_chars if uses_realtime_tools else self.max_message_chars
        max_total_chars = self.realtime_max_history_chars if uses_realtime_tools else self.max_history_chars
        if first.get("role") == "system":
            messages = [first] + self.history[1:][-max_messages:]
        else:
            messages = self.history[-max_messages:]
        return self._bounded_messages(
            messages,
            max_message_chars=max_message_chars,
            max_total_chars=max_total_chars,
        )

    def _service_tier_param(self) -> dict[str, str]:
        if not self.service_tier or self.service_tier in {"default", "none", "off"}:
            return {}
        return {"service_tier": self.service_tier}
