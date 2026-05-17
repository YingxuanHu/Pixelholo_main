import logging
import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
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


LLM_MODE_LEGACY_FAST = "legacy_fast"
LLM_MODE_LIVE_SEARCH = "live_search"
LLM_MODE_AUTO = "auto"
DEFAULT_LLM_MODE = LLM_MODE_LEGACY_FAST

LEGACY_FAST_MODEL = "llama-3.1-8b-instant"
LIVE_SEARCH_MODEL = "gpt-4o-mini-search-preview"
OPENAI_MODELS: frozenset[str] = frozenset({"gpt-4o-mini-search-preview", "gpt-4o-search-preview"})
WARMUP_MAX_COMPLETION_TOKENS = 16
REALTIME_MAX_COMPLETION_TOKENS = 320
MAX_HISTORY_MESSAGES = 8
DEFAULT_REALTIME_MAX_COMPLETION_TOKENS = REALTIME_MAX_COMPLETION_TOKENS
DEFAULT_REALTIME_HISTORY_MESSAGES = 2
DEFAULT_MAX_MESSAGE_CHARS = 1200
DEFAULT_MAX_HISTORY_CHARS = 6000
DEFAULT_REALTIME_MAX_MESSAGE_CHARS = 800
DEFAULT_REALTIME_MAX_HISTORY_CHARS = 2600
RETRY_AFTER_PATTERN = re.compile(r"try again in ([0-9.]+)s", re.IGNORECASE)
REQUEST_TOO_LARGE_PATTERN = re.compile(r"request entity too large", re.IGNORECASE)
# Matches OpenAI inline citation annotations like ([domain.com](https://...))
_OPENAI_CITATION_PATTERN = re.compile(r"\s*\(\[[^\]]*\]\([^)]*\)\)")


def _strip_openai_citations(text: str) -> str:
    return _OPENAI_CITATION_PATTERN.sub("", text)

SUPPORTED_LLM_MODES = {
    LLM_MODE_LEGACY_FAST,
    LLM_MODE_LIVE_SEARCH,
    LLM_MODE_AUTO,
}

MODEL_BY_MODE = {
    LLM_MODE_LEGACY_FAST: LEGACY_FAST_MODEL,
    LLM_MODE_LIVE_SEARCH: LIVE_SEARCH_MODEL,
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

@dataclass(frozen=True)
class LLMRoute:
    requested_mode: str
    resolved_mode: str
    model: str

    @property
    def uses_realtime_tools(self) -> bool:
        return self.model in OPENAI_MODELS

    @property
    def cache_key(self) -> str:
        return f"{self.resolved_mode}:{self.model}"


class LLMService:
    def __init__(self, system_prompt: str = "You are a helpful, concise AI assistant."):
        load_dotenv()
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("WARNING: GROQ_API_KEY not found in environment variables.")

        self.client = Groq(api_key=api_key)
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            print("WARNING: OPENAI_API_KEY not found — live_search mode will fall back to legacy_fast.")
        self.openai_client: OpenAIClient | None = OpenAIClient(api_key=openai_api_key) if openai_api_key else None
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
    ) -> Generator[str, None, None]:
        """
        Sends text to Groq and yields complete spoken-friendly chunks as they are generated.
        """
        self.history.append({"role": "user", "content": user_input})
        route = self.resolve_route(user_input=user_input, mode=mode, model=model)

        rate_limit_wait = self._realtime_rate_limit_wait_seconds()
        if route.uses_realtime_tools and rate_limit_wait > 0:
            if self._needs_current_info(user_input):
                message = (
                    "The live search model is rate-limited right now. "
                    f"Try again in about {rate_limit_wait} seconds."
                )
                logger.info(
                    "component=llm op=groq_chat status=skipped_rate_limit mode=%s resolved_mode=%s model=%s retry_after_seconds=%s",
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
        for idx, candidate in enumerate(candidates):
            yielded_chunks = 0
            try:
                route_stream = self._stream_from_route(
                    candidate,
                    min_words=min_words,
                    min_chars=min_chars,
                    max_chars=max_chars,
                )
                while True:
                    try:
                        sentence = next(route_stream)
                    except StopIteration as stop:
                        full_response_text = stop.value or ""
                        if not full_response_text.strip() and yielded_chunks == 0:
                            raise RuntimeError(f"{candidate.model} returned no text")
                        self._warmed_routes.add(candidate.cache_key)
                        break
                    yielded_chunks += 1
                    yield sentence
                break
            except Exception as exc:
                self._remember_realtime_rate_limit(candidate, exc)
                next_candidate = candidates[idx + 1] if idx + 1 < len(candidates) else None
                if yielded_chunks or next_candidate is None:
                    retry_after_seconds = self._retry_after_seconds(exc)
                    logger.exception(
                        "component=llm op=groq_chat status=error mode=%s resolved_mode=%s model=%s input_chars=%s request_too_large=%s retry_after_seconds=%s",
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
                    "component=llm op=groq_chat_fallback status=start from_model=%s to_model=%s error=%s",
                    candidate.model,
                    next_candidate.model,
                    self._short_error_message(exc),
                )
                continue

        if full_response_text:
            self.history.append({"role": "assistant", "content": full_response_text})

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
        route_model = model.strip() if model and model.strip() else MODEL_BY_MODE[resolved_mode]
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
    ) -> Generator[str, None, str]:
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
        if route.model in OPENAI_MODELS:
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

        full_response_text = ""
        buffer = ""
        min_chunk_size = max(min_chars, min_words * 2)
        # First sentence uses a lower threshold so TTS starts sooner.
        first_chunk_min = max(20, min_chunk_size // 2)
        first_yielded = False
        is_openai = route.model in OPENAI_MODELS

        print(f"{route.model} Thinking...", end="", flush=True)

        for chunk in stream:
            token = chunk.choices[0].delta.content
            if not token:
                continue
            if is_openai:
                token = _strip_openai_citations(token)
                if not token:
                    continue
            full_response_text += token
            buffer += token

            # Split only on strong punctuation once we have enough context.
            if any(punct in token for punct in (".", "?", "!", "\n")):
                threshold = first_chunk_min if not first_yielded else min_chunk_size
                if len(buffer) >= threshold:
                    yield buffer.strip()
                    buffer = ""
                    first_yielded = True
                    continue

            # Emergency split if buffer gets too long (avoid latency spikes).
            if len(buffer) >= max_chars:
                last_space = buffer.rfind(" ")
                if last_space != -1:
                    head = buffer[:last_space].strip()
                    tail = buffer[last_space:].strip()
                    if head:
                        yield head
                        first_yielded = True
                    buffer = tail

        if buffer.strip():
            yield buffer.strip()

        print(" Done.")
        return full_response_text

    def _messages_for_route(
        self,
        route: LLMRoute,
        messages: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        if not route.uses_realtime_tools:
            return list(messages)
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
            "Do not include raw URLs unless asked."
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
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in SUPPORTED_LLM_MODES:
            logger.warning("component=llm op=normalize_mode status=unknown mode=%s", mode)
            return DEFAULT_LLM_MODE
        return normalized

    def _auto_mode_for_prompt(self, user_input: str) -> str:
        if self._needs_current_info(user_input):
            return LLM_MODE_LIVE_SEARCH
        return LLM_MODE_LEGACY_FAST

    def _candidate_routes(self, route: LLMRoute, user_input: str) -> list[LLMRoute]:
        candidates = [route]
        if route.model != LEGACY_FAST_MODEL:
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

    def _needs_current_info(self, user_input: str) -> bool:
        # Tier 1: strong signals — always live, no override.
        if any(p.search(user_input) for p in _LIVE_STRONG):
            return True
        # Tier 2: weak signals — live unless a negative (tier 3) fires.
        if any(p.search(user_input) for p in _LIVE_WEAK):
            if not any(p.search(user_input) for p in _LEGACY_PREFER):
                return True
        return False

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
        if self._is_request_too_large(exc):
            return (
                "The live search request became too large. "
                "Please ask again; I will use a smaller conversation context."
            )
        if self._needs_current_info(user_input):
            retry_seconds = self._retry_after_seconds(exc)
            if retry_seconds is not None:
                return (
                    "The live search model is rate-limited right now. "
                    f"Try again in about {retry_seconds} seconds."
                )
            return "The live search model is unavailable right now. Try again in a moment."
        return f"Error calling Groq: {str(exc)}"

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
        uses_realtime_tools = bool(route and route.uses_realtime_tools)
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
