import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, Generator, List

from dotenv import load_dotenv
from groq import Groq


logger = logging.getLogger("pixelholo.llm")

LLM_MODE_LEGACY_FAST = "legacy_fast"
LLM_MODE_FRESH_FAST = "fresh_fast"
LLM_MODE_RESEARCH = "research"
LLM_MODE_AUTO = "auto"
DEFAULT_LLM_MODE = LLM_MODE_LEGACY_FAST

LEGACY_FAST_MODEL = "llama-3.1-8b-instant"
FRESH_FAST_MODEL = "groq/compound-mini"
RESEARCH_MODEL = "groq/compound"
WARMUP_MAX_COMPLETION_TOKENS = 16
REALTIME_MAX_COMPLETION_TOKENS = 320
MAX_HISTORY_MESSAGES = 8
RETRY_AFTER_PATTERN = re.compile(r"try again in ([0-9.]+)s", re.IGNORECASE)

SUPPORTED_LLM_MODES = {
    LLM_MODE_LEGACY_FAST,
    LLM_MODE_FRESH_FAST,
    LLM_MODE_RESEARCH,
    LLM_MODE_AUTO,
}

MODEL_BY_MODE = {
    LLM_MODE_LEGACY_FAST: LEGACY_FAST_MODEL,
    LLM_MODE_FRESH_FAST: FRESH_FAST_MODEL,
    LLM_MODE_RESEARCH: FRESH_FAST_MODEL,
}

FRESHNESS_KEYWORDS = {
    "after 2023",
    "breaking",
    "current",
    "currently",
    "exchange rate",
    "forecast",
    "latest",
    "live",
    "market",
    "news",
    "now",
    "price",
    "recent",
    "schedule",
    "score",
    "stock",
    "today",
    "tomorrow",
    "tonight",
    "traffic",
    "up to date",
    "up-to-date",
    "weather",
    "yesterday",
}

FRESHNESS_YEAR_PATTERN = re.compile(r"\b20(2[4-9]|3[0-9])\b")
LOCATION_PREPOSITION_PATTERN = re.compile(
    r"\b(?:in|near|for|at)\s+"
    r"(?!today\b|tomorrow\b|tonight\b|now\b|right\b|current\b|the\b|a\b|an\b)"
    r"[a-z][a-z .,'-]{2,}",
    re.IGNORECASE,
)
LOCAL_INFO_KEYWORDS = {
    "forecast",
    "traffic",
    "weather",
}
SIMPLE_CURRENT_INFO_KEYWORDS = {
    "exchange rate",
    "forecast",
    "market",
    "price",
    "schedule",
    "score",
    "stock",
    "traffic",
    "weather",
}
DEEP_RESEARCH_KEYWORDS = {
    "analyze",
    "analysis",
    "compare",
    "comprehensive",
    "deep dive",
    "evaluate",
    "investigate",
    "multiple sources",
    "pros and cons",
    "rank",
    "tradeoff",
    "tradeoffs",
}


@dataclass(frozen=True)
class LLMRoute:
    requested_mode: str
    resolved_mode: str
    model: str

    @property
    def uses_realtime_tools(self) -> bool:
        return self.model in {FRESH_FAST_MODEL, RESEARCH_MODEL}

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
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        self._warmed_routes: set[str] = set()
        self.service_tier = os.environ.get("GROQ_SERVICE_TIER", "").strip().lower()

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

        if route.uses_realtime_tools and self._needs_location_for_current_query(user_input):
            clarification = "Which city or location should I check?"
            self.history.append({"role": "assistant", "content": clarification})
            yield clarification
            return

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
                next_candidate = candidates[idx + 1] if idx + 1 < len(candidates) else None
                if yielded_chunks or next_candidate is None:
                    logger.exception(
                        "component=llm op=groq_chat status=error mode=%s resolved_mode=%s model=%s input_chars=%s",
                        candidate.requested_mode,
                        candidate.resolved_mode,
                        candidate.model,
                        len(user_input),
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
        messages = self._messages_for_route(route, self._conversation_window())
        create_params = {
            "model": route.model,
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
            **self._service_tier_param(),
        }
        if route.uses_realtime_tools:
            create_params["max_completion_tokens"] = REALTIME_MAX_COMPLETION_TOKENS
        stream = self.client.chat.completions.create(**create_params)

        full_response_text = ""
        buffer = ""
        min_chunk_size = max(min_chars, min_words * 2)

        print(f"{route.model} Thinking...", end="", flush=True)

        for chunk in stream:
            token = chunk.choices[0].delta.content
            if not token:
                continue
            full_response_text += token
            buffer += token

            # Split only on strong punctuation once we have enough context.
            if any(punct in token for punct in (".", "?", "!", "\n")):
                if len(buffer) >= min_chunk_size:
                    yield buffer.strip()
                    buffer = ""
                    continue

            # Emergency split if buffer gets too long (avoid latency spikes).
            if len(buffer) >= max_chars:
                last_space = buffer.rfind(" ")
                if last_space != -1:
                    head = buffer[:last_space].strip()
                    tail = buffer[last_space:].strip()
                    if head:
                        yield head
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
            "You are running in a real-time research mode with web search/tools available. "
            "For current facts, weather, news, prices, schedules, or post-2023 information, "
            "use those tools instead of answering from memory. If a local answer needs a city "
            "or location and none is provided, ask for the location in one short sentence. "
            "If the live tool is unavailable or rate-limited, say that clearly. "
            "Start with the direct answer in the first sentence. "
            "Keep the answer concise and spoken-friendly. Do not include raw URLs unless asked."
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
            "fresh": LLM_MODE_FRESH_FAST,
            "web": LLM_MODE_FRESH_FAST,
            "search": LLM_MODE_FRESH_FAST,
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in SUPPORTED_LLM_MODES:
            logger.warning("component=llm op=normalize_mode status=unknown mode=%s", mode)
            return DEFAULT_LLM_MODE
        return normalized

    def _auto_mode_for_prompt(self, user_input: str) -> str:
        if self._needs_current_info(user_input):
            return LLM_MODE_FRESH_FAST
        return LLM_MODE_LEGACY_FAST

    def _candidate_routes(self, route: LLMRoute, user_input: str) -> list[LLMRoute]:
        if route.requested_mode == LLM_MODE_RESEARCH and route.model in {FRESH_FAST_MODEL, RESEARCH_MODEL}:
            mini_route = LLMRoute(
                requested_mode=route.requested_mode,
                resolved_mode=LLM_MODE_FRESH_FAST,
                model=FRESH_FAST_MODEL,
            )
            full_route = LLMRoute(
                requested_mode=route.requested_mode,
                resolved_mode=LLM_MODE_RESEARCH,
                model=RESEARCH_MODEL,
            )
            prefers_full = self._prefers_full_compound(user_input)
            candidates = [full_route, mini_route] if prefers_full else [mini_route]
        else:
            candidates = [route]
        if route.model != LEGACY_FAST_MODEL and not self._needs_current_info(user_input):
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
        text = user_input.lower()
        return any(keyword in text for keyword in FRESHNESS_KEYWORDS) or bool(
            FRESHNESS_YEAR_PATTERN.search(text)
        )

    def _needs_location_for_current_query(self, user_input: str) -> bool:
        text = user_input.lower()
        if not any(keyword in text for keyword in LOCAL_INFO_KEYWORDS):
            return False
        if "near me" in text or "my location" in text or "here" in text:
            return True
        return LOCATION_PREPOSITION_PATTERN.search(text) is None

    def _prefers_compound_mini(self, user_input: str) -> bool:
        text = user_input.lower()
        return any(keyword in text for keyword in SIMPLE_CURRENT_INFO_KEYWORDS)

    def _prefers_full_compound(self, user_input: str) -> bool:
        text = user_input.lower()
        if self._prefers_compound_mini(user_input):
            return False
        return any(keyword in text for keyword in DEEP_RESEARCH_KEYWORDS)

    def _friendly_error_message(self, user_input: str, exc: Exception) -> str:
        if self._needs_current_info(user_input):
            retry_seconds = self._retry_after_seconds(exc)
            if retry_seconds is not None:
                return (
                    "The live research model is rate-limited right now. "
                    f"Try again in about {retry_seconds} seconds."
                )
            return "The live research model is unavailable right now. Try again in a moment."
        return f"Error calling Groq: {str(exc)}"

    def _retry_after_seconds(self, exc: Exception) -> int | None:
        match = RETRY_AFTER_PATTERN.search(str(exc))
        if not match:
            return None
        try:
            return max(1, round(float(match.group(1))))
        except ValueError:
            return None

    def _short_error_message(self, exc: Exception) -> str:
        text = str(exc).strip().replace("\n", " ")
        if len(text) <= 240:
            return text
        return f"{text[:237]}..."

    def _conversation_window(self) -> List[Dict[str, str]]:
        if not self.history:
            return []
        first = self.history[0]
        if first.get("role") == "system":
            return [first] + self.history[1:][-MAX_HISTORY_MESSAGES:]
        return self.history[-MAX_HISTORY_MESSAGES:]

    def _service_tier_param(self) -> dict[str, str]:
        if not self.service_tier or self.service_tier in {"default", "none", "off"}:
            return {}
        return {"service_tier": self.service_tier}
