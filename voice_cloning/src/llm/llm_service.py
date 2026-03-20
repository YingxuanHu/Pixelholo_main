import logging
import os
import re
from typing import Generator, List, Dict

from dotenv import load_dotenv
from groq import Groq

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None


logger = logging.getLogger("pixelholo.llm")

DEFAULT_CHAT_MODEL = "llama-3.1-8b-instant"
DEFAULT_LIVE_MODEL = "gpt-4o-mini-search-preview"
LIVE_ROUTING_TRUE = {"1", "true", "yes", "on"}
TIME_SENSITIVE_PATTERNS = (
    re.compile(r"\b(today|tonight|tomorrow|yesterday|currently|current|latest|recent|recently|right now|at the moment|as of|last night|this morning|this afternoon|this evening|this week|this month|this year)\b", re.I),
    re.compile(r"\b(202[5-9]|20[3-9]\d)\b"),
)
DYNAMIC_TOPIC_PATTERNS = (
    re.compile(r"\b(weather|forecast|temperature|rain|snow|air quality|uv index|pollen|sunrise|sunset)\b", re.I),
    re.compile(r"\b(news|headline|headlines|breaking news|current events)\b", re.I),
    re.compile(r"\b(score|scores|standings|schedule|schedules|next game|next match|upcoming game|upcoming match)\b", re.I),
    re.compile(r"\b(traffic|transit delays?|road closures?|flight status|opening hours|hours today)\b", re.I),
    re.compile(r"\b(stock price|share price|market cap|exchange rate|forex|trading at|bitcoin price|ethereum price|btc price|eth price|gold price|oil price)\b", re.I),
    re.compile(r"\bwhen is\b.*\b(next|upcoming)\b.*\b(game|match|flight)\b", re.I),
    re.compile(r"\bwhat time does\b.*\b(open|close)\b", re.I),
    re.compile(r"\bwhen does\b.*\b(open|close)\b", re.I),
)
CURRENT_ROLE_QUERY_PATTERNS = (
    re.compile(r"\b(who is|who's|is|tell me|name|identify)\b.*\b(president|prime minister|ceo|governor|mayor|chancellor|head coach)\b", re.I),
    re.compile(r"\b(current|incumbent)\b.*\b(president|prime minister|ceo|governor|mayor|chancellor|head coach)\b", re.I),
)
HISTORICAL_ROLE_HINT_PATTERNS = (
    re.compile(r"\bwho was\b", re.I),
    re.compile(r"\b(former|previous|historical|history of|back in|during)\b", re.I),
    re.compile(r"\b(1[0-9]{3}|20(?:0[0-9]|1[0-9]|2[0-4]))\b"),
)
NON_LIVE_HINT_PATTERNS = (
    re.compile(r"\b(write|story|poem|script|roleplay|pretend|imagine|fictional|lyrics)\b", re.I),
)


class LLMService:
    def __init__(self, system_prompt: str = "You are a helpful, concise AI assistant."):
        load_dotenv()
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not found in environment variables.")
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if OpenAI is None:
            logger.warning("OpenAI SDK is not installed. Live routing will fall back to Groq.")
        elif not openai_api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables. Live routing will fall back to Groq.")

        self.client = Groq(api_key=api_key)
        self.openai_client = OpenAI(api_key=openai_api_key) if (OpenAI is not None and openai_api_key) else None
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        self.default_model = os.environ.get("GROQ_MODEL_DEFAULT", DEFAULT_CHAT_MODEL).strip() or DEFAULT_CHAT_MODEL
        self.live_model = os.environ.get("OPENAI_MODEL_LIVE", DEFAULT_LIVE_MODEL).strip() or DEFAULT_LIVE_MODEL
        live_routing_flag = os.environ.get("LLM_ENABLE_LIVE_ROUTING")
        if live_routing_flag is None:
            live_routing_flag = os.environ.get("GROQ_ENABLE_LIVE_ROUTING", "1")
        self.live_routing_enabled = live_routing_flag.strip().lower() in LIVE_ROUTING_TRUE
        self._warmed_models: set[str] = set()

    @property
    def stream_warmed(self) -> bool:
        return self.default_model in self._warmed_models

    @staticmethod
    def _matches_any(patterns: tuple[re.Pattern[str], ...], text: str) -> re.Pattern[str] | None:
        for pattern in patterns:
            if pattern.search(text):
                return pattern
        return None

    def _should_use_live_model(self, user_input: str) -> tuple[bool, str]:
        if not self.live_routing_enabled:
            return False, "disabled"

        if self.openai_client is None:
            return False, "openai_unavailable"

        lowered = user_input.strip().lower()
        if not lowered:
            return False, "empty"

        if any(pattern.search(lowered) for pattern in NON_LIVE_HINT_PATTERNS):
            return False, "creative_prompt"

        pattern = self._matches_any(TIME_SENSITIVE_PATTERNS, lowered)
        if pattern is not None:
            return True, f"time_sensitive:{pattern.pattern}"

        pattern = self._matches_any(DYNAMIC_TOPIC_PATTERNS, lowered)
        if pattern is not None:
            return True, f"dynamic_topic:{pattern.pattern}"

        pattern = self._matches_any(CURRENT_ROLE_QUERY_PATTERNS, lowered)
        if pattern is not None:
            if self._matches_any(HISTORICAL_ROLE_HINT_PATTERNS, lowered) is not None:
                return False, "historical_role_query"
            return True, f"current_role:{pattern.pattern}"

        return False, "no_live_signal"

    def _start_groq_stream(self, model: str, messages: List[Dict[str, str]]):
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=0.7,
        )

    def _start_openai_stream(self, model: str, messages: List[Dict[str, str]]):
        if self.openai_client is None:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        return self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )

    @staticmethod
    def _fallback_live_error(error: Exception) -> bool:
        message = str(error).lower()
        fallback_signals = (
            "rate limit",
            "quota",
            "429",
            "temporarily unavailable",
            "overloaded",
            "timeout",
            "timed out",
            "connection",
            "service unavailable",
        )
        return any(signal in message for signal in fallback_signals)

    def warmup(self) -> bool:
        if self.stream_warmed:
            return False

        try:
            stream = self._start_groq_stream(
                self.default_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": "Reply with exactly one short word."},
                ],
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    break
            self._warmed_models.add(self.default_model)
            logger.info("component=llm op=warmup status=ok model=%s", self.default_model)
            return True
        except Exception:
            logger.exception("component=llm op=warmup status=error model=%s", self.default_model)
            raise

    def stream_response(
        self,
        user_input: str,
        min_words: int = 8,
        min_chars: int = 40,
        max_chars: int = 180,
    ) -> Generator[str, None, None]:
        """
        Sends text to the configured LLM provider and yields complete text chunks as they are generated.
        """
        self.history.append({"role": "user", "content": user_input})
        use_live_model, route_reason = self._should_use_live_model(user_input)
        current_model = self.live_model if use_live_model else self.default_model
        current_provider = "openai" if use_live_model else "groq"

        full_response_text = ""
        buffer = ""
        min_chunk_size = max(min_chars, min_words * 2)
        attempted_fallback = False

        logger.info(
            "component=llm op=route status=ok provider=%s model=%s live=%s reason=%s input_chars=%s",
            current_provider,
            current_model,
            use_live_model,
            route_reason,
            len(user_input),
        )

        while True:
            try:
                if current_provider == "openai":
                    stream = self._start_openai_stream(current_model, self.history)
                else:
                    stream = self._start_groq_stream(current_model, self.history)
                print(f"LLM Thinking ({current_provider}:{current_model})...", end="", flush=True)

                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        full_response_text += token
                        buffer += token

                        if any(punct in token for punct in (".", "?", "!", "\n")):
                            if len(buffer) >= min_chunk_size:
                                yield buffer.strip()
                                buffer = ""
                                continue

                        if len(buffer) >= max_chars:
                            last_space = buffer.rfind(" ")
                            if last_space != -1:
                                head = buffer[:last_space].strip()
                                tail = buffer[last_space:].strip()
                                if head:
                                    yield head
                                buffer = tail
                break
            except Exception as e:
                can_fallback = (
                    current_provider == "openai"
                    and not attempted_fallback
                    and not full_response_text
                    and self._fallback_live_error(e)
                )
                if can_fallback:
                    logger.warning(
                        "component=llm op=route_fallback status=retry provider=%s model=%s fallback_provider=groq fallback_model=%s reason=%s input_chars=%s",
                        current_provider,
                        current_model,
                        self.default_model,
                        route_reason,
                        len(user_input),
                    )
                    attempted_fallback = True
                    current_model = self.default_model
                    current_provider = "groq"
                    continue

                logger.exception(
                    "component=llm op=chat status=error provider=%s model=%s input_chars=%s route_reason=%s",
                    current_provider,
                    current_model,
                    len(user_input),
                    route_reason,
                )
                yield f"Error calling language model: {str(e)}"
                return

        if buffer.strip():
            yield buffer.strip()

        self.history.append({"role": "assistant", "content": full_response_text})
        self._warmed_models.add(current_model)
        print(" Done.")
