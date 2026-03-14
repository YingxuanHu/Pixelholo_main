import logging
import os
import re
from typing import Generator, List, Dict

from dotenv import load_dotenv
from groq import Groq


logger = logging.getLogger("pixelholo.llm")

DEFAULT_CHAT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DEFAULT_LIVE_MODEL = "groq/compound-mini"
LIVE_ROUTING_TRUE = {"1", "true", "yes", "on"}
LIVE_SIGNAL_PATTERNS = (
    re.compile(r"\b(today|tonight|tomorrow|yesterday|currently|current|latest|recent|recently|right now|at the moment|as of)\b", re.I),
    re.compile(r"\b(weather|forecast|temperature|rain|snow|news|headline|price|stock|market|score|scores|standings|schedule|schedules|traffic)\b", re.I),
    re.compile(r"\b(who is|who won|what happened|what's happening|what is happening|what time|when is|when did)\b", re.I),
    re.compile(r"\b(202[5-9]|20[3-9]\d)\b"),
)
NON_LIVE_HINT_PATTERNS = (
    re.compile(r"\b(write|story|poem|script|roleplay|pretend|imagine|fictional|lyrics)\b", re.I),
)

class LLMService:
    def __init__(self, system_prompt: str = "You are a helpful, concise AI assistant."):
        load_dotenv()
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("WARNING: GROQ_API_KEY not found in environment variables.")

        self.client = Groq(api_key=api_key)
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        self.default_model = os.environ.get("GROQ_MODEL_DEFAULT", DEFAULT_CHAT_MODEL).strip() or DEFAULT_CHAT_MODEL
        self.live_model = os.environ.get("GROQ_MODEL_LIVE", DEFAULT_LIVE_MODEL).strip() or DEFAULT_LIVE_MODEL
        self.live_routing_enabled = os.environ.get("GROQ_ENABLE_LIVE_ROUTING", "1").strip().lower() in LIVE_ROUTING_TRUE
        self._warmed_models: set[str] = set()

    @property
    def stream_warmed(self) -> bool:
        return self.default_model in self._warmed_models

    def _should_use_live_model(self, user_input: str) -> tuple[bool, str]:
        if not self.live_routing_enabled:
            return False, "disabled"

        if self.live_model == self.default_model:
            return False, "same_model"

        lowered = user_input.strip().lower()
        if not lowered:
            return False, "empty"

        if any(pattern.search(lowered) for pattern in NON_LIVE_HINT_PATTERNS):
            return False, "creative_prompt"

        for pattern in LIVE_SIGNAL_PATTERNS:
            if pattern.search(lowered):
                return True, f"matched:{pattern.pattern}"

        return False, "no_live_signal"

    def _start_stream(self, model: str, messages: List[Dict[str, str]]):
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=0.7,
        )

    def warmup(self) -> bool:
        if self.stream_warmed:
            return False

        try:
            stream = self._start_stream(
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
            logger.info(
                "component=llm op=warmup status=ok model=%s",
                self.default_model,
            )
            return True
        except Exception:
            logger.exception(
                "component=llm op=warmup status=error model=%s",
                self.default_model,
            )
            raise

    def stream_response(
        self,
        user_input: str,
        min_words: int = 8,
        min_chars: int = 40,
        max_chars: int = 180,
    ) -> Generator[str, None, None]:
        """
        Sends text to Groq (Llama 3) and yields COMPLETE SENTENCES as they are generated.
        """
        self.history.append({"role": "user", "content": user_input})
        use_live_model, route_reason = self._should_use_live_model(user_input)
        model = self.live_model if use_live_model else self.default_model

        try:
            stream = self._start_stream(model, self.history)
        except Exception as e:
            logger.exception(
                "component=llm op=groq_chat status=error model=%s input_chars=%s route_reason=%s",
                model,
                len(user_input),
                route_reason,
            )
            yield f"Error calling Groq: {str(e)}"
            return

        full_response_text = ""
        buffer = ""
        min_chunk_size = max(min_chars, min_words * 2)

        logger.info(
            "component=llm op=route status=ok model=%s live=%s reason=%s input_chars=%s",
            model,
            use_live_model,
            route_reason,
            len(user_input),
        )
        print(f"LLM Thinking ({model})...", end="", flush=True)

        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
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

        self.history.append({"role": "assistant", "content": full_response_text})
        self._warmed_models.add(model)
        print(" Done.")
