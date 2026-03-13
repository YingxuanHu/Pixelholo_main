import logging
import os
from typing import Generator, List, Dict

from dotenv import load_dotenv
from groq import Groq


logger = logging.getLogger("pixelholo.llm")

class LLMService:
    def __init__(self, system_prompt: str = "You are a helpful, concise AI assistant."):
        load_dotenv()
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("WARNING: GROQ_API_KEY not found in environment variables.")

        self.client = Groq(api_key=api_key)
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        self._stream_warmed = False

    @property
    def stream_warmed(self) -> bool:
        return self._stream_warmed

    def warmup(self) -> bool:
        if self._stream_warmed:
            return False

        try:
            stream = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": "Reply with exactly one short word."},
                ],
                stream=True,
                temperature=0.7,
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    break
            self._stream_warmed = True
            logger.info("component=llm op=warmup status=ok")
            return True
        except Exception:
            logger.exception("component=llm op=warmup status=error")
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

        try:
            stream = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=self.history,
                stream=True,
                temperature=0.7,
            )
        except Exception as e:
            logger.exception(
                "component=llm op=groq_chat status=error input_chars=%s",
                len(user_input),
            )
            yield f"Error calling Groq: {str(e)}"
            return

        full_response_text = ""
        buffer = ""
        min_chunk_size = max(min_chars, min_words * 2)

        print("Llama 3 Thinking...", end="", flush=True)

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
        print(" Done.")
