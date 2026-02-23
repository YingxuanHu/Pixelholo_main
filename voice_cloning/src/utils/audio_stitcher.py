import numpy as np


class AudioStitcher:
    def __init__(self, sample_rate: int = 24000, fade_len_ms: float = 15.0) -> None:
        self.fade_len = int(sample_rate * (fade_len_ms / 1000.0))
        self.prev_tail: np.ndarray | None = None

    def process(self, chunk: np.ndarray) -> np.ndarray:
        if chunk.size == 0:
            return chunk
        chunk = chunk.astype(np.float32, copy=False)
        if self.fade_len <= 0:
            return chunk

        if self.prev_tail is None:
            # First chunk: hold back a tail for the next crossfade.
            fade_len = min(self.fade_len, chunk.size)
            if fade_len <= 1:
                return chunk
            output = chunk[:-fade_len]
            self.prev_tail = chunk[-fade_len:].copy()
            return output

        fade_len = min(self.fade_len, chunk.size, self.prev_tail.size)
        if fade_len <= 1:
            self.prev_tail = chunk[-1:].copy()
            return chunk

        new_head = chunk[:fade_len]
        fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        crossfaded = (self.prev_tail * fade_out) + (new_head * fade_in)
        if chunk.size <= fade_len:
            self.prev_tail = chunk.copy()
            return crossfaded

        tail_len = min(self.fade_len, chunk.size)
        body = chunk[fade_len:-tail_len] if chunk.size > fade_len + tail_len else np.array([], dtype=np.float32)
        self.prev_tail = chunk[-tail_len:].copy()
        if body.size == 0:
            return crossfaded
        return np.concatenate((crossfaded, body))

    def flush(self) -> np.ndarray:
        if self.prev_tail is None:
            return np.array([], dtype=np.float32)
        tail = self.prev_tail
        self.prev_tail = None
        return tail
