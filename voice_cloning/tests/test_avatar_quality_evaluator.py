import unittest

import numpy as np

from src.evaluate_musetalk import _decode_audio_chunk, _numeric_summary, _sustained_silence_mask


class AvatarQualityEvaluatorTests(unittest.TestCase):
    def test_sustained_silence_requires_a_full_run(self):
        envelope = np.array([0.04, 0.001, 0.001, 0.04, 0.001, 0.001, 0.001, 0.001, 0.04])

        result = _sustained_silence_mask(envelope, threshold=0.006, min_frames=4)

        self.assertFalse(result[1])
        self.assertTrue(np.all(result[4:8]))

    def test_pcm_s16le_decoder_respects_channels_and_sample_rate(self):
        stereo = np.array([32767, -32768, 0, 0], dtype="<i2").tobytes()

        audio, sample_rate = _decode_audio_chunk(
            stereo,
            {"audio_format": "pcm_s16le", "sample_rate": 16000, "audio_channels": 2},
        )

        self.assertEqual(sample_rate, 16000)
        np.testing.assert_allclose(audio, [-1.0 / 65536.0, 0.0], atol=1e-7)

    def test_numeric_summary_ignores_missing_values(self):
        summary = _numeric_summary([None, 1.0, float("nan"), 3.0])

        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["median"], 2.0)
        self.assertEqual(summary["p95"], 2.9)


if __name__ == "__main__":
    unittest.main()
