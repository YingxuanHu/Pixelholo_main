"""Regression coverage for anonymous-workspace assistant isolation.

These tests deliberately avoid model providers and GPU engines.  They exercise
the state boundaries that must stay correct when public visitors use the same
backend process.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.inference import (
    ProfileRuntimeSettingsRequest,
    _plan_stream_tts_chunks,
    _profile_system_prompt,
    _profile_runtime_settings_path,
    _gpu_keepalive_interval_seconds,
    update_profile_runtime_settings,
)
from src.llm.llm_service import LEGACY_FAST_MODEL, LLMRoute, LLMService
from src.musetalk_bridge import MuseTalkBridge
from config import PROFILE_TYPE_AVATAR, reset_workspace_id, set_workspace_id, workspace_data_root


class _NoProviderLLM(LLMService):
    """A provider-free LLM that preserves the real conversation code path."""

    def __init__(self):
        # Do not construct external API clients for a state-boundary test.
        self.system_prompt = "default prompt"
        self._conversations = {}
        import threading
        from contextvars import ContextVar

        self._conversations_lock = threading.Lock()
        self._conversation_context = ContextVar("test_pixelholo_conversation", default=None)
        self._warmed_routes = set()
        self._realtime_rate_limit_until = 0.0
        self.auto_live_followup_turns = 0
        self.auto_live_topic_turns = 0
        self.auto_live_followup_ttl_seconds = 60
        self.realtime_history_messages = 2
        self.max_message_chars = 1200
        self.max_history_chars = 6000
        self.realtime_max_message_chars = 800
        self.realtime_max_history_chars = 2600

    def resolve_route(self, **_kwargs):
        return LLMRoute("legacy_fast", "legacy_fast", LEGACY_FAST_MODEL)

    def _candidate_routes(self, route, _user_input):
        return [route]

    def _stream_from_route(self, _route, **_kwargs):
        def _stream():
            yield "A private response."
            return "A private response."

        return _stream()


class _OpenAIResponseRecorder:
    """Captures a Responses request without contacting OpenAI."""

    def __init__(self):
        self.params = None

    def create(self, **params):
        self.params = params
        return iter([SimpleNamespace(type="response.output_text.delta", delta="Hello there.")])


class _OpenAIResponseLLM(LLMService):
    """Minimal service instance for exercising the OpenAI request contract."""

    def __init__(self, recorder: _OpenAIResponseRecorder):
        self.openai_client = SimpleNamespace(responses=recorder)
        self.system_prompt = "Reply concisely."
        self.openai_realtime_max_output_tokens = 512
        self.openai_reasoning_effort = "none"
        self._warmed_routes = set()

    def _warmup_routes(self, _mode, _model):
        return [LLMRoute("live_search", "live_search", "gpt-4o-mini")]


class WorkspaceConversationSafetyTests(unittest.TestCase):
    def test_musetalk_keeps_visemes_enabled_in_the_default_stream_path(self):
        # The public request path does not pass an override.  It must not
        # freeze the generated mouth for quiet samples because low-energy
        # phonemes and word boundaries are still speech.
        bridge = object.__new__(MuseTalkBridge)
        bridge.suppress_silence_motion = False

        self.assertFalse(bridge._silence_hold_enabled(None))
        self.assertTrue(bridge._silence_hold_enabled(True))
        self.assertFalse(bridge._silence_hold_enabled(False))

    def test_musetalk_holds_only_sustained_silence(self):
        fps = 25
        audio = np.full(16000, 0.08, dtype=np.float32)
        # A 440 ms quiet region is a real conversational pause and should not
        # be rendered as a moving mouth.  The threshold must be long enough
        # that quiet consonants and ordinary word boundaries keep their
        # model-generated visemes.
        audio[int(0.28 * 16000):int(0.72 * 16000)] = 0.0
        mask = MuseTalkBridge._stable_silence_mask(
            audio,
            frame_count=25,
            fps=fps,
            rms_threshold=0.004,
            min_duration_ms=280,
        )
        self.assertTrue(mask[7:18].all())
        self.assertFalse(mask[:6].any())
        self.assertFalse(mask[19:].any())

        # A 240 ms gap can occur between normally spoken words.  It must
        # preserve MuseTalk's generated motion around that boundary.
        short_gap = np.full(16000, 0.08, dtype=np.float32)
        short_gap[int(0.32 * 16000):int(0.56 * 16000)] = 0.0
        short_mask = MuseTalkBridge._stable_silence_mask(
            short_gap,
            frame_count=25,
            fps=fps,
            rms_threshold=0.004,
            min_duration_ms=280,
        )
        self.assertFalse(short_mask.any())

    def test_musetalk_preserves_the_full_mouth_and_source_chin(self):
        # This is intentionally model-free. The complete generated mouth must
        # survive compositing. Cutting the mask near 0.65 leaves the source
        # lower lip in the output, which visibly follows the reference video
        # instead of the generated speech. Only the bottom of the chin should
        # fade back to the source frame.
        bridge = object.__new__(MuseTalkBridge)
        bridge.mouth_mask_bottom_ratio = 0.84
        bridge.mouth_mask_bottom_feather = 0.05
        bridge.chin_mask_bottom_ratio = 0.66
        bridge.mouth_core_center_y_ratio = 0.68
        bridge.mouth_core_half_width_ratio = 0.30
        bridge.mouth_core_half_height_ratio = 0.16
        bridge.mouth_core_feather = 0.25
        alpha = np.ones((100, 80), dtype=np.float32)

        protected = bridge._restrict_chin_blend(
            alpha,
            face_box=(10, 20, 70, 80),
            crop_box=(0, 0, 80, 100),
        )

        # The center of the generated mouth remains opaque through the lower
        # lip, while the same row at the side has already returned to source.
        self.assertEqual(float(protected[61, 40]), 1.0)
        self.assertGreater(float(protected[67, 40]), 0.0)
        self.assertEqual(float(protected[67, 8]), 0.0)
        # The bottom of the chin is always the original portrait.
        self.assertEqual(float(protected[76, 40]), 0.0)

    def test_musetalk_resizes_the_generated_face_only_once(self):
        # Multiple resize passes soften the already 256px MuseTalk output.
        # The compositor must calculate the final placement first, then resize
        # the generated face directly to that size.
        bridge = object.__new__(MuseTalkBridge)
        bridge.face_scale = 0.96
        bridge.detail_sharpen = 0.0
        bridge.color_match_strength = 0.0
        bridge.temporal_smooth = 0.0
        bridge._prev_generated_face = None
        bridge.mouth_mask_bottom_ratio = 0.84
        bridge.mouth_mask_bottom_feather = 0.05
        bridge.chin_mask_bottom_ratio = 0.66
        bridge.mouth_core_center_y_ratio = 0.68
        bridge.mouth_core_half_width_ratio = 0.30
        bridge.mouth_core_half_height_ratio = 0.16
        bridge.mouth_core_feather = 0.25

        frame = np.zeros((100, 80, 3), dtype=np.uint8)
        generated = np.full((256, 256, 3), 127, dtype=np.uint8)
        alpha = np.ones((100, 80), dtype=np.float32)
        with patch("src.musetalk_bridge.cv2.resize", wraps=__import__("cv2").resize) as resize:
            bridge._blend_frame(
                frame,
                generated,
                face_box=[10, 20, 70, 80],
                crop_box=[0, 0, 80, 100],
                alpha=alpha,
            )

        self.assertEqual(resize.call_count, 1)

    def test_musetalk_holds_the_last_rendered_portrait_during_a_pause(self):
        # The original reference clip keeps moving.  A sustained TTS pause
        # must not switch back to that clip, otherwise its mouth and chin can
        # visibly jump as the generated stream becomes quiet.
        bridge = object.__new__(MuseTalkBridge)
        source = np.zeros((3, 4, 3), dtype=np.uint8)
        rendered = np.full((3, 4, 3), 127, dtype=np.uint8)
        bridge._last_rendered_frame = rendered

        held = bridge._pause_frame(source)
        self.assertTrue(np.array_equal(held, rendered))
        self.assertIsNot(held, rendered)

        bridge._last_rendered_frame = None
        self.assertTrue(np.array_equal(bridge._pause_frame(source), source))

    def test_musetalk_uses_a_stable_face_track_for_runtime_compositing(self):
        # A face detector can vary by a few pixels between adjacent source
        # frames. The generated image must follow a stable track or its chin
        # and jaw visibly resize even when the source subject is still.
        raw = np.array(
            [
                [20, 15, 80, 95],
                [24, 13, 84, 98],
                [15, 17, 75, 92],
                [23, 14, 83, 96],
                [17, 16, 77, 94],
            ],
            dtype=np.int32,
        )
        stable = MuseTalkBridge._smooth_xyxy_boxes(raw, width=120, height=140, window=5)
        raw_center_x = (raw[:, 0] + raw[:, 2]) / 2.0
        stable_center_x = (stable[:, 0] + stable[:, 2]) / 2.0
        self.assertLess(float(np.ptp(stable_center_x)), float(np.ptp(raw_center_x)))

    def test_musetalk_request_coordinate_source_overrides_the_preset(self):
        # Production keeps MuseTalk's full-face conditioning. Keep an explicit
        # baked-track override for controlled A/B evaluations without allowing
        # a previous request's mutable renderer state to leak into the next one.
        bridge = object.__new__(MuseTalkBridge)
        bridge.default_temporal_smooth = 0.025
        bridge.default_face_scale = 0.96
        bridge.default_detail_sharpen = 0.70
        bridge.default_mouth_mask_bottom_ratio = 0.84
        bridge.default_chin_mask_bottom_ratio = 0.66
        bridge.default_mouth_core_center_y_ratio = 0.68
        bridge.default_mouth_core_half_width_ratio = 0.30
        bridge.default_mouth_core_half_height_ratio = 0.16
        bridge.default_mouth_core_feather = 0.25
        bridge.default_infer_fps = 25.0
        bridge.default_audio_history_sec = 2.0

        with patch.dict("os.environ", {"MUSE_TALK_COORD_SOURCE": "legacy"}):
            bridge.configure_for_request(preset="realistic")
            self.assertEqual(bridge.coord_source, "legacy")

            bridge.configure_for_request(preset="realistic", coord_source="baked")
            self.assertEqual(bridge.coord_source, "baked")

            bridge.configure_for_request(preset="realistic", mouth_mask_bottom_ratio=0.72)
            self.assertEqual(bridge.mouth_mask_bottom_ratio, 0.72)

            # A profile-specific calibration is mutable renderer state. A
            # different profile must never inherit its mouth/chin boundary.
            bridge.chin_mask_bottom_ratio = 0.70
            bridge.mouth_core_center_y_ratio = 0.74
            bridge.mouth_core_half_width_ratio = 0.25
            bridge.mouth_core_half_height_ratio = 0.13
            bridge.mouth_core_feather = 0.16
            bridge._profile_calibration = {"landmark_samples": 24}
            bridge.configure_for_request(preset="realistic")
            self.assertEqual(bridge.mouth_mask_bottom_ratio, 0.84)
            self.assertEqual(bridge.chin_mask_bottom_ratio, 0.66)
            self.assertEqual(bridge.mouth_core_center_y_ratio, 0.68)
            self.assertEqual(bridge.mouth_core_half_width_ratio, 0.30)
            self.assertEqual(bridge.mouth_core_half_height_ratio, 0.16)
            self.assertEqual(bridge.mouth_core_feather, 0.25)
            self.assertIsNone(bridge._profile_calibration)

    def test_musetalk_applies_profile_calibrated_mouth_geometry(self):
        bridge = object.__new__(MuseTalkBridge)
        bridge._request_mouth_mask_bottom_ratio = None
        bridge._profile_calibration = None
        bridge.mouth_core_center_y_ratio = 0.68
        bridge.mouth_core_half_width_ratio = 0.30
        bridge.mouth_core_half_height_ratio = 0.16
        bridge.mouth_core_feather = 0.25
        bridge.chin_mask_bottom_ratio = 0.66
        bridge.mouth_mask_bottom_ratio = 0.84

        with tempfile.TemporaryDirectory() as tmp:
            calibration_path = Path(tmp) / "musetalk_profile_calibration.json"
            calibration_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "landmark_samples": 24,
                        "mouth_core_center_y_ratio": 0.71,
                        "mouth_core_half_width_ratio": 0.27,
                        "mouth_core_half_height_ratio": 0.15,
                        "mouth_core_feather": 0.20,
                        "mouth_mask_bottom_ratio": 0.86,
                        "chin_mask_bottom_ratio": 0.69,
                    }
                )
            )
            bridge._apply_profile_calibration(Path(tmp))

        self.assertEqual(bridge.mouth_core_center_y_ratio, 0.71)
        self.assertEqual(bridge.mouth_core_half_width_ratio, 0.27)
        self.assertEqual(bridge.mouth_core_half_height_ratio, 0.15)
        self.assertEqual(bridge.mouth_core_feather, 0.20)
        self.assertEqual(bridge.mouth_mask_bottom_ratio, 0.86)
        self.assertEqual(bridge.chin_mask_bottom_ratio, 0.69)

    def test_musetalk_request_override_beats_profile_calibration(self):
        bridge = object.__new__(MuseTalkBridge)
        bridge._request_mouth_mask_bottom_ratio = 0.82
        bridge._profile_calibration = None
        bridge.mouth_core_center_y_ratio = 0.68
        bridge.mouth_core_half_width_ratio = 0.30
        bridge.mouth_core_half_height_ratio = 0.16
        bridge.mouth_core_feather = 0.25
        bridge.chin_mask_bottom_ratio = 0.66
        bridge.mouth_mask_bottom_ratio = 0.82

        with tempfile.TemporaryDirectory() as tmp:
            calibration_path = Path(tmp) / "musetalk_profile_calibration.json"
            calibration_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "landmark_samples": 24,
                        "mouth_core_center_y_ratio": 0.71,
                        "mouth_core_half_width_ratio": 0.27,
                        "mouth_core_half_height_ratio": 0.15,
                        "mouth_core_feather": 0.20,
                        "mouth_mask_bottom_ratio": 0.86,
                        "chin_mask_bottom_ratio": 0.69,
                    }
                )
            )
            bridge._apply_profile_calibration(Path(tmp))

        self.assertEqual(bridge.mouth_mask_bottom_ratio, 0.82)

    def test_musetalk_pads_only_the_invisible_tail_batch(self):
        # The final short model batch used to create a new CUDA execution
        # shape, which can stall a warm stream for seconds. Padding repeats the
        # final sample for execution only, while the logical output count stays
        # unchanged.
        bridge = object.__new__(MuseTalkBridge)
        bridge.static_batch_padding = True
        bridge.batch_size = 24
        whisper = torch.arange(6 * 2 * 3, dtype=torch.float32).reshape(6, 2, 3)
        latent = torch.arange(6 * 2 * 2 * 2, dtype=torch.float32).reshape(6, 2, 2, 2)

        padded_whisper, padded_latent, logical_size = bridge._pad_runtime_batch(whisper, latent)

        self.assertEqual(logical_size, 6)
        self.assertEqual(tuple(padded_whisper.shape), (24, 2, 3))
        self.assertEqual(tuple(padded_latent.shape), (24, 2, 2, 2))
        self.assertTrue(torch.equal(padded_whisper[:6], whisper))
        self.assertTrue(torch.equal(padded_latent[:6], latent))
        self.assertTrue(torch.equal(padded_whisper[6:], whisper[-1:].expand(18, -1, -1)))

    def test_gpu_keepalive_interval_is_configurable_and_bounded(self):
        with patch.dict("os.environ", {"PIXELHOLO_GPU_KEEPALIVE_INTERVAL_SEC": "0"}):
            self.assertEqual(_gpu_keepalive_interval_seconds(), 0.0)
        with patch.dict("os.environ", {"PIXELHOLO_GPU_KEEPALIVE_INTERVAL_SEC": "999"}):
            self.assertEqual(_gpu_keepalive_interval_seconds(), 10.0)
        with patch.dict("os.environ", {"PIXELHOLO_GPU_KEEPALIVE_INTERVAL_SEC": "invalid"}):
            self.assertEqual(_gpu_keepalive_interval_seconds(), 0.5)

    def test_first_stream_chunk_keeps_a_playback_cushion(self):
        chunks = _plan_stream_tts_chunks(
            "A short first sentence. A second phrase gives the player enough audio to stay smooth.",
            max_chars=120,
            max_words=48,
            first_chunk_max_chars=72,
            is_first_global_chunk=True,
        )
        self.assertGreaterEqual(len(chunks[0]), 64)
        self.assertLessEqual(len(chunks[0]), 120)

    def test_live_gpt_keeps_search_available_without_forcing_it(self):
        recorder = _OpenAIResponseRecorder()
        llm = _OpenAIResponseLLM(recorder)

        response = list(
            llm._stream_openai_responses_tokens(
                [
                    {"role": "system", "content": "Reply concisely."},
                    {"role": "user", "content": "Say hello."},
                ],
                "gpt-4o-mini",
            )
        )

        self.assertEqual(response, ["Hello there."])
        self.assertIsNotNone(recorder.params)
        self.assertEqual(recorder.params["tool_choice"], "auto")
        self.assertEqual(recorder.params["tools"], [{"type": "web_search", "search_context_size": "low"}])

    def test_live_gpt_warmup_opens_a_real_provider_stream(self):
        recorder = _OpenAIResponseRecorder()
        llm = _OpenAIResponseLLM(recorder)

        self.assertTrue(llm.warmup(mode="live_search"))
        self.assertEqual(recorder.params["model"], "gpt-4o-mini")
        self.assertEqual(recorder.params["tool_choice"], "auto")
        self.assertTrue(llm.is_warmed(mode="live_search"))

    def test_conversation_history_and_persona_are_keyed_per_profile(self):
        llm = _NoProviderLLM()
        list(llm.stream_response("first visitor", conversation_key="workspace-a:avatar:alvin", system_prompt="You are Alvin."))
        list(llm.stream_response("second visitor", conversation_key="workspace-b:avatar:hannah", system_prompt="You are Hannah."))

        alvin = llm._conversations["workspace-a:avatar:alvin"]
        hannah = llm._conversations["workspace-b:avatar:hannah"]
        self.assertEqual(alvin.history[0]["content"], "You are Alvin.")
        self.assertEqual(hannah.history[0]["content"], "You are Hannah.")
        self.assertIn("first visitor", alvin.history[1]["content"])
        self.assertNotIn("second visitor", " ".join(item["content"] for item in alvin.history))
        self.assertNotIn("first visitor", " ".join(item["content"] for item in hannah.history))

    def test_saved_profile_prompt_is_private_and_request_override_is_ephemeral(self):
        workspace = str(uuid.uuid4())
        token = set_workspace_id(workspace)
        try:
            profile_dir = workspace_data_root() / "avatar_profiles" / "ava"
            profile_dir.mkdir(parents=True)
            update_profile_runtime_settings(
                "ava",
                ProfileRuntimeSettingsRequest(
                    profile_type=PROFILE_TYPE_AVATAR,
                    system_prompt="You are Ava, a patient science teacher.",
                ),
            )
            self.assertTrue(_profile_runtime_settings_path("ava", PROFILE_TYPE_AVATAR).exists())
            saved_prompt = _profile_system_prompt("ava", PROFILE_TYPE_AVATAR, None)
            preview_prompt = _profile_system_prompt("ava", PROFILE_TYPE_AVATAR, "You are Ava, a comedian.")
            self.assertIn("patient science teacher", saved_prompt)
            self.assertIn("comedian", preview_prompt)
            self.assertNotIn("patient science teacher", preview_prompt)
        finally:
            reset_workspace_id(token)
            shutil.rmtree(workspace_data_root(workspace), ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
