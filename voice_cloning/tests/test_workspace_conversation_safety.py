"""Regression coverage for anonymous-workspace assistant isolation.

These tests deliberately avoid model providers and GPU engines.  They exercise
the state boundaries that must stay correct when public visitors use the same
backend process.
"""

from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

from src.inference import (
    ProfileRuntimeSettingsRequest,
    _profile_system_prompt,
    _profile_runtime_settings_path,
    update_profile_runtime_settings,
)
from src.llm.llm_service import LEGACY_FAST_MODEL, LLMRoute, LLMService
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
        self.openai_realtime_max_output_tokens = 512
        self.openai_reasoning_effort = "none"


class WorkspaceConversationSafetyTests(unittest.TestCase):
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
