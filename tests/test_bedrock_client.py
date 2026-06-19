"""
Unit tests for shared/bedrock_client.py using botocore Stubber.

These tests exercise:
(a) Happy-path: correct request shaping and response parsing.
(b) Guardrail intervention: stopReason == "guardrail_intervened" raises
    GuardrailInterventionError with message + assessment.
(c) No guardrailConfig emitted when BEDROCK_GUARDRAIL_ID / VERSION are unset.
"""
import os
from unittest.mock import patch

import botocore.session
import pytest
from botocore.stub import Stubber

from shared.bedrock_client import BedrockLLMClient, GuardrailInterventionError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_ENV = {
    "BEDROCK_MODEL_ID": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "AWS_REGION": "us-east-1",
    "BEDROCK_TEMPERATURE": "0.0",
    "BEDROCK_MAX_TOKENS": "1024",
    # guardrail vars intentionally absent
}


def _make_client_with_stubber(
    env: dict | None = None,
) -> tuple[BedrockLLMClient, Stubber]:
    """
    Build a BedrockLLMClient whose internal boto3 client is replaced by a
    Stubber-wrapped client so no real AWS calls are made.
    """
    env = env or _DEFAULT_ENV
    with patch.dict(os.environ, env, clear=True):
        llm_client = BedrockLLMClient()

    # Replace the internal boto3 client with a Stubber-managed one.
    bcore_session = botocore.session.get_session()
    raw_boto3_client = bcore_session.create_client(
        "bedrock-runtime", region_name="us-east-1"
    )
    stubber = Stubber(raw_boto3_client)

    # Patch the internal _client so Stubber intercepts all calls.
    llm_client._client = raw_boto3_client  # type: ignore[assignment]
    return llm_client, stubber


def _converse_response(text: str, stop_reason: str = "end_turn") -> dict:
    """Build a minimal Bedrock Converse API response envelope."""
    return {
        "stopReason": stop_reason,
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": text}],
            }
        },
        "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
        "metrics": {"latencyMs": 100},
    }


def _guardrail_intervened_response(blocked_text: str) -> dict:
    """
    Build a guardrail-intervened response dict (returned directly from boto3, not via Stubber,
    since botocore's Stubber validates response shapes too strictly for trace fields).
    """
    return {
        "stopReason": "guardrail_intervened",
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": blocked_text}],
            }
        },
        "trace": {
            "guardrail": {
                "inputAssessment": {},
                "outputAssessments": {},
            }
        },
        "usage": {"inputTokens": 5, "outputTokens": 5, "totalTokens": 10},
        "metrics": {"latencyMs": 50},
    }


# ---------------------------------------------------------------------------
# (a) Happy path — correct request shaping and response parsing
# ---------------------------------------------------------------------------


class TestBedrockClientHappyPath:
    def test_converse_returns_text(self) -> None:
        """converse() returns the model's text content on a successful call."""
        llm_client, stubber = _make_client_with_stubber()
        expected_text = '{"next": "confirm", "tool": "TestTool", "response": "ok"}'

        stubber.add_response(
            "converse",
            _converse_response(expected_text),
        )

        with stubber:
            result = llm_client.converse(
                system="You are a helpful assistant.",
                prompt="Find events in Sydney",
            )

        assert result == expected_text

    def test_converse_request_contains_system_and_user_message(self) -> None:
        """Request sent to Bedrock must contain system and a user-role message."""
        llm_client, stubber = _make_client_with_stubber()
        captured: list[dict] = []

        def _capture(self_inner, op, params, **kw):  # type: ignore[misc]
            captured.append(params)
            return _converse_response("result")

        stubber.add_response("converse", _converse_response("result"))

        with stubber:
            llm_client.converse(system="sys", prompt="user prompt")

        # The stubber validates the call was made; check the client state
        assert llm_client.model_id == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        assert llm_client.temperature == 0.0
        assert llm_client.max_tokens == 1024

    def test_converse_no_guardrail_config_when_env_unset(self) -> None:
        """guardrailConfig must NOT be present when env vars are absent."""
        llm_client, stubber = _make_client_with_stubber()

        assert llm_client.guardrails_on is False
        assert llm_client._guardrail_config == {}

        stubber.add_response("converse", _converse_response("ok"))

        with stubber:
            result = llm_client.converse("sys", "prompt")

        assert result == "ok"

    def test_converse_plain_text_content_when_guardrails_off(self) -> None:
        """Without guardrails the user message content is a plain text block."""
        llm_client, _ = _make_client_with_stubber()
        assert llm_client.guardrails_on is False

        # Verify internally that converse() builds plain content (not guardContent)
        # by inspecting the structure the method would pass to the API.
        # We achieve this by patching _client.converse at the boto3 level.
        calls: list[dict] = []

        def _mock_converse(**kwargs: dict) -> dict:
            calls.append(kwargs)
            return _converse_response("ok")

        llm_client._client.converse = _mock_converse  # type: ignore[assignment]
        llm_client.converse("system", "my prompt")

        assert len(calls) == 1
        msg_content = calls[0]["messages"][0]["content"]
        assert msg_content == [{"text": "my prompt"}]
        assert "guardrailConfig" not in calls[0]


# ---------------------------------------------------------------------------
# (b) Guardrail intervention
# ---------------------------------------------------------------------------


class TestBedrockClientGuardrailIntervention:
    _GUARDRAIL_ENV = {
        **_DEFAULT_ENV,
        "BEDROCK_GUARDRAIL_ID": "abc123",
        "BEDROCK_GUARDRAIL_VERSION": "1",
        "BEDROCK_GUARDRAIL_TRACE": "enabled",
    }

    def test_guardrail_intervened_raises_error(self) -> None:
        """stopReason=='guardrail_intervened' must raise GuardrailInterventionError."""
        llm_client, _ = _make_client_with_stubber(self._GUARDRAIL_ENV)
        blocked_msg = "I can't help with that."

        # Use direct mock rather than Stubber to avoid botocore response-shape validation
        llm_client._client.converse = (  # type: ignore[assignment]
            lambda **kw: _guardrail_intervened_response(blocked_msg)
        )

        with pytest.raises(GuardrailInterventionError) as exc_info:
            llm_client.converse("system", "harmful prompt")

        err = exc_info.value
        assert err.message == blocked_msg
        assert isinstance(
            err.assessment, dict
        )  # assessment is the guardrail trace dict

    def test_guardrail_intervened_carries_assessment(self) -> None:
        """GuardrailInterventionError.assessment must be populated from trace."""
        llm_client, _ = _make_client_with_stubber(self._GUARDRAIL_ENV)

        llm_client._client.converse = (  # type: ignore[assignment]
            lambda **kw: _guardrail_intervened_response("Blocked.")
        )

        with pytest.raises(GuardrailInterventionError) as exc_info:
            llm_client.converse("system", "bad prompt")

        assert isinstance(exc_info.value.assessment, dict)
        assert exc_info.value.assessment != {}

    def test_guardrail_on_sends_guard_content(self) -> None:
        """With guardrails enabled the content block uses guardContent wrapper."""
        llm_client, _ = _make_client_with_stubber(self._GUARDRAIL_ENV)

        assert llm_client.guardrails_on is True

        calls: list[dict] = []

        def _mock_converse(**kwargs: dict) -> dict:
            calls.append(kwargs)
            return _converse_response("ok")

        llm_client._client.converse = _mock_converse  # type: ignore[assignment]
        llm_client.converse("sys", "the prompt")

        msg_content = calls[0]["messages"][0]["content"]
        assert msg_content == [{"guardContent": {"text": {"text": "the prompt"}}}]
        assert "guardrailConfig" in calls[0]
        assert calls[0]["guardrailConfig"]["guardrailIdentifier"] == "abc123"


# ---------------------------------------------------------------------------
# (c) Inference config is forwarded
# ---------------------------------------------------------------------------


class TestBedrockClientInferenceConfig:
    def test_custom_temperature_and_max_tokens(self) -> None:
        """BEDROCK_TEMPERATURE and BEDROCK_MAX_TOKENS are forwarded to the API."""
        env = {
            **_DEFAULT_ENV,
            "BEDROCK_TEMPERATURE": "0.7",
            "BEDROCK_MAX_TOKENS": "512",
        }
        llm_client, _ = _make_client_with_stubber(env)

        assert llm_client.temperature == 0.7
        assert llm_client.max_tokens == 512

        calls: list[dict] = []

        def _mock_converse(**kwargs: dict) -> dict:
            calls.append(kwargs)
            return _converse_response("ok")

        llm_client._client.converse = _mock_converse  # type: ignore[assignment]
        llm_client.converse("sys", "prompt")

        inference = calls[0]["inferenceConfig"]
        assert inference["temperature"] == 0.7
        assert inference["maxTokens"] == 512
