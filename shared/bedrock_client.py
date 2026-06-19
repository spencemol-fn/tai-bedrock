"""
AWS Bedrock Converse API client for the Temporal AI Agent.

Replaces the litellm transport layer with a direct boto3 call to the
Bedrock Converse API. Optionally enforces Bedrock Guardrails when
BEDROCK_GUARDRAIL_ID and BEDROCK_GUARDRAIL_VERSION are both set.
"""
import os
from typing import Any, Dict, Optional

import boto3


class GuardrailInterventionError(Exception):
    """Raised when Bedrock Guardrails blocks the request."""

    def __init__(self, message: str, assessment: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.assessment = assessment or {}


class BedrockLLMClient:
    """
    Thin client wrapping the Bedrock Converse API.

    One instance is created per worker (thread-safe; the boto3 client
    reuses an underlying connection pool).  Configuration is read from
    environment variables at construction time:

    Required:
      BEDROCK_MODEL_ID   – inference-profile or model ARN
                           (default: us.anthropic.claude-haiku-4-5-20251001-v1:0)
      AWS_REGION         – AWS region (default: us-east-1)

    Optional inference config:
      BEDROCK_TEMPERATURE  – float, default 0.0
      BEDROCK_MAX_TOKENS   – int, default 1024

    Optional guardrails (both required to enable):
      BEDROCK_GUARDRAIL_ID
      BEDROCK_GUARDRAIL_VERSION
      BEDROCK_GUARDRAIL_TRACE  – default "enabled"
    """

    def __init__(self) -> None:
        self.model_id: str = os.environ.get(
            "BEDROCK_MODEL_ID",
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        )
        self.region: str = os.environ.get("AWS_REGION", "us-east-1")
        self.temperature: float = float(os.environ.get("BEDROCK_TEMPERATURE", "0.0"))
        self.max_tokens: int = int(os.environ.get("BEDROCK_MAX_TOKENS", "1024"))

        guardrail_id: str = os.environ.get("BEDROCK_GUARDRAIL_ID", "")
        guardrail_version: str = os.environ.get("BEDROCK_GUARDRAIL_VERSION", "")
        self.guardrails_on: bool = bool(guardrail_id and guardrail_version)
        self._guardrail_config: Dict[str, Any] = {}
        if self.guardrails_on:
            self._guardrail_config = {
                "guardrailIdentifier": guardrail_id,
                "guardrailVersion": guardrail_version,
                "trace": os.environ.get("BEDROCK_GUARDRAIL_TRACE", "enabled"),
            }

        self._client = boto3.client("bedrock-runtime", region_name=self.region)

    def converse(self, system: str, prompt: str) -> str:
        """
        Send a single-turn prompt to Bedrock and return the model's text reply.

        Args:
            system: System-level instructions (context + date suffix).
            prompt: The user turn assembled by agent_toolPlanner.

        Returns:
            The raw text content of the first output message block.

        Raises:
            GuardrailInterventionError: When guardrails block the request.
            botocore.exceptions.ClientError: Propagated as-is so the caller
                can classify retryable vs. non-retryable errors.
        """
        if self.guardrails_on:
            content = [{"guardContent": {"text": {"text": prompt}}}]
        else:
            content = [{"text": prompt}]

        kwargs: Dict[str, Any] = {
            "modelId": self.model_id,
            "system": [{"text": system}],
            "messages": [{"role": "user", "content": content}],
            "inferenceConfig": {
                "temperature": self.temperature,
                "maxTokens": self.max_tokens,
            },
        }
        if self.guardrails_on:
            kwargs["guardrailConfig"] = self._guardrail_config

        resp = self._client.converse(**kwargs)

        if resp.get("stopReason") == "guardrail_intervened":
            blocked_text: str = (
                resp.get("output", {})
                .get("message", {})
                .get("content", [{}])[0]
                .get("text", "Your request was blocked by a content policy.")
            )
            assessment: Dict[str, Any] = resp.get("trace", {}).get("guardrail", {})
            raise GuardrailInterventionError(blocked_text, assessment)

        return resp["output"]["message"]["content"][0]["text"]
