# Swap litellm → AWS Bedrock Converse API (with Guardrails)

## Context

The agent currently calls LLMs through `litellm.completion()`. We are replacing that
with a direct **AWS Bedrock Converse API** client, standardizing on the **US cross-region
inference profile for Claude Haiku 4.5** (`us.anthropic.claude-haiku-4-5-20251001-v1:0`),
and adding **Bedrock Guardrails** enforcement on LLM input/output.

The work must **preserve existing behavior and the Temporal multi-turn-chat-via-signals
pattern** (per the Temporal AI reference architecture and `/temporal-developer` conventions
for signals/queries/activities). Crucially, the LLM is invoked from a **single chokepoint**:
`ToolActivities.agent_toolPlanner` (`activities/tool_activities.py:137`). `agent_validatePrompt`
and the continue-as-new summary both route *through* `agent_toolPlanner`, and the agent uses a
**prompt-based JSON tool-selection** pattern (model returns `{response, next, tool, args}` text
that we parse) — **not** native function calling. Therefore the swap is confined to the activity,
a new client module, config, and tests. **No workflow, signal, query, prompt-generator, or
tool-dispatch code changes** — the Temporal pattern is untouched.

All work happens on an isolated **git worktree** so the main checkout stays clean while real AWS
creds are dropped into `.env` for smoke testing. Emulate Bedrock via mocks first; live smoke test last.

## Decisions (aligned via grill)

- **Preserve** the prompt-based JSON pattern; swap transport only (no `toolConfig`).
- Extract a thin **`BedrockLLMClient`** (`shared/bedrock_client.py`) exposing `converse(system, prompt) -> str`.
- Config: **`BEDROCK_MODEL_ID`** (default `us.anthropic.claude-haiku-4-5-20251001-v1:0`),
  **`AWS_REGION`** (default `us-east-1`), standard boto3 cred chain. **Drop** `LLM_MODEL`/`LLM_KEY`/`LLM_BASE_URL`.
- Call boto3 (sync) via **`asyncio.to_thread`** from the async activity (fixes latent event-loop blocking).
- `inferenceConfig`: **`temperature=0.0`, `maxTokens=1024`**, overridable via `BEDROCK_TEMPERATURE` / `BEDROCK_MAX_TOKENS`.
- **Guardrails optional**, attached only when `BEDROCK_GUARDRAIL_ID` **and** `BEDROCK_GUARDRAIL_VERSION` are set
  (+ `BEDROCK_GUARDRAIL_TRACE`, default `enabled`). User message wrapped in a **`guardContent`** block when on.
- **Guardrail intervention** (`stopReason == "guardrail_intervened"`): client raises typed
  `GuardrailInterventionError`; `agent_toolPlanner` catches it and returns a graceful planner dict
  `{"next":"question","response":<blocked msg>,"tool":null,"args":{}}` (no retry, conversation continues).
- **Error classification**: `AccessDeniedException`/`ValidationException`/`ResourceNotFoundException`
  → `ApplicationError(non_retryable=True)` (fail fast); throttling/timeout/5xx → propagate (Temporal retries).
- **Full litellm removal** from `pyproject.toml`; add `boto3`. Update `.env.example`, docs, README.
- Tests: re-target existing mocks to `BedrockLLMClient.converse`; add a **botocore `Stubber`** unit test
  for the client (request shaping + response parsing + guardrail intervention).

## Setup

```bash
git worktree add ../tai-bedrock -b bedrock-converse-migration
cd ../tai-bedrock && uv sync
```

## Changes

### 1. NEW `shared/bedrock_client.py`
A self-contained client + exceptions. Reads config from env in `__init__`, creates one
`boto3.client("bedrock-runtime", region_name=...)` (thread-safe; reused across calls).

- `class GuardrailInterventionError(Exception)`: carries `message` (blocked text) + `assessment` (trace dict).
- `class BedrockLLMClient`:
  - `__init__(self)`: read `BEDROCK_MODEL_ID`, `AWS_REGION`, `BEDROCK_TEMPERATURE`, `BEDROCK_MAX_TOKENS`,
    `BEDROCK_GUARDRAIL_ID`, `BEDROCK_GUARDRAIL_VERSION`, `BEDROCK_GUARDRAIL_TRACE`. Set
    `self.guardrails_on = bool(guardrail_id and guardrail_version)`. Build the boto3 client.
  - `converse(self, system: str, prompt: str) -> str`:
    - `content = [{"guardContent": {"text": {"text": prompt}}}]` if `guardrails_on` else `[{"text": prompt}]`
    - `kwargs = {modelId, system=[{"text": system}], messages=[{"role":"user","content": content}],
       inferenceConfig={"temperature": temp, "maxTokens": max_tokens}}`
    - if `guardrails_on`: add `guardrailConfig={"guardrailIdentifier","guardrailVersion","trace"}`
    - call `resp = self._client.converse(**kwargs)`
    - if `resp.get("stopReason") == "guardrail_intervened"`: extract blocked text from
      `output.message.content[0].text` and `trace.guardrail`, `raise GuardrailInterventionError(...)`
    - else return `resp["output"]["message"]["content"][0]["text"]`
    - wrap `botocore.exceptions.ClientError`: re-raise as-is (the activity classifies it) — keep client transport-only.

### 2. `activities/tool_activities.py`
- Remove `from litellm import completion`; add `import asyncio` and
  `from shared.bedrock_client import BedrockLLMClient, GuardrailInterventionError` and
  `from botocore.exceptions import ClientError`.
- `__init__`: replace the `llm_model`/`llm_key`/`llm_base_url` reads with `self.llm_client = BedrockLLMClient()`.
  Update the print/log lines to report model id / region / guardrail-on.
- `agent_toolPlanner` (lines 111–154): keep the system/user message construction (the date suffix and
  `context_instructions`), but call the client:
  ```python
  system = input.context_instructions + ". The current date is " + datetime.now().strftime("%B %d, %Y")
  try:
      response_content = await asyncio.to_thread(self.llm_client.converse, system, input.prompt)
  except GuardrailInterventionError as e:
      activity.logger.warning(f"Guardrail intervened: {e.assessment}")
      return {"next": "question", "response": e.message, "tool": None, "args": {}}
  except ClientError as e:
      code = e.response.get("Error", {}).get("Code", "")
      if code in NON_RETRYABLE_BEDROCK_ERRORS:  # AccessDenied/Validation/ResourceNotFound
          raise ApplicationError(f"Bedrock {code}: {e}", non_retryable=True)
      raise
  response_content = self.sanitize_json_response(response_content)
  return self.parse_json_response(response_content)
  ```
- Keep `sanitize_json_response` / `parse_json_response` unchanged.
- **Remove** the Ollama-specific `warm_up_ollama` method (litellm-era dead code).

### 3. `scripts/run_worker.py`
- Remove the `llm_model` read + Ollama warm-up block (lines ~24–56). Replace with a Bedrock-oriented
  startup print (`BEDROCK_MODEL_ID`, `AWS_REGION`, guardrail on/off). Worker registration unchanged.

### 4. `pyproject.toml`
- Remove the `litellm!=...` line; add `"boto3>=1.35,<2"`. (`botocore` comes transitively; `Stubber` is in `botocore.stub`.)

### 5. `.env.example`
- Replace the `### LLM configuration` block (`LLM_MODEL`/`LLM_KEY`) with:
  ```
  ### Bedrock LLM configuration
  BEDROCK_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0
  AWS_REGION=us-east-1
  AWS_ACCESS_KEY_ID=
  AWS_SECRET_ACCESS_KEY=
  # AWS_SESSION_TOKEN=        # if using temporary creds
  # BEDROCK_TEMPERATURE=0.0
  # BEDROCK_MAX_TOKENS=1024
  ### Bedrock Guardrails (optional — both required to enable enforcement)
  # BEDROCK_GUARDRAIL_ID=
  # BEDROCK_GUARDRAIL_VERSION=1   # or DRAFT
  # BEDROCK_GUARDRAIL_TRACE=enabled
  ```

### 6. Tests
- `tests/test_tool_activities.py`: change import to `BedrockLLMClient`; replace
  `patch("activities.tool_activities.completion")` with patching `BedrockLLMClient.converse`
  (return the JSON *string* directly, e.g. `'{"next":"confirm","tool":"TestTool","response":"Test response"}'`).
  Update `test_agent_toolPlanner_success` assertions (no `choices`/`message.content`; assert on parsed dict).
  **Delete** `test_agent_toolPlanner_with_custom_base_url` (base_url removed); replace with a guardrail-intervention
  test: patch `converse` to `side_effect=GuardrailInterventionError("Blocked.", {...})` and assert the activity
  returns `{"next":"question","response":"Blocked.","tool":None,"args":{}}`. Keep the JSON-parse-error test
  (patch `converse` to return non-JSON, assert it raises).
- NEW `tests/test_bedrock_client.py`: botocore `Stubber` tests —
  (a) feed a canned Converse envelope (`output.message.content[0].text`) and assert `converse()` returns the text
  and that request params include `system`, `messages` with `guardContent`, and `inferenceConfig`;
  (b) feed `stopReason="guardrail_intervened"` envelope and assert `GuardrailInterventionError` with message+assessment;
  (c) with guardrail env unset, assert no `guardrailConfig` key and plain-text content.
- `conftest.py`: no change expected (fixtures don't touch litellm).

### 7. Docs
- `docs/setup.md` and `README.md`: replace litellm/`LLM_MODEL` configuration sections with Bedrock setup
  (creds, region, inference profile, guardrail env vars). Note that the model id is an inference profile.

## Temporal-developer conformance
- Signals (`user_prompt`, `confirm`, `end_chat`, debugging toggles), queries (`get_conversation_history`,
  `get_agent_goal`, `get_latest_tool_data`, summary) and the continue-as-new loop are **unchanged**.
- All Bedrock I/O stays inside an **activity** (no workflow non-determinism). Non-retryable config/auth
  failures use `ApplicationError(non_retryable=True)`; transient errors rely on the existing retry policy.
  Existing LLM activity timeouts in `workflows/workflow_helpers.py` are kept (Haiku is fast; revisit only if needed).

## Verification

**Offline (mocks first):**
```bash
cd ../tai-bedrock
uv run pytest --workflow-environment=time-skipping
uv run pytest tests/test_bedrock_client.py tests/test_tool_activities.py -v
uv run poe lint
```
All activity/workflow tests pass with `converse` mocked; Stubber tests validate request shaping,
response parsing, and guardrail intervention — no AWS calls.

**Live smoke (after real `.env` is copied in):**
1. Confirm the inference profile id resolves: `aws bedrock list-inference-profiles --region $AWS_REGION` (verify Haiku 4.5 profile).
2. Start Temporal dev server (`temporal server start-dev`, defaults to `localhost:7233`).
3. `uv run scripts/run_worker.py` (startup print shows Bedrock model/region/guardrail status).
4. `uv run uvicorn api.main:app --reload`.
5. Drive a full chat: `POST /start-workflow`, then `POST /send-prompt`; `GET /get-conversation-history`
   and `GET /tool-data` to confirm tool planning + execution work end-to-end against Bedrock.
6. **Guardrail check** (if a guardrail is provisioned): send a prompt that trips the policy; confirm the
   agent replies gracefully (`next:"question"` with the blocked message) and the workflow keeps running
   (no crash, no retry storm). Verify the `trace.guardrail` assessment appears in worker logs.
7. **Fail-fast check**: temporarily set a bad `BEDROCK_MODEL_ID`; confirm the workflow surfaces a
   non-retryable error promptly instead of retrying for 30 minutes.
