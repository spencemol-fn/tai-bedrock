# Verification Steps

## 1. Start Worker
```bash
uv run scripts/run_worker.py
```
Verify output is printed to console.

## 2. Start API
```bash
uv run uvicorn api.main:app --reload
```
API runs on port 8000.

## 3. Start Frontend
```bash
cd frontend && npm install && npx vite
```
Frontend runs on port 5173.

## 4. Drive a Chat
- POST `/start-workflow` with a prompt
- Verify `GET /get-conversation-history` shows tool planning working end-to-end against Bedrock

## 5. Fail-fast Check
- Set `BEDROCK_MODEL_ID=bad-model-id`
- Confirm workflow surfaces a non-retryable error promptly
- Verify no 30-min retry storm occurs

## 6. Guardrails (when provisioned)
- Set `BEDROCK_GUARDRAIL_ID` and `BEDROCK_GUARDRAIL_VERSION`
- Send a policy-violating prompt
- Confirm agent replies gracefully (next: "question") with no crash/retry