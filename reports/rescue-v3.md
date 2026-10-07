# rescue v3 — official-log regression repair

Input: gosim-observer-1fdc6d60-2026-10-07.zip, rescue v2.

## Evidence
- D1 hit the 10800-second real-time hard cap: run 10802.867s; CPU 440.195s. This is not a CPU-budget exhaustion.
- D1 made 15747 observe actions, versus 6987 in pro-fast-fault-v4; the v2 five-minute horizon restriction increases decision pressure. Removing it addresses a contributing factor, not proof that all timeout causes are gone.
- duty_log calls failed with ValueError 41/51/48/58 times on A1/B1/C1/D1. No successful scheduled duty reports were logged. Original logs lack response bodies/finish reasons, so truncation is a hypothesis, not a confirmed cause of all 198 failures.
- Absolute low-quality fault detection was placed after the E sample-count/freshness gate. A regression test reproduced suppression when E samples are absent.

## Changes
1. Remove the unconditional 300-second low-quality exposure horizon. Preserve scheduled fault/test boundaries and required-target rescue selection.
2. Evaluate fresh absolute-quality evidence independently of E availability, while retaining earthquake protection, report spacing and false-report limits. Do not reuse pre-report or old-hour samples.
3. Integrate existing conservative public maintenance notice parser: explicit times/timezones, cancellation/postponement, and flat/cap windows. Ambiguous/conditional language remains with the LLM. No hidden event dates or target IDs are embedded. Both existing LLM advisory stages remain enabled.
4. Raise configurable LLM output cap from 2000 to 8192 tokens (PRO_LLM_MAX_TOKENS). Reject truncated replies and ambiguous multiple JSON objects. Accept one complete JSON object despite non-JSON prose braces. Log structural failures without response text/secrets.
5. Parse/merge local maintenance notices only on new observation-request messages, not every decision.

## Offline validation
- Regression tests first reproduced six failures in v2 behavior.
- Whole suite: 100 passed in 31.20 seconds; includes real agent.py stdin/stdout smoke test without network.
- Public-notice replay counts in rescue-v3-public-notices.json. Those counts are extraction coverage, not verified repair precision, recall, or simulated score.
- No paid API calls made for validation. Live school/API compatibility and actual improvement need an official evaluation.

## Deployment
Upload my-survey-rescue-v3-20261007.zip as a new version. Model settings remain in the platform. Old versions are retained. No GitHub push or official submission performed.

API reference for truncation semantics: https://api-docs.deepseek.com/api/create-chat-completion/
