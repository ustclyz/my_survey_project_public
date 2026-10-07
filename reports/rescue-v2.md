# Rescue v2 — 2026-10-07

Based on the local rescue-v1 (upstream exp/duty-log 84258a9 plus delivery fixes).

Changes:
- The existing duty-log model call also extracts confirmed flat-lamp/cap test
  windows. The agent waits during these windows instead of observing invalid
  frames or treating the test as an instrument fault. Known repair events still
  take precedence. Both exposures and waits respect upcoming boundaries.
- Reject malformed/reversed/over-four-hour/out-of-survey test intervals and
  fault timestamps outside the survey. Process one handover per model call to
  limit expanded JSON output length; retain the rest of the queue.
- Cache target-duration gains within each decision only, bounded at 32768 entries.
  No cross-turn cache: changing weather, repair, requests and target progress are
  recalculated. PRO_CACHE_GAIN=0 provides the original comparison path.

Verification: 82 full-suite tests passed, then 8 affected duty/window tests passed
after reducing the extraction batch to one. No live model or official score run.

Paired two-night cardB synthetic probe, rescue enabled:

| Cache | Estimated required completed | Decisions | CPU seconds |
|---|---:|---:|---:|
| off | 601 | 264 | 112.2 |
| on | 601 | 264 | 96.4 |

Both action traces have SHA256
`d4622d22e73c71e341508eeb608c2ad69bfb4f4664e0dbef180badb06a251fe5`.
Thus the measured action sequences match exactly. The 14% CPU reduction is a
single-run observation under concurrent load, not a stable benchmark or a score
increase. Do not compare absolute CPU seconds to the earlier v1 experiment.

Latest official logs remain pro-fast-fault-v4 (2f5adb7e), not this version.
Their A1/B1/C1/D1 handover streams contain 31/41/30/62 messages mentioning tests.
The new extraction has only been validated with fixture model responses: actual
model recall/precision and net leaderboard benefit remain unmeasured.

Recommended upload name: rescue-v2. Preserve v1/v4 as controls. Compare official
score, missing required targets, false reports, model failures and CPU exhaustion
before selecting the final version. There is no demonstrated +10000 score yet.
