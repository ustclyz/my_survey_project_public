# Duty-log delivery fix — 2026-10-07

Base: origin/exp/duty-log, 84258a9 (11:40 +0800), fetched from
https://github.com/ustclyz/my_survey_project_public.

The branch already adds model-based handover parsing. Five regression tests
reproduced integration errors before the fix:

1. Completed model calls were collected only when another handover arrived.
   Poll every decision and drain queued messages without empty prompts.
2. Rejected submissions discarded the queue; selecting the last four chunks
   also dropped older messages. Send oldest batches and retain unsubmitted data.
3. The schedule was append-only. Add cancelled_utc and send pending events plus
   original message timestamps so corrections can explicitly remove old events.
4. Exposures and waits could cross known repair times. Bound the planner horizon
   before generating predictions, and cap waits at the next event.
5. Scheduled repairs did not update report counters/last-report time, and several
   overdue translations could cause duplicate reports. Update accounting and
   consume overdue events together. Reset consecutive-report counts on all waits.

Validation: 73 tests passed, including pro stdin/stdout smoke tests. No live
model calls or official evaluation were performed for this patch.

Latest supplied evaluation: 2f5adb7e, version pro-fast-fault-v4. All cards ended
with survey_complete. This evaluation is NOT a measurement of the patched code.

| Card | Score | Missing required | Correct reports |
|---|---:|---:|---:|
| A | 22643.92 | 7 | 1 |
| B | 36901.09 | 3 | 1 |
| C | 25039.28 | 0 | 1 |
| D | 31242.40 | 3 | 2 |
| A1 | -6897.70 | 354 | 26 |
| B1 | -44811.66 | 1140 | 19 |
| C1 | -23009.84 | 567 | 10 |
| D1 | 33246.26 | 24 | 108 |

Compared with prior pro-kernel-port results, the hard-card gains support focusing
on timely repairs. They do not isolate the causal effect of each upstream change.
No unsupported target-priority experiments were merged into this branch.

Remaining limits: model extraction can be wrong, cancelled_utc must be emitted
correctly, failed model requests still rely on statistical detection, and the
upstream 12-hour report window remains. This patch does not add the separate
deterministic maintenance parser/test-window calendar. Compare official results
before choosing a final version. Suggested upload name: duty-log-delivery-fix.
