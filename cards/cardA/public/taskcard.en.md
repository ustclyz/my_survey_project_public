# Task card A: First light

A season to learn the rules at a classic southern site.

## At a glance

| | |
|---|---|
| Site | Paranal, Chile (virtual). Latitude −24.62°, longitude −70.40°. |
| Survey | 2026-10-01 to 2027-01-31, 123 nights. You observe when the sun is below −18°. |
| Targets | 30,000 targets on 6,000 deg² of sky, in 3 regions. 1,500 are required. |
| Instrument | 16 contiguous fibre assignment cells in a 4 × 4 grid. The field covers 6.4 deg² and is about 2.53° across. |
| Time limit | 900 s of wall-clock time for the whole survey. |
| Weather | Hidden. Only the public inputs (targets, sky outline, site, telescope and fibres, observing schedule, scoring parameters) are published, when the competition starts. During a run your agent receives a short bulletin every 15 minutes and a forecast about once a week. |
| Extra messages | Time-limited observation requests (`observation_request`) and their results (`observation_request_result`). |

## Your goal

1. Observe as many valuable targets as you can, as well as you can.
2. Observe every **required** target well enough. Each one you miss costs 50 points.
3. Spread your work across the sky. Leaving parts of the sky empty costs up to 200 points.

## What your agent gets

**Once, at the start (`initialize`):**
- the site, the list of nights and when each night starts and ends;
- every target: position, class, brightness, weight and whether it is required;
- the sky regions, the instrument layout and the full score settings;
- the time limit.

**At every decision (`decision_request`):**
- the current time;
- the latest bulletin and forecast, and all messages since your last decision;
- the result of your last observation: which targets hit their fibre, and their scores;
- the time-limited observation requests in progress and how far along they are (`active_requests`);
- the time you have left.

## What your agent sends

One action per decision:

| Action | Meaning |
|---|---|
| `observe` | Point the telescope, put up to 16 targets on fibres, expose for 60–3600 s, and declare a program (DARK, BRIGHT or BACKUP). |
| `wait` | Let time pass: a number of seconds, or until a given time (for example the next night). |
| `report` | Say that the instrument is faulty now. Right: +100, and the fault is repaired. Counting from the start or from your last correct report, the first 2 false reports are free; each later false report costs −150. At most 32 `report` actions in a row. |
| `finish` | End the survey now. |

## How the score works

- A target scores only if it falls in its assigned fibre's cell and stays at or above 30° altitude.
- Its score grows with brightness, exposure time and sky quality, up to a cap.
- A matching program adds 20% (DARK), 12% (BRIGHT) or 6% (BACKUP). A wrong program adds nothing.
- Only the best exposure of each target counts.
- Time-limited observation requests are issued during the survey. Each lists a set of targets, a minimum number to complete, a completion threshold, a reward and a deadline. Valid exposures that lie wholly between its issue and its deadline count automatically; there is nothing to accept. Reaching the minimum (judged on the completion factor, without the program bonus) earns the reward at the deadline; a missed request costs nothing.
- Final score = sum of best scores − 50 × missing required targets − unevenness penalty ± reports + request rewards.

**Example.** A target has brightness 0.60 and weight 1.0. You expose it for 900 s. The sky quality is 0.75.
Its factor is 0.60 × 900 × 0.75 ÷ 450 = 0.90. You declared DARK and the sky was DARK, so the score is
1.0 × 0.90 × 1.20 = **1.08**. With a 300 s exposure the factor is only 0.30. A required target needs at
least 0.50, so it would still count as missing.

## Common mistakes

- Printing logs to stdout. Only JSON answers go to stdout. Logs go to stderr.
- Answering with the wrong `decision_sequence`, or adding unknown fields. The run stops with `agent_error`.
- Calling a language model on every decision. The time limit runs out long before the survey ends.
- Short exposures on faint targets. Exposures do not add up; only the best one counts.
- Pointing low in a direction a bulletin warns about.
- Reporting a fault after one bad exposure. Weather also lowers scores.
- Ignoring `active_requests`. A request only counts exposures completed after it is issued and before its deadline; observing later does not count.

## Try it

1. After the competition starts, download the public inputs of card A (targets, sky outline, observing schedule, fibre layout, scoring parameters) from the Resources page and use them to check your planning.
2. The weather of this card is hidden, so it cannot be scored locally. Tune your agent locally on cards α, β, γ and δ, whose weather is public.
3. Pack with `python3 pack_agent.py --out ../my-agent.zip` and submit it on Participate. Each evaluation runs cards A, B, C and D once each, up to 900 s per card. Daily limits are on the Rules page.

The live board updates in real time but does not decide the final ranking. See the Rules page for the final ranking.
