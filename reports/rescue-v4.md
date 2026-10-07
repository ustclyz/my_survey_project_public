# Rescue v4: duty output reliability experiment
Baseline fd4f2aaa completed all 8 cards. A-D mean 29618.43710025; A1 25028.971735, B1 40142.161881, C1 20901.948197, D1 48265.621845.
All surveys completed. Duty calls truncated 17/39/16/0 times on A1/B1/C1/D1. D1 also had 50 false reports; their source is not yet established and this revision does not change report policy.
Change only DeepSeek flash/v4-pro duty requests: reasoning_effort=low and max_tokens=16384 (PRO_DUTY_MAX_TOKENS override). Other models and nightly advisory calls unchanged. This is an experiment to improve completed answers, not a guaranteed score increase; lower reasoning may affect interpretation accuracy and requires formal evaluation.
Reference: https://api-docs.deepseek.com/guides/thinking_mode/
