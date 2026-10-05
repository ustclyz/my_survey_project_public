# 协议字段对照表 (protocol schema)

本文件把智能体用到的**全部协议字段集中定义**, 并逐字段标注来源, 便于与官方协议
对账. 官方协议一旦公布/变更, 只需修改 `protocol.py` 一处即可对齐, 不影响决策内核.

- **协议版本**: `participant-agent-protocol-v4`
- **传输格式**: JSON Lines (jsonl-v4), 一行一个 JSON 对象
- **提交清单**: 项目根 `observer.project.json` 声明 `"protocol": "jsonl-v4"`
- **本项目的字段定义位置**: `protocol.py` (`_RESPONSE_FIELDS`, `InitializeData`,
  `DecisionRequest`, `send_response`)

## 来源标注

| 标注 | 含义 |
|---|---|
| 规则原文 | 来自官方 `docs/participant-guide.{zh,en}.md` 与赛题规则, 权威 |
| 官方示例 | 来自官方 `资源/python.zip` 的 `agent_core/protocol.py`, 权威 |
| 任务卡 | 来自任务卡 `cards/*/public/taskcard.zh.md` / `v4_scenario.json` |
| 推断-待核对 | 由本项目推断, 若官方补充字段定义需核对 (本仓库中未发现该情形) |

> 核对结论: 官方在 `资源/python.zip`、`资源/python-pro.zip` 中**提供了完整示例
> 项目与权威协议文档** (`docs/participant-guide.en.md`, 79KB). 因此本表所有字段
> 均有权威来源, 无"未发现官方协议样例"的情况.

---

## 1. 顶层消息 (stdin / stdout)

### 1.1 `initialize` (服务端 -> 智能体, 无需回复)

| 字段 | 类型 | 来源 | 说明 |
|---|---|---|---|
| `protocol_version` | string | 规则原文 | 固定 `participant-agent-protocol-v4` |
| `message_type` | string | 规则原文 | `"initialize"` |
| `payload` | object | 规则原文 | 见 1.1.1 |

#### 1.1.1 `initialize.payload`

| 字段 | 类型 | 来源 | 说明 |
|---|---|---|---|
| `schema_version` | string | 规则原文 | `v4-initialize-v1` |
| `task_card` | object | 规则原文 | `{card_id, scenario_slug, phase}` |
| `site.latitude_deg` | number | 规则原文 | 站点纬度 |
| `site.longitude_deg` | number | 规则原文 | 站点经度 |
| `site.utc_offset_hours` | number | 规则原文 | UTC 偏移 |
| `site.sun_altitude_limit_deg` | number | 规则原文 | 太阳低于此值才可观测 (如 -18) |
| `site.minimum_altitude_deg` | number | 规则原文 | 目标最低高度角 (如 30) |
| `survey.start_utc` | string | 规则原文 | 巡天开始 (ISO, Z 结尾) |
| `survey.end_utc` | string | 规则原文 | 巡天结束 |
| `survey.slot_seconds` | int | 规则原文 | 每 slot 秒数 (当前 900) |
| `survey.nights[]` | array | 规则原文 | `{night_id, night_date, observing_start_utc, observing_end_utc, slot_count}` |
| `instrument.n_fibers` | int | 规则原文 | 光纤数 (如 16) |
| `instrument.grid_side` | int | 规则原文 | 方格边长 (如 4) |
| `instrument.fiber_area_deg2` | number | 规则原文 | 单根光纤面积 (如 0.4) |
| `instrument.gap_deg` | number | 规则原文 | 间隙 (如 0) |
| `instrument.glass_side_deg` | number | 规则原文 | 光纤玻璃边长 |
| `instrument.pitch_deg` | number | 规则原文 | 光纤中心间距 |
| `instrument.fov_side_deg` | number | 规则原文 | 视场边长 |
| `instrument.exposure.min_duration_seconds` | int | 规则原文 | 最短曝光 (60) |
| `instrument.exposure.max_duration_seconds` | int | 规则原文 | 最长曝光 (3600) |
| `scoring` | object | 规则原文 | 完整公开计分配置, 见 1.1.2 |
| `footprint[]` | array | 规则原文 | `{component_id, vertices[[ra,dec],...]}` |
| `targets.columns[]` | array[string] | 规则原文 | 目标表列名 |
| `targets.rows[][]` | array | 规则原文 | 与 columns 同序的行 |
| `limits.global_wallclock_seconds` | int | 规则原文 | 整卡预算 (归一化 CPU 秒, 900) |
| `limits.max_consecutive_reports` | int | 规则原文 | 连续 report 上限 (32) |
| `limits.response_max_bytes` | int | 规则原文 | 单条回复大小上限 (524288) |

目标列名 (`targets.columns`): `target_id, ra_deg, dec_deg, target_class,
feature_flux, science_weight, required` (来源: 规则原文).

#### 1.1.2 `initialize.payload.scoring`

| 字段 | 来源 | 说明 |
|---|---|---|
| `q0` | 规则原文 | 天空质量基准 (如 0.68) |
| `flux_zero_point` | 规则原文 | 流量零点 (如 0.5) |
| `exposure_zero_point_seconds` | 规则原文 | 曝光零点秒 (如 900) |
| `airmass_exponent` | 规则原文 | 大气质量指数 (如 0.6) |
| `lunar_model.{maximum_penalty,altitude_exponent,angular_decay_scale_deg}` | 规则原文 | 月光模型 |
| `program.bands.{DARK,BRIGHT}` | 规则原文 | 程序档位阈值 |
| `program.multipliers.{DARK,BRIGHT,BACKUP}` | 规则原文 | 1.20 / 1.12 / 1.06 |
| `program.mismatch_multiplier` | 规则原文 | 声明错档 ×1.00 |
| `required.penalty_per_missing` | 规则原文 | 漏必观测扣 50 |
| `required.observed_factor_threshold` | 规则原文 | 0.5 |
| `uniformity.{weight,ra_band_width_deg,observed_factor_threshold}` | 规则原文 | 均匀度, 最多扣 200 |
| `reporting.{correct_reward,false_penalty,false_report_free_allowance,max_consecutive_reports}` | 规则原文 | +100 / -150 / 2 / 32 |
| `observation_requests.{completion_factor_threshold,miss_penalty}` | 规则原文 | 请求门槛 0.5, miss_penalty=0 |

### 1.2 `decision_request` (服务端 -> 智能体, 必须回复一次)

| 字段 | 类型 | 来源 | 说明 |
|---|---|---|---|
| `protocol_version` | string | 规则原文 | 同 1.1 |
| `message_type` | string | 规则原文 | `"decision_request"` |
| `decision_sequence` | int | 规则原文 | **必须在回复中回填** |
| `payload.schema_version` | string | 规则原文 | `v4-decision-snapshot-v1` |
| `payload.now_utc` | string | 规则原文 | 当前模拟时间 |
| `payload.survey_end_utc` | string | 规则原文 | 巡天结束时间 |
| `payload.observe_action_index` | int | 规则原文 | 已执行 observe 数 |
| `payload.running_total` | number | 规则原文 | 各目标最好得分之和 |
| `payload.wallclock` | object | 规则原文 | 时间预算, 见 1.2.1 |
| `payload.latest_bulletin` | object\|null | 规则原文 | 最新公报, 见 1.3 |
| `payload.latest_forecast` | object\|null | 规则原文 | 最新预报, 见 1.4 |
| `payload.active_requests[]` | array | 规则原文 | 进行中的限时请求, 见 1.5 |
| `payload.new_messages[]` | array | 规则原文 | 上次决策后的全部新消息 |
| `payload.last_result` | object\|null | 规则原文 | 上次动作结果, 见 1.6 |

#### 1.2.1 `payload.wallclock`

| 字段 | 来源 | 说明 |
|---|---|---|
| `remaining_seconds` | 规则原文 | 剩余**归一化 CPU** 预算 |
| `remaining_real_cpu_seconds` | 规则原文 | 剩余预算折算为本机真实 CPU 秒 (与 `time.process_time()` 同单位) |
| `wall_remaining_seconds` | 规则原文 | 距 30 分钟硬上限的真实秒数 |
| `speed_factor` | 规则原文 | 本机速度因子 (1.0=中位机) |
| `elapsed_seconds`,`cpu_seconds`,`wait_seconds`,`clock_mode` | 规则原文 | 计时细节 |

预算保护以 `remaining_real_cpu_seconds` (回退 `remaining_seconds`) 为准.

### 1.3 `bulletin`

| 字段 | 来源 | 说明 |
|---|---|---|
| `record_type` | 规则原文 | `"bulletin"` |
| `slot_id` | 规则原文 | slot 标识 |
| `night_id` | 规则原文 | 夜标识 |
| `issued_at_utc` | 规则原文 | 发布时刻 |
| `initial` | 规则原文 | 仅全程第一条为 true (含地形遮挡) |
| `notices[].event_kind` | 规则原文 | rain/storm/overcast/haze/cold_snap/rocket_launch/earthquake/terrain_obstruction |
| `notices[].direction` | 规则原文 | N/NE/E/SE/S/SW/W/NW/ALL |

### 1.4 `forecast`

| 字段 | 来源 | 说明 |
|---|---|---|
| `record_type` | 规则原文 | `"forecast"` |
| `issued_at_utc` | 规则原文 | 发布时刻 |
| `coverage_start_utc`,`coverage_end_utc` | 规则原文 | 覆盖时间范围 |
| `notices[].event_kind` | 规则原文 | 同上 |
| `notices[].direction` | 规则原文 | 同上 |
| `notices[].nights[]` | 规则原文 | 受影响夜日期 (仅预报有) |

### 1.5 `active_requests[]` / `observation_request`

| 字段 | 来源 | 说明 |
|---|---|---|
| `record_type` | 规则原文 | `"observation_request"` |
| `request_id` | 规则原文 | 仅用于追踪, **observe 动作不得包含** |
| `issued_at_utc`,`deadline_utc` | 规则原文 | 只有完全落在窗口内的曝光计入 |
| `target_ids[]` | 规则原文 | 涉及的目标 |
| `minimum_completed` | 规则原文 | 最少完成数 |
| `completion_factor_threshold` | 规则原文 | 每目标完成因子门槛 (不计程序加成) |
| `completion_reward` | 规则原文 | 一次性奖励 |
| `reason` | 规则原文 | 简述 |
| `completed_target_ids[]`,`completed_count`,`remaining_count` | 规则原文 | 仅 active_requests 追加的进度字段 |

`observation_request_result`: `record_type, issued_at_utc, request_id, status
(completed|expired), completed_target_ids, completed_count, minimum_completed,
score_delta, revised`.

### 1.6 `last_result`

| 字段 | 来源 | 说明 |
|---|---|---|
| `action` | 规则原文 | `"observe"` / `"wait"` / `"report"` |
| `observe_index` | 规则原文 | observe 序号 |
| `assigned_count` | 规则原文 | 上次指派目标数 |
| `hit_count` | 规则原文 | 命中并过高度角的目标数 |
| `hits[].target_id`,`hits[].score` | 规则原文 | 命中目标及其本次得分 (**不返回光纤号**) |
| `report_result`: `correct,repaired,score_delta` | 规则原文 | report 反馈 |

### 1.7 `finish` (服务端 -> 智能体)

`payload.{schema_version, termination_reason, decisions, observe_actions,
last_decision_sequence, grace_seconds}`. (来源: 规则原文)

### 1.8 其他 `new_messages` 记录 (Hard 模式)

- `state_resync`: `record_type, invalidated_window{action_index_start,
  action_index_end_exclusive}, best_scores[], observation_requests[]`
- `report_result`: `record_type, issued_at_utc, correct, repaired, score_delta`
- `pointing_offset`: Hard 模式指向偏移 (以任务卡为准)

---

## 2. `decision_response` (智能体 -> 服务端)

外层 envelope (`send_response` 构造):

| 字段 | 来源 | 说明 |
|---|---|---|
| `protocol_version` | 规则原文 | `participant-agent-protocol-v4` |
| `message_type` | 规则原文 | `"decision_response"` |
| `decision_sequence` | 规则原文 | 回填请求中的整数 |
| `action` | 规则原文 | `observe`/`wait`/`report`/`finish` |
| `reason` | 规则原文 | 可选, 不影响计分 |
| `decision_source` | 规则原文 | 可选, 不影响计分 |

### 2.1 `observe` 允许字段 (官方示例 `_RESPONSE_FIELDS`)

| 字段 | 类型 | 来源 | 约束 |
|---|---|---|---|
| `pointing.alt_deg` | number | 规则原文 | `[0,90]` |
| `pointing.az_deg` | number | 规则原文 | `[0,360)`, 天顶时也须给 (决定方格朝向) |
| `assignments` | object | 规则原文 | `{纤维ID字符串: 目标ID}`; 必填; 同一纤维/目标各至多一次 |
| `duration_seconds` | int | 规则原文 | `[60,3600]` |
| `program` | string | 规则原文 | DARK/BRIGHT/BACKUP (省略按 BACKUP) |

### 2.2 `wait` 允许字段

`duration_seconds` (int, `[60,3600]`) **或** `until_utc` (以 `Z` 结尾), 二者互斥.

### 2.3 `report` / `finish` 允许字段

无动作专属字段 (仅 envelope + 可选 reason/decision_source).

### 2.4 非法情形 (会导致 `agent_error`, `protocol.py` 已全部规避)

非 JSON / 超出 `response_max_bytes` / protocol_version 或 decision_sequence 错 /
未知或多余字段 / 越界或非有限数值 / 非整数秒 / 未知目标 / 重复纤维或目标 /
超过连续 report 上限.

---

## 3. 本项目内部模型 (不下发平台)

`models.py` 中 `Action` / `NightPlan` / `DecisionState` 为内部结构, 不直接序列化到
stdout; 由 `Action.to_protocol_fields()` 映射到第 2 节字段.
