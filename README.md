# GOSIM 巡天智能体项目 (Agentic Observer)

本仓库是 GOSIM 2026 Agentic Observer Hackathon 的**完整可提交智能体项目**:
一个自主巡天观测智能体, 在模拟天文望远镜巡天环境中逐轮接收 `decision_request`,
自主决策并返回 `observe` / `wait` / `report` / `finish` 动作, 目标是最大化科学得分.

- **协议**: `participant-agent-protocol-v4` (JSON Lines, `jsonl-v4`)
- **提交清单**: `observer.project.json` (ZIP 根, 声明 `"protocol": "jsonl-v4"`)
- **入口**: `python3 -u agent.py`
- **评奖硬门槛**: 至少 2 个环节由 LLM 驱动 —— 本项目的**任务规划**与**行动决策**
  两个环节由 LLM 承担 (另有**计划自适应**加分环节); 未配置密钥时静态回退可跑.

---

## 1. 项目结构

```
my_survey_project/
├── agent.py                 # 智能体主程序: 命令行入口 + 评测主循环
├── protocol.py              # 协议适配层: initialize/decision_request 解析, 动作构造, JSON 编解码
├── planner.py               # 决策内核: 封装 preplan.py 为规划工具 + 几何/落成动作
├── llm.py                   # LLM 客户端 + 环节A任务规划 / 环节B行动决策 / 环节C计划自适应
├── models.py                # 数据模型: Target / Action / NightPlan / DecisionState / LLMConfig
├── config.py                # 运行配置: 从环境变量读取密钥与模型 (多服务商)
├── pack_agent.py            # 打包脚本: 打成平台可提交的 ZIP (<50MB)
├── observer.project.json    # 平台项目清单 (jsonl-v4)
├── schemas/
│   └── protocol.md          # 协议字段集中定义 + 逐字段来源标注 (便于对账)
├── tests/
│   ├── test_protocol.py     # 协议解析/动作构造单测
│   ├── test_planner.py      # 规划内核单测 (含 LLM 环节假客户端测试)
│   └── replay_demo.py       # 本地闭环回放 (练习卡 alpha)
├── preplan.py               # 原有: 独立预规划模块 (纯静态, 被 planner 复用)
├── card_static_check.py     # 原有: 任务卡静态校验 (8/8)
├── cards/                   # 原有: 8 张任务卡解压资源 (只读)
├── reports/                 # 原有: 静态校验报告产物
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 2. 启动方式

```bash
# 平台评测: 读取 stdin 的 JSON Lines, 写 stdout 的 decision_response
python3 -u agent.py

# 本地调试 (可用 replay_demo 模拟闭环, 见第 5 节)
```

程序约定:
- **标准输出只写 JSON 回复**, 所有日志写**标准错误**;
- 收到 `initialize` 不回复; 每个 `decision_request` 回复**恰好一个** `decision_response`;
- 回复必须回填相同的整数 `decision_sequence`, 且不含未知字段.

---

## 3. 环境变量 (密钥绝不写入源码)

模型 API 由选手自备, 密钥通过环境变量注入. 支持多组服务商前缀, **按优先级选择第一组可用**:

| 优先级 | 密钥 | 可选 base_url / model |
|---|---|---|
| 1 | `OPENAI_API_KEY` | `OPENAI_BASE_URL` / `OPENAI_MODEL` |
| 2 | `KIMI_API_KEY` | `KIMI_BASE_URL` / `KIMI_MODEL` |
| 3 | `DEEPSEEK_API_KEY` | `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL` |
| 4 | `MOONSHOT_API_KEY` | `MOONSHOT_BASE_URL` / `MOONSHOT_MODEL` |

示例 (kimi coding):

```bash
export OPENAI_BASE_URL=https://api.kimi.com/coding/v1
export OPENAI_MODEL=kimi-for-coding
export OPENAI_API_KEY=<你的密钥>       # 只在运行页/环境变量设置, 绝不提交
```

示例 (中科大词元计划 USTC, OpenAI 兼容, 本地调试实测可用):

```powershell
$env:OPENAI_BASE_URL = "https://api.llm.ustc.edu.cn/v1"
$env:OPENAI_MODEL    = "deepseek-flash-2"   # 或 claude-sonnet-4-6 等
$env:OPENAI_API_KEY  = "<你的密钥>"          # 仅本会话环境变量, 绝不写入源码
```

- `LLM_TIMEOUT_SECONDS`: 单次调用超时 (默认 30s, 裁剪到 [5,120]).
- `OBSERVER_MODEL_DISABLED=1`: 平台无模型评测时置 1, 程序走纯静态模式, 不需要密钥.
- 平台会设置 `HTTPS_PROXY`, 常见 SDK 无需额外配置.

**未配置任何密钥时不会退出**, 而是退化为"纯静态规划 + 启发式决策"并在 stderr 提示
`未启用 LLM, 仅静态模式`, 便于本地调试.

依赖: 优先使用 `openai` 官方 SDK; 缺失时 `llm.py` 自动落到标准库
`urllib.request` 直连 `{OPENAI_BASE_URL}/chat/completions`, 无需额外配置.

**要求模型支持 `response_format={"type":"json_object"}`** (本项目三个 LLM 环节都
要求返回 JSON). USTC 的 `deepseek-flash-2` 与 `claude-sonnet-4-6` 均已实测支持.

---

## 4. 协议说明 (摘要, 完整见 `schemas/protocol.md`)

### 4.1 智能体收到什么

- **`initialize`** (一次, 无需回复): 站点、夜历、全部目标 (`targets.columns` +
  `targets.rows`)、天区、仪器布局、完整计分参数、时间限制.
- **`decision_request`** (每轮): `now_utc`、`survey_end_utc`、`wallclock`
  (剩余时间)、`latest_bulletin` / `latest_forecast`、`new_messages`、`last_result`、
  `active_requests`. `decision_sequence` 必须回填.
- **`finish`**: 巡天结束, 记录摘要后退出.

### 4.2 智能体发送什么 (`decision_response`)

| 动作 | 字段 |
|---|---|
| `observe` | `pointing{alt_deg,az_deg}` + `assignments{"纤维":"目标ID"}` + `duration_seconds`(60-3600) + `program`(DARK/BRIGHT/BACKUP) |
| `wait` | `duration_seconds`(60-3600) **或** `until_utc`(以 Z 结尾), 二者互斥 |
| `report` | (无动作字段) |
| `finish` | (无动作字段) |

可选 `reason` / `decision_source`. 本项目在 `protocol.py` 严格裁剪字段, 避免
`agent_error`.

---

## 5. 本地自测步骤

```powershell
# 1) 任务卡静态校验 (应 8/8 通过)
py card_static_check.py

# 2) 单元测试 (协议 + 规划, 全绿)
py -m pip install pytest
py -m pytest tests/test_protocol.py tests/test_planner.py -q

# 3) 本地闭环回放 (练习卡 alpha): 模拟 decision_request -> 智能体 -> decision_response
py tests/replay_demo.py --card alpha --rounds 60
#    验证: 输出只走 stdout JSON, 日志走 stderr, 动作合法, 不崩溃

# 4) 端到端 (可选): 允许调用真实 LLM
set OPENAI_API_KEY=...
py tests/replay_demo.py --card alpha --rounds 30 --with-llm
```

`replay_demo.py` 会打印动作统计与 `llm_seen` / `static_seen`,
用于确认 LLM 环节被触发或被明确标记静态回退.

---

## 6. 打包与提交

```powershell
# 打包 (<50MB; 自动排除 .env / __pycache__ / .git 等; 输出到项目外)
py pack_agent.py --out ..\my-agent.zip
```

提交方式二选一:
1. **公开 GitHub 仓库**: 把本目录推到 `my_survey_project_public`;
2. **≤50MB 私有 ZIP**: 上传 `pack_agent.py` 生成的 ZIP (作为完整项目).

---

## 7. 设计要点

- **复用 `preplan.py`**: 目标优先级 (`compute_priorities`) 与 4×4 光纤方格模型
  (`FiberGrid`) 直接复用, 未重写规划逻辑.
- **适配器模式**: 所有协议字段集中在 `protocol.py` 与 `schemas/protocol.md`;
  官方协议一旦变更, 只改这里即可对齐.
- **预算保护 (900s CPU 成败项)**:
  - **环节A (plan_night)** 仅在新夜调用一次, 并按夜缓存;
  - **环节B (decide_action)** **绝不每轮调用**: 只在关键点 (新夜 / 新限时请求 /
    结果异常) 或按周期的轮次调用; 周期由剩余 CPU 预算自适应 (预算越少间隔越大,
    上限 20 轮), 且总调用次数与每轮平均 CPU 占比均有硬上限;
  - 其余轮次用静态内核落成 `observe`, 保证巡天能在预算内完成;
  - 按 `wallclock` 剩余时间自适应收紧锚点搜索 (pace 0/1/2).
  本地实测 (USTC deepseek-flash-2): 60 轮 cardA 全流程约 110s, 仅 5 次环节B 调用.
- **观测记忆 (避免重复曝光)**: 记录已得分目标 (`observed_ids`) 并强惩罚, 使规划
  转向未观测目标; 记录已尝试目标 (`attempted_ids`) 弱惩罚; 记录上次指向并对重复
  视场施加惩罚 (规则 5.4: 同一目标多次曝光只算最好一次, 不累加).
- **空间收敛 / 扫描连续性**: 为避免相邻曝光指向大幅跳变 ("瞬移"), 在**质量带**
  (相对最优 20% / 绝对 1000 分) 内用"连续性目标函数"择优: 邻近项 (靠近上次指向)
  + 惯性项 (延续扫描方向) + 回访去重; 并生成"延续扫描"的候选视场中心使其进入
  搜索范围 (`Planner._continuity_bonus` / `_continuity_seed_centers`)。
  实测 (alpha, 40 轮) 相邻指向中位跳变 **44.7° → 12.5°**, 指向轨迹由瞬移变为平滑
  扫天; 代价约为 ~3% 去重目标数。
- **必观测保障 (首要得分项, 漏一个 −50)**: 未完成必观测目标 (a) 不被"已尝试"惩罚、
  (b) 作为**锚点视场中心**进入候选 (即使孤立也会被评估) 并获视场加成
  (`Planner._required_anchor_centers` / `required_field_bonus`); (c) 曝光时长按
  "达到完成因子 0.5 所需"给足且**不参与 duration_scale 缩放** (参考 Cao 2025 的
  曝光时间计算器思想). 实测 alpha 全巡天必观测漏失 **152 → 34**。
- **可审计**: 每次 LLM 调用的 system/user/返回都写 stderr (含时间戳与调用点).

> **运行可视化**: 项目外层 `survey_run_analysis.ipynb` 在本地驱动智能体跑闭环,
> 并绘制预测 vs 真实落点 (焦平面)、预测/真实光纤、天区覆盖、收敛性 (指向轨迹与
> 跳变) 等图, 便于直观评估策略质量。

### 本地合成天气评估 (`tools/`, 仅本地)

任务卡只下发 `public/` (目标/天区/夜历/计分配置), **不下发** `truth/` (逐 slot 的
seeing / transparency / sky_quality / instrument_efficiency、天气事件等). 为在本地
可量化评估策略, 本项目提供两个**仅本地**工具 (不进提交包, `pack_agent.py` 已排除):

- `tools/weather_gen.py`: 随机生成语义与官方一致的**合成天气真值**
  (`sim_data/<card>/weather_slots.csv` / `events.csv` / `meta.json`);
- `tools/local_scorer.py`: **复刻官方计分公式**的本地评分器, 用合成天气跑闭环并给出
  真实得分 (含完成因子 Q/g、程序档位/倍数、必观测扣分、Jain 均匀度)。

用法:

```powershell
py tools/weather_gen.py --card alpha --seed 20261005   # 生成合成天气
py tools/local_scorer.py --card alpha --seed 20261005 --rounds 5000
```

> 该工具链曾定位并修复一个**严重几何缺陷** (`tangent_offsets` 返回 `(north, east)`
> 却被当作 `(east, north)` 使用), 使必观测命中率从 26% 提升到 99%、本地合成得分
> 从 −5622 提升到 +4721.


---

## 8. 原有静态工具说明 (保留)

### 8.1 `card_static_check.py` — 任务卡静态校验

遍历 `cards/` 下 8 张卡, 检查必需文件/配置/核心统计, 报告写
`reports/static_check_report.txt`. 退出码 `0` 全部通过.

### 8.2 `preplan.py` — 独立预规划模块 (纯静态)

命令行 `py preplan.py --card alpha`; 作为库:

```python
from preplan import CardData, plan_card, plan_by_name
result = plan_by_name("cardA")
```

---

## 9. 注意事项

1. **只读原始资源**: `资源/` 下的原始 zip 不被修改或删除.
2. **密钥安全**: 密钥只经环境变量注入, 绝不写入源码/仓库/输出; `.gitignore`
   已排除 `.env` 等文件.
3. **编码**: 所有生成文件为 UTF-8; Windows GBK 控制台可先
   `$env:PYTHONIOENCODING="utf-8"`.
4. **工作日志**: `opencode_worklog.txt` 被 `.gitignore` 排除, 仅本地保留.
5. **报告产物**: `reports/` 随仓库提交.

---

## 10. 许可证

本项目仅用于 GOSIM 巡天智能体赛题相关科研与工程实践用途.
