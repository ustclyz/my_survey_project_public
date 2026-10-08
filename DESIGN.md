# survey26 巡天观测智能体 —— 架构说明与复现指南

> **最终评测版本**：revision `1b225832`（标题 `all-in-v6-maint`），对应 git tag **`final-2026-10-08`**（commit `7fade43`，分支 `exp/duty-maint`）。
> **平台成绩**：八卡 super **138659.2** = 0.2·Σ(A–D) + 0.8·Σ(A1–D1)；A–D 均值 29514.4。
>
> *English abstract.* A standard-library Python agent for the GOSIM 2026 Agentic Observer Hackathon
> (survey26). It couples a gradient-free joint planner (pointing × fibre assignment × exposure
> duration × observing program) with three narrowly-scoped LLM sessions (night plan, fault review,
> duty-log reading) and a deterministic parser for handover logs. All heavy decisions stay on the
> deterministic path; the model is used where it earns its cost. Every design choice in this
> document is backed by a platform A/B measurement, including four rejected hypotheses.

---

## 1. 任务是什么

一张「任务卡」＝ 一个完整的虚拟巡天季度：智利帕拉纳尔虚拟台站，123 个夜晚，
30 000–50 000 个光谱目标，16 根可指派光纤排成 4×4 无缝网格，视场宽 2.53°。
智能体每收到一个 `decision_request` 就下发**一个**动作：

| 动作 | 参数 | 说明 |
|---|---|---|
| `observe` | 指向(alt/az)、最多 16 根光纤各自的目标、曝光 60–3600 s、程序档位(DARK/BRIGHT/BACKUP) | 主要得分手段 |
| `wait` | 秒数或直到某个 UTC 时刻 | 等目标升起 / 等坏天气过去 |
| `report` | — | 报告仪器故障：报对 +100 且立即修复；前 2 次误报免费，之后每次 −150 |
| `finish` | — | 提前结束 |

计分（官方口径，本地 `tools/local_scorer.py` 逐条复刻）：

```
g(i,e) = min(1, flux_i · T_e · quality(i,e) / (f0·T0))        # 完成因子，上限 1
s(i,e) = w_i · g(i,e) · m(declared, actual)                    # 档位配对 ×1.20/1.12/1.06，否则 ×1.00
总分   = Σ_i max_e s(i,e)                    # 每个目标只算它最好的一次
        − 50 × 漏掉的必观测目标数
        − 均匀度扣分(最多 200)
        ± 报修得失  + 限时观测请求奖励
```

**两条约束塑造了整个架构**：

1. **每个目标只算最好的一次** —— 短曝光是永久性丢分，重复观测没有累加收益；
2. **整张卡只有 900 秒真实 CPU 预算**（`wallclock.remaining_real_cpu_seconds`）——
   常规做法「每步都问一次大模型」根本跑不完一场 123 夜的巡天。

我们线上跑的是 **A/B/C/D + A1/B1/C1/D1 八张卡**；赛后隐藏卡 E/F/G/H 用同一个 revision 各跑 3 次取平均，**最终排名只看隐藏卡**。

---

## 2. 交付物与运行契约

平台只认两个东西：`observer.project.json`（清单）和 ZIP 根目录下的 `agent.py`。

```json
{ "schema_version": "observer-project-v1", "protocol": "jsonl-v4",
  "image": "python:3.12-slim", "build": [],
  "run": ["python3", "-u", "agent.py"], "working_directory": ".",
  "environment": { "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "/workspace/.deps" } }
```

进程契约：**stdin 一行一个 JSON 消息，stdout 一行一个 JSON 回复，日志一律走 stderr**。

我们的入口 `agent.py`（34 行）只做一件与健壮性有关的事——**stdout 硬化**：把 `sys.stdout`
重定向到 stderr，只保留真实 stdout 作为协议通道，之后才 import 决策内核。这样任何第三方库、
调试 `print`、甚至内核里的异常回溯都不可能污染协议流。

---

## 3. 系统架构

```
                        stdin (JSONL)                    stdout (JSONL, 唯一协议通道)
                            │                                      ▲
   ┌────────────────────────▼──────────────────────────────────────┴─────────────┐
   │ agent.py            入口：Python≥3.9 检查 → stdout 硬化 → 交给内核            │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │ pro/agent.py        ObserverAgent：协议循环、决策主循环、预算与节奏、报修执行  │
   │   ├── 决策主循环 _respond()                                                   │
   │   ├── 时钟/预算 _clock() _pace() _decisions_left()                            │
   │   ├── 模型会话编排：夜计划 / 故障研判 / 值班日志（异步，不阻塞决策）           │
   │   └── 报修：_maybe_report() _due_report() _on_report_result()                 │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │ pro/planner.py      Planner：指向 × 光纤分配 × 时长 × 档位 的联合搜索          │
   │ pro/advisor.py      两个窄职责模型会话：今晚计划 / 当前是否像故障              │
   │ pro/llm_client.py   纯标准库 OpenAI 兼容客户端（后台线程、超时、重试、并发上限）│
   │ pro/maintenance.py  值班日志确定性解析器（工程通知 → 绝对 UTC 报修时刻）        │
   │ pro/skymath.py      球面几何 / 大气质量 / 时角 / gnomonic 投影                │
   └──────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 决策主循环（`pro/agent.py::_respond`）

每个 `decision_request` 按固定顺序走一遍，任何一环都能直接给出动作：

1. **更新现场状态**：新消息（预报 / 简报 / 值班日志 / 观测请求结果）、地震静默、上一动作结果；
2. **值班日志支路**：有新日志 → 交给解析链路（§3.4）；
3. **推进规划器**：`on_messages` / `on_requests` / `on_result` 把平台反馈变成学习信号；
4. **节奏控制** `_pace()`：按实测的单次决策成本与剩余决策数，选择搜索层级（0 最贵最细，4 最粗）；
5. **夜间边界**：夜晚开始时做一次「今晚计划」模型调用；白天直接 `wait` 到下一夜；
6. **到点报修** `_due_report()` → 若命中排程，立即 `report`；
7. **质量报修** `_maybe_report()` → 价格判据 / 绝对质量下限判据 → `report`；
8. **规划** `Planner.plan()` → `observe`；无可用目标时 `wait` 到下一个可用时段。

### 3.2 规划内核（`pro/planner.py`，927 行）

一次 `plan()` 就是一次**联合搜索**，而不是「先选目标再定曝光」的流水线：

* **候选目标池**：用高度角线性模型 `sin(alt) = A + B·cos(LST) + C·sin(LST)` 预筛（无三角函数，
  复用系数），再按「计划价值 × 粗略大气质量 × 紧迫度」取前 N 个作为代理打分；
* **视场候选**：以若干「锚目标」为中心生成指向，把邻近目标投影到 4×4 网格的切平面上分类到光纤格；
* **分配 + 时长 + 档位一起评估**：对每个候选指向、每个候选时长，
  逐格取增益最大的目标，累计 `total(T) = Σ_j gain(j,T)`；
* **时间价格**：`net = total − λ·T`，其中 `λ = LAMBDA_FRAC × rate_ema × (scarcity 缩放) × LLM 倍率`，
  `rate_ema` 是最近测到的「单位时间最佳增益率」。时间稀缺的卡（如 C 卡 scarcity 1.22）价格更高、
  曝光更短；时间宽裕的卡（如 D 卡 0.05）价格自动降到 0.12 倍。
* **精修**：择优后做局部搜索（微调指向以吃下格边缘目标）+ 一档固定时长的复评。

学习回路：未饱和命中给出天空质量估计，**饱和命中直接暴露档位倍率**，用来拟合程序档位；
每个目标维护 `cur[i]`（已知最好完成因子），因此规划器天然避免重复观测已打满的目标。

### 3.3 模型层的三条会话，职责严格分离

| 会话 | 触发 | 输入 | 输出 | 为什么这样切 |
|---|---|---|---|---|
| **夜计划** `advisor.start_night` | 每个夜晚开始 | 今晚预报 + 简报 + 自己的质量表 | 坏夜标记、需规避方位 | 只需每晚一次，允许等待 |
| **故障研判** `advisor.fault_review` | 准备付代价报修前 | 自己的质量时序 | 故障可能性、是否立刻报修 | 只在关键决策点调用 |
| **值班日志** `duty`（多轮有界历史） | 有新值班日志 | 日志原文 + 本地时区 + 运行状态快照 | 报修时刻表 / 取消 / 测试窗口 / 地形遮挡 / 坏夜 | 需要读懂密文与长上下文，见 §3.4 |

三条会话**不共享历史**：日志会话只做「日志 → 事实」，态势判断由确定性规则和故障研判完成，
避免把「读日志」和「做决策」混在一个提示词里互相污染。

### 3.4 值班日志链路：确定性解析器 + LLM，同一条通道

硬卡（A1–D1）会收到中/日/英文混写、带凯撒密文/摩斯电码伪装的值班日志，
核心事实是「**导星相机在某时刻坏掉，从那以后所有曝光都是废片，必须在那个时刻报修**」。

```
值班日志原文 ─┬─▶ pro/maintenance.py  MaintenanceSchedule（确定性）
             │      · 只认显式工程通知；传闻/未确认/假如一律丢弃
             │      · 凯撒密文只在解出 "guider camera" 整句时采信
             │      · 时区(UTC / UTC+9 东京 / 站点本地)显式换算，缺失时不猜
             │      · 支持「推迟 N 小时」「以最后日期为准」「取消」「测试结束再等 N 小时」
             └─▶ pro/agent.py  duty LLM 会话（多轮、有界历史、可大 token 预算）
                    · 输出结构化事实：report_utc / cancelled_utc / test_windows /
                      no_report_utc / bad_nights / terrain / avoid_directions / 倍率旋钮
                        │
                        ▼
             同一条合并通道：duty_times ∪ duty_cancelled ∪ duty_test_windows
                        │
                        ▼
     _due_report(): 到点 → report        duty_bad_nights → 规划器避让
     duty_test_windows → 那段时间只 wait（既不观测也不误报）
     terrain / avoid / prefer / duration_scale / lambda_scale → 直接改规划器参数
```

设计要点：**确定性解析器是主路径，LLM 是增益**。两者写同一个状态，模型可以补充、纠正或取消
解析器的结论；解析器在模型失败/超时时独立兜底。整条链路静默降级——模型不可用时行为与
「只有解析器」完全一致（有单元测试覆盖，见 §6.3）。

### 3.5 故障报修（不依赖值班日志的那一半）

硬卡上不是所有故障都会被预告，所以还有一条纯统计的判据：定义
`E = 质量水平 / 档位水平`（天气同时压低两者，故障只压低前者）。当 `E` 持续偏低时按
「免费探测 → 付费报修」分级出手，并带三个安全阀：误报封锁（报错后一段时间不再报）、
全场雨/厚云不判故障、连续 `report` 不超过 3 次（平台对连报有上限）。

### 3.6 预算与节奏

* 模型调用全部**异步**（后台线程 + `submit/collect`），等待时间不计入 CPU 预算；
* `_pace()` 用实测的每决策 CPU/墙钟成本反推「还做得起多细的搜索」，预算见底时自动降级；
* 值班日志的 `max_tokens` 走环境变量 `PRO_DUTY_MAX_TOKENS`（默认 8192，线上设 32768）——
  平台模型在这类提示词下推理长度 12k–20k token，预算不足时**推理会吃掉全部输出**，
  `content` 为空，看起来像「模型读不懂日志」，实际是输出被截断；
* 入口处一次 `stdout` 硬化 + 所有异常兜底成 `wait` 一个时段，绝不因内部异常丢整场。

---

## 4. 关键设计决策与实测依据

| # | 决策 | 为什么 | 平台实测 |
|---|---|---|---|
| 1 | 线上内核改用官方 `python-pro` 参考实现（保留我们的协议层与硬化入口） | 自研静态内核受限于本地合成天气，泛化差 | A–D 从 ~12 221 → ~29 700 |
| 2 | 把值班日志做成「确定性解析器 + LLM」同通道 | LLM 单独读密文日志不稳，正则单独读不到变体 | B1 28 508 → 38 814，D1 56 349 → 57 886，C1 18 778 → 21 515 |
| 3 | 同一解析器移植到另一条代码线（scale-fault 基底） | 验证增益来自「解析器」而不是「某条线」 | 该线 D1 56 349 → 57 576，B1 28 508 → 37 916 ✔ 结论成立 |
| 4 | 值班日志 `max_tokens` 提到 32768 | 推理长度 12k–20k，8192 会 100% 空输出 | 日志解析从 28 次全空 → 17 轮全部产出结构化事实 |
| 5 | 时间价格随「时间稀缺度」自动缩放 | C 卡真正缺时间、D 卡时间几乎免费 | A/B/C/D 的 λ 自动落在 0.60/0.35/0.60/0.12 |
| 6 | 模型只用在三处窄职责 | 900 秒预算下每步都问模型跑不完 | 每张卡实耗 CPU 386–641 / 900 s，全部 `survey_complete` |

### 4.1 被实测否决的方案（同样重要）

| 假设 | 预期 | 平台结果 | 处置 |
|---|---|---|---|
| 时间价格 `LAMBDA_FRAC` 0.60 → 0.30（更长曝光） | 本地仿真 +8~12% | A–D −1% | 回退到 0.60 |
| 把曝光拉长到「刚好让本视场全部饱和」（`SAT_STRETCH`） | 补齐 25~32% 的饱和缺口 | A −1750 / B −2800 / C −1800（剂量越大越差） | 放弃，`exp/sat` 分支留档 |
| 给 LLM 更大自由度做值班日志→决策（rescue v4） | 补上确定性解析器读不到的变体 | 逐卡 ≤ rescue v3，D1 直接 cancelled | 不作为 final |
| 只调档位声明阈值 `PROGRAM_BAND_SHIFT` | 补回 12~34% 的档位错配 | 错配实为 5.8~11.8%，全修也只 +0.5~2.6% | 标记为无效常量，不启用 |

### 4.2 已定位但来不及进最终版本的缺陷（留给后续）

值班日志的 LLM 会话把模型的回答当作**「全量状态」直接替换**排程表。模型的回答在轮次间并不
自洽（平台日志里同一张卡的排程条数在 3→53 之间反复跳），于是**已经排上但还没到点的时刻会被
下一轮挤掉**。直接证据：all-in-v5 在 A1 卡上 17 轮解析全部成功（每轮抽到 3~53 个时刻，
还带地形遮挡/避让方位/坏夜），却 **0 次触发到点报修**（同期 rescue v3 触发 92 次）。

修复很小：排程只增不减，只有「已报修」或「模型明确取消」才移除。该修复在分支
**`exp/duty-accum`**（commit `27a795b`），平台实测 B1 38 814 → **40 083**、C1 21 515 → **21 940**，
D1 因评测窗口耗尽未出分，因此**没有进入最终版本**——在截止前无法用完整八卡数据背书，
按本文档 §4 的方法论只能放弃。`tools/check_duty_maintenance.py` 用一条断言把这个回归固定下来。

**方法论**：所有改动先在本地闭环仿真里筛，再上平台 A/B；本地合成天气会系统性高估，
因此**只有平台实测才作为接受标准**。四条被否决的假设都没进最终版本。

---

## 5. 平台成绩（八卡，最终版本 `1b225832`）

| 卡 | A | B | C | D | A1 | B1 | C1 | D1 |
|---|---|---|---|---|---|---|---|---|
| 得分 | 23 597.1 | 37 224.3 | 25 400.8 | 31 835.3 | 25 594.8 | 38 813.9 | 21 515.1 | 57 885.8 |

* A–D 均值 **29 514.4**（线上榜口径），八卡 super **138 659.2**
* 参照：官方 `pro` 参考实现 baseline A–D 均值 29 710.0，`basic` baseline 22 810.1
* 该分数来自线上阶段评测批次 `335f1b31`（revision `1b225832`）。平台的批间差用另一个
  revision 的双批次实测约 **±0.5%**（同一版本两次评测 A–D 均值 29 618 / 29 473），
  因此小于这个量级的版本间差异不应作为结论。

---

## 6. 复现步骤

### 6.1 环境

* **Python 3.9+**（平台镜像 `python:3.12-slim`）。**运行智能体不需要任何第三方包**，
  全部为标准库；`requirements.txt` 只服务于本地实验脚本。
* 无 GPU、无系统依赖。

### 6.2 取代码（离线单文件恢复）

仓库镜像已经打成 git bundle，含全部分支、tag 与 worktree HEAD：

```bash
git clone survey26-repo-2026-10-08.bundle my_survey_project_public
cd my_survey_project_public
git tag                        # 应看到 final-2026-10-08
git checkout final-2026-10-08  # 取到最终评测版本（1b225832 对应代码）
```

也可以直接从远程仓库取：`git clone https://github.com/ustclyz/my_survey_project_public.git`
然后 `git checkout final-2026-10-08`。

### 6.3 静态检查与单元测试

```bash
python -m compileall -q agent.py pro tools tests   # 语法/导入自检 (无输出即通过)
python -m pytest tests -q                          # 87 项回归, 约 40 s
python tools/check_duty_maintenance.py             # 值班日志链路 14 项自检 (打印 PASS/FAIL)
```

`tools/check_duty_maintenance.py` 覆盖：显式工程通知命中、重复 ingest 幂等、
传闻/未确认/假如丢弃、平场灯与镜盖测试识别为「窗口」而非故障、取消生效、
时区缺失时不猜测、LLM 结果与解析器合并、LLM 能否决解析器，以及
**「排程时刻只增不减」**（模型下一轮的全量状态漏掉旧时刻时不得把它挤掉——
本版本尚未包含该修复，脚本会打印 `SKIP`；修复在分支 `exp/duty-accum`，见 §4.1）。

### 6.4 本地仿真（不联网、不用平台密钥）

```bash
# (a) 官方计分公式 + 合成天气的闭环模拟（走仓库里的静态内核，用于规划算法回归）
python tools/local_scorer.py --card cardA
python tools/local_scorer.py --card cardD --seed 7

# (b) 用真实决策循环跑线上内核 pro，回答「900 秒 CPU 预算够不够、覆盖多少目标」
python tools/pro_sim.py --card cardA --max-decisions 4000 --progress 500

# (c) 只算预算，不评分（压缩 CPU 成本时用）
python tools/budget_sim.py --card cardB
```

### 6.5 打包与平台评测

```bash
# 打包（只含 agent.py + observer.project.json + pro/*.py）
python tools/pack_min.py . ../my-agent.zip

# 平台 CLI（自带，无需安装第三方包）
python survey26.py project upload ../my-agent.zip --title "my agent"
python survey26.py project wait <rev>          # 等公开测试
python survey26.py project confirm <rev> --yes # 复核执行设置与适配文件
python survey26.py eval start <rev> --yes      # 消耗 1 次当日评测额度
python survey26.py results show <batch>        # 逐卡得分
python survey26.py results download <run>      # 下载该卡完整结果（含 score_report.json）
python survey26.py final set <rev> --yes       # 选定用于隐藏卡 E/F/G/H 的版本
```

### 6.6 平台侧需要设置的环境变量

| 变量 | 值 | 作用 |
|---|---|---|
| `SOAD_API_KEY` / `SOAD_BASE_URL` / `SOAD_MODEL` | 团队自己的模型服务 | 线上评测使用的模型（我们用 OpenAI 兼容端点 + deepseek 系列） |
| `PRO_DUTY_MAX_TOKENS` | `32768` | 值班日志会话的输出预算；`pro/agent.py` 的 `_env()` 统一读 `PRO_` 前缀 |
| `PRO_LAMBDA_FRAC` | `0.60` | 时间价格系数（默认值即最终值，列出便于对照实验） |

### 6.7 能复现什么、不能复现什么

* **可以**：代码、打包、协议行为、单元测试、练习卡 α–δ 的完整闭环、CPU/时间预算结论、
  所有「被否决方案」的对照实验配置。
* **不能**：竞赛卡 A–D / A1–D1 / 隐藏卡 E–H 的**分数**——它们的天气、预报与故障事件是平台私有输入，
  只在评测容器里可见。因此本地仿真只用于**筛掉坏想法**，接受标准始终是平台 A/B。
* 本仓库 `reports/` 与根目录 `HANDOFF.md` / `LIVE_STATE.md` / `FINAL_PICK.md` 保留了完整的
  实验日志（含失败方案与批次号），可逐条对账。

---

## 7. 仓库结构

```
agent.py                 入口：stdout 硬化 → 交给 pro 内核（34 行）
observer.project.json    平台清单（协议 jsonl-v4，镜像 python:3.12-slim）
pro/                     ★ 线上运行的全部代码
  agent.py               协议循环 + 决策主循环 + 预算/节奏 + 报修 + 三条模型会话编排
  planner.py             指向 × 光纤 × 时长 × 档位的联合搜索与学习
  advisor.py             夜计划 / 故障研判两个窄职责会话
  llm_client.py          标准库 OpenAI 兼容客户端（异步、超时、重试、并发上限）
  maintenance.py         值班日志确定性解析器（工程通知 → 绝对 UTC 时刻）
  skymath.py             球面几何 / 大气质量 / 时角 / gnomonic 投影
tools/                   本地仿真与评测工具
  local_scorer.py        官方计分公式 + 合成天气的闭环模拟；pro_sim.py 预算仿真
  duty_replay.py         用历史值班日志回放解析链路；duty_log_report.py 汇总报修得失
  analyze_observations.py / summarize_results.py / card_stats.py  结果分析
  pack_min.py            生成最小可提交 ZIP
tests/                   规划器(978 行)、协议、入口、值班日志回归
cards/                   公开输入：cardA–D 与练习卡 α/β/γ/δ
planner.py / preplan.py / llm.py / protocol.py / models.py   早期静态内核（本地实验保留，线上不用）
reports/                 实验记录：rescue v1–v3、值班日志交付修复、静态检查报告
```

---

## 8. 一句话总结

这套智能体的核心判断是：**在 900 秒 CPU 预算和「只算最好一次」的计分下，胜负取决于
「把确定性算法用在能算清楚的地方，把模型用在只有模型能读懂的地方」，并且每一条这样的判断
都必须由平台 A/B 数据背书**。最终版本的八卡成绩、四条被否决的假设和全部实验批次号都留在
仓库里，可直接复核。
