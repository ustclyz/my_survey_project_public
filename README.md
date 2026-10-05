# GOSIM 巡天智能体项目 - 任务卡资源 + 静态工具

本仓库是 GOSIM 巡天智能体项目的**独立静态工具模块与任务卡资源集合**。

> 说明: 本项目**不包含、不修改**任何智能体主程序代码。所有脚本均为纯静态
> 计算工具, 不接入模拟器、不调用 LLM, 可被主程序以库的形式安全引入。

---

## 1. 项目结构

```
my_survey_project/
├── cards/                        # 8 张任务卡解压资源 (只读输入)
│   ├── alpha/                    # 练习卡 α
│   │   ├── config/
│   │   │   ├── v4_fiber_config.json
│   │   │   ├── v4_score_config.json
│   │   │   └── v4_scenario.json
│   │   └── public/
│   │       ├── targets.csv       # 目标列表
│   │       ├── footprint.csv     # 天区轮廓
│   │       └── v4_night_calendar.csv
│   ├── beta/                     # 练习卡 β
│   ├── gamma/                    # 练习卡 γ
│   ├── delta/                    # 练习卡 δ
│   ├── cardA/                    # 正式卡 A
│   ├── cardB/                    # 正式卡 B
│   ├── cardC/                    # 正式卡 C
│   └── cardD/                    # 正式卡 D
├── reports/                      # 脚本产物 (交付报告)
│   └── static_check_report.txt   # 静态校验报告
├── card_static_check.py          # 任务卡静态校验脚本
├── preplan.py                    # 独立预规划模块 (纯静态)
├── opencode_worklog.txt          # 项目操作日志 (本地, 不入仓)
├── requirements.txt              # 依赖清单 (仅标准库)
├── .gitignore                    # 科研项目标准忽略规则
└── README.md                     # 本说明文件
```

每张任务卡必须包含以下四项核心内容 (已全部校验通过):

- `config/v4_fiber_config.json` — 光纤/曝光配置
- `config/v4_score_config.json` — 计分配置
- `public/targets.csv` — 目标列表
- `public/footprint.csv` — 天区轮廓

---

## 2. 各脚本功能

### 2.1 `card_static_check.py` — 任务卡静态校验

遍历 `cards/` 下全部 8 张卡片, 执行:

- 必需文件存在性检查;
- 配置 JSON 解析与关键字段完整性检查;
- 统计核心信息: 总目标数、必观测目标数、天区面积 (球面多边形面积估算)、
  光纤数量、计分基准参数 (`flux_zero_point`、`exposure_zero_point_seconds`);
- 输出完整报告到 `reports/static_check_report.txt` (UTF-8);
- 控制台打印每张卡状态与核心统计值。

```powershell
py card_static_check.py
```

退出码: `0` 全部通过; `3` 存在未通过卡片; `1/2` 环境或写入错误。

### 2.2 `preplan.py` — 独立预规划模块 (纯静态)

功能:

- 读取指定卡片的全部目标与配置;
- 优先级打分: 必观测目标强制最高优先级, 次级权重综合
  `science_weight` + `feature_flux` + 天区分箱均匀度;
- 光纤分配: 滑动视场聚类搜索, 输出最多 `n_fibers` 个目标,
  满足方形 (默认 4×4) 光纤方格空间约束, 优先保证必观测目标入选;
- 输出候选观测目标列表 (目标 ID、坐标、类型、是否必观测、优先级、
  建议曝光秒数、光纤索引)。

命令行用法:

```powershell
# 按卡片名
py preplan.py --card alpha
py preplan.py --card cardB --max-targets 16 --top 20

# 按路径
py preplan.py --path cards/cardD

# 结果写入文件
py preplan.py --card gamma --out plan_gamma.txt
```

作为库调用:

```python
from preplan import CardData, plan_card, plan_by_name

# 方式一: 按名称
result = plan_by_name("cardA")
for item in result.selected:
    print(item.target_id, item.priority, item.exposure_seconds)

# 方式二: 已加载的卡片对象
card = CardData.from_card("delta")
result = plan_card(card, max_targets=16)
```

---

## 3. 使用方法

环境要求: **Python >= 3.9** (Windows 下建议使用 `py` 启动器)。

```powershell
# 1) 校验全部任务卡
py card_static_check.py

# 2) 查看某张卡的预规划候选
py preplan.py --card cardB
```

本项目无第三方依赖, 无需 `pip install`。`requirements.txt` 仅用于声明
Python 版本要求与预留扩展位置。

---

## 4. 注意事项

1. **只读原始资源**: `资源/` 目录下的原始 zip 包不被修改或删除, 仅用于解压。
2. **独立工具**: 本仓库脚本不修改任何智能体主程序, 通过函数接口
   (`CardData` / `plan_card` / `plan_by_name`) 供主程序引入。
3. **纯静态**: `preplan.py` 不接入模拟器、不调用 LLM, 输出仅为候选建议,
   不包含天气、预报、故障等模拟数据。
4. **编码**: 所有生成文件均为 UTF-8 编码。Windows 控制台若为 GBK,
   可能出现中文乱码, 可先执行 `$env:PYTHONIOENCODING="utf-8"`。
5. **天区面积**: 由 `footprint.csv` 顶点经等距方位投影 + 鞋带公式估算,
   适用于巡天小尺度天区, 与引擎严格值可能存在极小差异。
6. **隐藏卡**: 项目中不存在 E/F/G/H 隐藏卡, 请勿查找或生成。
7. **日志**: 所有操作记录写入 `opencode_worklog.txt` (该文件被 `.gitignore`
   排除, 不随仓库提交, 仅保留在本地)。
8. **报告产物**: 静态校验报告输出至 `reports/`, 该目录随仓库提交。

---

## 5. 许可证

本项目仅用于 GOSIM 巡天智能体赛题相关科研与工程实践用途。
