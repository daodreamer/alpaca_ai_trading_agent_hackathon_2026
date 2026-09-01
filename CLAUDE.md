# CLAUDE.md — AlphaGate（Alpaca AI Trading Agents Hackathon, 2026-08-28 → 09-04）

## 1. Role

你是本项目的 AI 软件工程协作者。这是一个 7 天单人 hackathon 项目，采用 SDD + TDD。
先读 `specs/`，再写代码。

## 2. 与上游项目的关系

`alphagate.core` 和 `alphagate.infra` 是从 Personal Market Monitor 抽取的既有代码
（见 [adr/0001-core-reuse.md](adr/0001-core-reuse.md)）。**不要重写、不要"顺手改进"、
不要重构它们。** 它们已经 green，改动只会消耗比赛时间。

上游 CLAUDE.md 的第 14 条"不实现自动交易或下单"**在本仓库不适用**——本项目的核心
就是在 Alpaca paper 账户上下期权单。这是有意的、有范围限定的例外：仅限 paper
trading，永不接真实资金。其余架构纪律全部继承。

## 2b. 本文件的适用范围

**第 3 到第 7 条只管 `backend/`。**

`ai_quant_researcher/` 是同仓库里的另一个项目——实现
[specs/trading_strategy_architecture.md](specs/trading_strategy_architecture.md)
的股票策略研究系统。它有自己的 `pyproject.toml`、自己的 venv、自己的测试和自己的
边界测试，**不 import `alphagate` 的任何东西**。

它不持有钱（所以没有 `Decimal` 要求）、不下单（所以没有 Risk Gate）、不碰 Alpaca。
它的 LLM 边界由它自己的
[tests/test_boundaries.py](ai_quant_researcher/tests/test_boundaries.py) 强制。

两个项目回答的是不同的问题，刻意保持隔离。**不要去"统一"它们**，也不要把
AlphaGate 的 invariant 搬过去或反过来。改动 `ai_quant_researcher/` 时用它自己的
命令（见该目录的 README），不要用第 6 节的命令。

**两者之间的接口只有文件，没有别的**。现在有三个，都是单向的（研究端写，
`backend/` 读）：

| 文件 | 谁写 | 承载什么 |
| --- | --- | --- |
| `runs/target_books/*.json` | `aqr target-book` | 股票的权重向量（specs/09 D0） |
| `runs/option_books/*.json` | `aqr option-book` | 期权的**规则**，故意不含行权价（specs/07 D1） |
| `data-options-sealed/volatility_history/SPY.csv` | `aqr options-pull` | `alphagate iv-seed` 读的隐含波动率历史 |

期权 book 里没有行权价是有意的：周二收盘写下的 5480 到周三开盘就是错的，所以走的
是规则，由执行端对着实时链自己解析——这也是为什么 `backend/` 必须有自己的 delta
选腿代码而不能 import `aqr.options.chain`。

第三行是文件而不是函数调用，理由同上：`iv-seed` 用标准库 `csv` 读，把解析好的
mapping 交给 `IvHistoryStore.seed_from_vendor_history`，不 import 任何 `aqr` 的
东西。

两边都不 import 对方，`backend/tests/test_boundaries.py` 的 guard 9 双向强制这一点，
`scripts/pipeline.py` 也只用 subprocess 调两边的 CLI。

想加"共享一个常量"或"直接调对方一个函数"的时候：那正是这条线存在的原因。

## 3. Non-negotiable rules

1. `core` / `options` / `risk` 三层是 pure：只依赖标准库和彼此。由
   `backend/tests/test_boundaries.py` 强制，不靠自觉。
2. **只有 `agent/` 可以调用 LLM。** Risk Gate 里出现模型调用，Gate 就不再是 Gate。
3. **所有到达 Alpaca 的订单都经过 Risk Gate。** 没有 bypass、没有 `force=True`、
   没有 debug 开关。`execution/` 只接受 `GatedOrder`，而 `GatedOrder` 只有
   `risk.gate` 能构造。
4. 钱是 `Decimal`，端到端。Greeks 和 IV 是 `float`（它们是估计值，不是钱）。
5. 所有时间 tz-aware UTC。Gate 和 domain 不读时钟，时间永远是参数传入。
6. **不做裸卖期权。** 结构在类型层面就不可表达（specs/02 D3）。
7. Domain 运算 deterministic：同输入同配置必须同结果，包括 checks 的顺序。
8. 不允许 look-ahead bias。回测和实盘走同一条代码路径，唯一差别是时钟。
9. 不接真实资金、不做投资建议。产出是 agent 状态和理由，不是买卖建议。
10. 不泄露 API key / account id 到日志、journal、dashboard 或 demo 视频。

## 4. TDD

RED → GREEN → REFACTOR。`options/` 和 `risk/` 是正确性面，必须先写测试。
`agent/`、`interface/`、dashboard 可以务实推进——别为 dashboard 的 type stub
花比赛时间。

不得为了让测试通过而削弱 invariant。不得用 mock 掩盖真实行为。

## 5. 时间纪律（这是 hackathon，不是产品）

- 每天结束前 commit，保持 main 可运行。
- 任何超过 2 小时没有产出的方向，停下来重新评估。
- 9/3 之后只做演示、文档和 bug 修复，不加功能。
- 交易要尽早跑起来：目标 ≥ 30 笔成交，最晚 D3 上线。

## 6. 常用命令

在仓库根目录执行。CLI 的默认路径锚定在仓库根，所以不用加参数。

| 目的 | 命令 |
| --- | --- |
| 测试 | `uv run --directory backend --extra dev pytest` |
| lint | `uv run --directory backend --extra dev ruff check .` |
| 类型检查 | `uv run --directory backend --extra dev mypy` |
| 开盘前体检 | `uv run --directory backend python -m alphagate preflight` |
| 跑一个周期（默认不下单） | `uv run --directory backend python -m alphagate once` |
| 跑一整天 | `uv run --directory backend python -m alphagate run [--dry-run]` |
| 看当前状态 | `uv run --directory backend python -m alphagate status` |
| 看某天的日志 | `uv run --directory backend python -m alphagate show [-v] [--day YYYY-MM-DD]` |
| dashboard | `uv run --directory backend python -m alphagate serve` |

股票侧（执行 `ai_quant_researcher` 验证过的那一个策略，见 specs/09）：

| 目的 | 命令 |
| --- | --- |
| 只跑股票这条链 | `python scripts/pipeline.py --only equity` |
| 只演练，不下单 | `python scripts/pipeline.py --only equity --dry-run` |
| 开盘前体检（book + 账户） | `uv run --directory backend python -m alphagate equity-preflight` |
| 一次再平衡，只过 Gate 不下单 | `uv run --directory backend python -m alphagate equity-plan` |
| 一次再平衡，真下单 | `uv run --directory backend python -m alphagate equity-rebalance` |
| 跑一整天（心跳 + 一次再平衡） | `uv run --directory backend python -m alphagate equity-run [--dry-run]` |
| 看当前持仓与偏离 | `uv run --directory backend python -m alphagate equity-status` |

期权侧（执行 `ai_quant_researcher` 验证过的那一条期权规则，见 specs/07 D1、
specs/10）：

| 目的 | 命令 |
| --- | --- |
| 只跑期权这条链 | `python scripts/pipeline.py --only options` |
| 只演练，不下单 | `python scripts/pipeline.py --only options --dry-run` |
| 两条链都跑（股票先） | `python scripts/pipeline.py` |
| 补 IV 历史（`iv_rank` 的输入） | `uv run --directory backend python -m alphagate iv-seed` |

`ALPHAGATE_STRATEGY_FINGERPRINT` 和 `ALPHAGATE_OPTION_FINGERPRINT` 都必须在
`.env.local` 里钉死。这是"只执行研究端验证过的那个策略"这句话唯一可校验的地方
——fingerprint 不匹配的 book 会被 `load_target_book` / `load_option_book` 按名字
拒掉，**没有默认值是故意的**。两个 pin 是分开的：两条 sleeve 执行的是两条不同的
规则，各自对着不同的封存窗口验证过，一个 pin 管两边会让"当时跑的是哪条规则"在它们
分叉的那一刻起就无法回答——而它们已经分叉了。

`iv-seed` 没有股票侧的对应物，也不是优化。研究出来的规则入口是 `iv_rank() < 15`，
而 `iv_rank` 需要一年的隐含波动率历史，Alpaca 不签 OPRA 就不给
（见 `agent/iv_store.py`）。没播种的话规则不是"假"，是**无法判定**，agent 每个
cycle 都会站在一边——从外面看和市场安静一模一样。所以它每个交易日跑一次。

## 6b. 两条 sleeve 的资金分割

$100,000 的账户，一次性分成两份，之后不再随市值浮动：

| Sleeve | 分配 | 常量 |
| --- | --- | --- |
| 股票 | $90,000 | `equity/policy.py` 的 `EQUITY_SLEEVE_ALLOCATION` |
| 期权 | $10,000 | `risk/limits.py` 的 `OPTIONS_SLEEVE_ALLOCATION` |

两者之和必须等于账户总额，有测试强制——Alpaca 只有一个 buying power 池子，不知道
什么叫 sleeve，这个分割只在加得起来的时候才有意义。

期权侧是 $10,000 而不是更小，理由是算出来的而不是凑的：规则卖 0.16 delta 的 SPY
put、买 0.08 delta 的翼，实测一张的最大亏损是 $1,389；$5,000 的 sleeve 每笔预算
只有 $1,000，`agent/sizing.py` 会把张数向下取整到 **0**，规则永远开不出仓——而且
在日志里看起来和"市场没机会"一模一样。$10,000 下每笔预算正好 $2,000，等于研究端
跑的 sizing（$100,000 的 2%，specs/10 D8a）。

前端（dashboard 的 Live 页，Vite + React + shadcn/ui）：

| 目的 | 命令 |
| --- | --- |
| 构建进 Python 包 | `cd frontend && npm run build` |
| 开发模式 | `cd frontend && npm run dev` |
| 检查 | `cd frontend && npx eslint . && npm run typecheck` |

**`run` 不加 `--dry-run` 就是真下单。** `once` 默认不下单，`run` 默认下单——
调试用的命令不该会顺手下单，而跑一整天本来就是要交易。

## 7. Definition of Done

- spec 已存在且明确
- 正向 + 边界 + 确定性测试覆盖
- 无 look-ahead
- 错误路径有处理，且可观测
- pytest / ruff / mypy 全绿（改了 `frontend/` 还要 eslint / tsc 全绿）
