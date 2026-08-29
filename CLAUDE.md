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
