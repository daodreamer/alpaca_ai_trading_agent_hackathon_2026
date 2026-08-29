# AI Quant Researcher

> AI 驱动的量化策略研究、回测、验证与自动进化平台
>
> 目标：利用 LLM + Machine Learning + Quantitative Research 自动发现具有统计优势、能够跨市场 Regime 泛化的交易策略。
>
> 非 HFT 系统，主要面向分钟级、小时级、日级以及数日持仓策略。

## 1. 项目目标

### 1.1 核心目标

构建一个自动化 Quant Research Lab：

```
Market Data / News
        ↓
Feature Engineering
        ↓
LLM Strategy Research
        ↓
Strategy Generation
        ↓
Backtesting
        ↓
Statistical Validation
        ↓
Walk-Forward Validation
        ↓
Paper Trading
        ↓
Live Trading
        ↓
Performance Feedback
        ↓
Strategy Research
```

系统的核心不是让 LLM 直接预测股票价格，而是：

1. 发现市场中的潜在规律
2. 生成交易假设
3. 将假设转化为可执行策略
4. 自动进行历史回测
5. 自动检测过拟合
6. 进行 Out-of-Sample 验证
7. 在 Paper Trading 中验证
8. 根据真实交易结果继续优化

## 2. 核心设计原则

### 2.1 LLM 不直接负责交易

LLM 不直接执行：

- BUY
- SELL
- Position sizing
- Stop loss execution
- Order execution

LLM 主要负责：

- Strategy research
- Hypothesis generation
- Feature discovery
- Event interpretation
- Market regime reasoning
- Strategy explanation
- Backtest analysis
- Research iteration

最终交易决策必须由确定性的 Strategy Engine / ML Model / Risk Engine 执行。

## 3. 系统总体架构

```
                         ┌──────────────────────┐
                         │      Data Sources    │
                         │                      │
                         │ Market Data          │
                         │ News                 │
                         │ Fundamentals         │
                         │ Macro                │
                         │ Alternative Data     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Data Pipeline     │
                         │                      │
                         │ Ingestion             │
                         │ Cleaning              │
                         │ Normalization         │
                         │ Storage               │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    ▼                                ▼
          ┌──────────────────┐              ┌──────────────────┐
          │ Feature Engine   │              │ News/Event Engine│
          │                  │              │                  │
          │ Technical        │              │ LLM extraction   │
          │ Volatility       │              │ Event type       │
          │ Volume           │              │ Sentiment        │
          │ Market Structure │              │ Surprise         │
          │ Macro            │              │ Impact score     │
          └────────┬─────────┘              └────────┬─────────┘
                   │                                 │
                   └───────────────┬─────────────────┘
                                   ▼
                        ┌──────────────────────┐
                        │  Market Regime Engine │
                        │                      │
                        │ Bull / Bear          │
                        │ High / Low Vol       │
                        │ Trending / Ranging   │
                        │ Risk On / Risk Off   │
                        └──────────┬───────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │  LLM Research Agent  │
                        │                      │
                        │ Hypothesis generation │
                        │ Factor discovery     │
                        │ Strategy generation  │
                        └──────────┬───────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │ Strategy Compiler    │
                        │                      │
                        │ DSL / Python         │
                        │ Validation            │
                        └──────────┬───────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │   Backtest Engine    │
                        │                      │
                        │ Historical testing   │
                        │ Slippage             │
                        │ Fees                 │
                        │ Position sizing      │
                        └──────────┬───────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │ Validation Engine    │
                        │                      │
                        │ OOS                 │
                        │ Walk Forward         │
                        │ Monte Carlo          │
                        │ Robustness            │
                        │ Overfitting           │
                        └──────────┬───────────┘
                                   │
                         ┌─────────┴─────────┐
                         ▼                   ▼
                    REJECT                ACCEPT
                         │                   │
                         │                   ▼
                         │        ┌──────────────────┐
                         │        │ Strategy Registry│
                         │        └────────┬─────────┘
                         │                 │
                         │                 ▼
                         │        ┌──────────────────┐
                         │        │ Paper Trading    │
                         │        └────────┬─────────┘
                         │                 │
                         │                 ▼
                         │        ┌──────────────────┐
                         │        │ Live Trading     │
                         │        └────────┬─────────┘
                         │                 │
                         └─────────────────┤
                                           ▼
                                ┌──────────────────────┐
                                │ Performance Feedback │
                                └──────────┬───────────┘
                                           │
                                           ▼
                                  LLM Research Agent
```

## 4. 系统模块

### 4.1 Data Layer

负责获取和存储所有研究数据。

#### Market Data

- OHLCV
- Tick / Trade
- Quote
- VWAP
- Volume
- Market Cap
- Float
- Short Interest
- Options

主要时间周期：

- 1m
- 5m
- 15m
- 30m
- 1h
- 4h
- 1D
- 1W

MVP：

- 5m
- 15m
- 1h
- 1D

#### News Data

- News headline
- News body
- Publication time
- Source
- Ticker
- Sector
- Author
- URL

必须保存：

- `event_time`
- `ingestion_time`

防止 Look-Ahead Bias。

#### Fundamental Data

- Revenue
- EPS
- EPS growth
- Revenue growth
- PE
- Forward PE
- Debt
- Cash
- Free Cash Flow
- Guidance
- Analyst estimates

#### Macro Data

- VIX
- Interest rate
- 10Y yield
- DXY
- SPY
- QQQ
- IWM
- Sector ETFs
- Market breadth

### 4.2 Data Storage

推荐：

```
Raw Data
    ↓
Parquet
    ↓
DuckDB
```

用于研究。

Production：`PostgreSQL`

用于：

- Strategy metadata
- Signals
- Orders
- Positions
- Performance
- Research experiments

Redis 用于：

- Cache
- Real-time state
- Event queue
- Signal state

## 5. Feature Engine

Feature Engine 不使用 LLM 计算基础指标。

这些应该使用确定性代码计算。

### 5.1 Technical Features

- SMA
- EMA
- RSI
- MACD
- ADX
- ATR
- Bollinger Bands
- VWAP
- Stochastic
- ROC
- Momentum

### 5.2 Price Structure

- Swing High
- Swing Low
- Higher High
- Higher Low
- Lower High
- Lower Low
- BOS
- CHOCH
- Support
- Resistance
- Distance to support
- Distance to resistance

### 5.3 Volatility

- ATR
- Realized Volatility
- Historical Volatility
- IV
- VIX
- Volatility percentile

### 5.4 Volume

```
Volume
Relative Volume

RVOL =
Current Volume / Average Volume

Volume acceleration

Volume percentile
```

### 5.5 Market Context

- SPY return
- QQQ return
- IWM return
- Sector return
- Market breadth
- VIX regime
- Interest rate regime

## 6. News / Event Engine

这是本项目非常重要的模块。

目标：

> 将非结构化新闻转换成结构化 Market Event。

### 6.1 LLM News Extraction

输入：

- Headline
- Article
- Company
- Time

输出：

```json
{
  "ticker": "MRNA",
  "event_type": "CLINICAL_TRIAL_SUCCESS",
  "phase": "PHASE_III",
  "indication": "MELANOMA",
  "surprise": 0.95,
  "materiality": 0.94,
  "sentiment": 0.97,
  "confidence": 0.92
}
```

### 6.2 Event Types

- FDA_APPROVAL
- FDA_REJECTION
- PHASE_1_SUCCESS
- PHASE_2_SUCCESS
- PHASE_3_SUCCESS
- CLINICAL_TRIAL_FAILURE
- EARNINGS_BEAT
- EARNINGS_MISS
- GUIDANCE_RAISE
- GUIDANCE_CUT
- M_AND_A
- PARTNERSHIP
- CONTRACT_WIN
- CONTRACT_LOSS
- PRODUCT_LAUNCH
- CEO_CHANGE
- BUYBACK
- SEC_INVESTIGATION
- BANKRUPTCY

### 6.3 Event Score

定义：

```
Event Score =
Materiality
× Surprise
× Confidence
× Market Relevance
```

例如：

```
FDA Approval
Materiality = 0.95
Surprise = 0.90
Confidence = 0.98
Relevance = 0.95

Event Score = 0.795
```

## 7. Market Regime Engine

目标：

> 判断当前市场环境，而不是预测具体价格。

### Regime

- TREND_BULL
- TREND_BEAR
- RANGE_LOW_VOL
- RANGE_HIGH_VOL
- HIGH_VOL_BULL
- HIGH_VOL_BEAR
- RISK_ON
- RISK_OFF

### Features

```
SPY > EMA200
QQQ > EMA200

ADX
ATR percentile

VIX

Market breadth

Sector momentum

Momentum dispersion
```

## 8. LLM Research Agent

这是整个系统的“大脑”。

但是 LLM 不直接交易。

### 8.1 Research Agent 的任务

1. 分析历史实验
2. 发现潜在规律
3. 提出 Hypothesis
4. 设计 Factor
5. 生成 Strategy
6. 分析失败原因
7. 修改 Strategy
8. 生成新的 Experiment

### 8.2 Strategy Hypothesis

例如：

```
Hypothesis:

当市场处于 Bull Trend 时：

SPY > EMA200
ADX > 20

同时：

Price pullback 到 EMA20
RSI > 40
RVOL > 1.2

那么：

未来 3 个交易日上涨概率
可能显著高于 baseline。
```

### 8.3 Strategy DSL

不要让 LLM 随便生成 Python。

推荐设计自己的 DSL。

例如：

```yaml
strategy:
  name: ema_pullback_v1

  universe:
    type: stocks
    market: US

  regime:
    trend: bullish

  entry:
    all:
      - close > ema(200)
      - adx(14) > 20
      - close <= ema(20) * 1.01
      - rsi(14) > 40
      - rvol(20) > 1.2

  exit:
    stop_loss:
      type: atr
      multiplier: 2

    take_profit:
      type: risk_reward
      ratio: 2

    max_holding:
      bars: 20
```

这样可以：

```
LLM
 ↓
DSL
 ↓
Validator
 ↓
Backtester
```

避免 LLM 生成恶意或错误代码。

## 9. Strategy Generator

Strategy Generator 将 LLM 的 Hypothesis 转换成标准 Strategy。

### Strategy 必须包含

- Universe
- Entry
- Exit
- Stop Loss
- Take Profit
- Position Sizing
- Holding Period
- Maximum Positions
- Market Regime
- Transaction Cost

## 10. Backtest Engine

Backtester 是系统核心。

必须模拟真实交易。

### 必须考虑

- Commission
- Spread
- Slippage
- Liquidity
- Partial fills
- Market impact
- Position limits
- Trading hours
- Gap
- Overnight risk

### 10.1 Backtest Metrics

- Total Return
- CAGR
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown
- Calmar Ratio
- Win Rate
- Profit Factor
- Expectancy
- Average Win
- Average Loss
- Number of Trades
- Average Holding Period
- Exposure
- Turnover

## 11. 防止 Look-Ahead Bias

这是系统最高优先级之一。

错误的做法：

```
在 2026-01-01 使用 2026-01-02 的数据
```

必须保证：

```
signal_time
<
available_time
```

所有数据必须记录：

- `event_time`
- `available_time`

## 12. Train / Validation / Test

推荐：

```
Historical Data

├── Train       60%
├── Validation  20%
└── Test        20%
```

关键规则：

> LLM 不允许看到 Test 数据。

## 13. Walk Forward Validation

固定 Train/Test 不够。

使用：

```
Train
2020 ───────── 2022

Test
2023

↓

Train
2020 ───────── 2023

Test
2024

↓

Train
2020 ───────── 2024

Test
2025
```

最终：`Average OOS Performance`

## 14. Robustness Testing

策略必须通过多个测试。

### 14.1 Parameter Perturbation

例如：

```
EMA = 20
```

测试：

- 15
- 18
- 20
- 22
- 25

如果：

```
20 → Sharpe 2.1
19 → Sharpe 0.4
21 → Sharpe 0.3
```

说明严重过拟合。

### 14.2 Market Test

测试：

- Bull Market
- Bear Market
- Sideways Market
- High Volatility
- Low Volatility

### 14.3 Asset Test

不能只测试：`SPY`

应该：

- SPY
- QQQ
- IWM
- DIA
- Technology
- Healthcare
- Financial
- Energy
- Consumer

### 14.4 Monte Carlo

对交易序列进行：

- Random shuffle
- Bootstrap
- Trade resampling

估计：

- Expected Drawdown
- Worst Case Drawdown
- Probability of Ruin

## 15. Strategy Evaluator

Evaluator 给每个策略一个综合评分。

例如：

```
Strategy Score

30% OOS Sharpe
20% Profit Factor
15% Max Drawdown
15% Stability
10% Cross Asset
10% Regime Robustness
```

### Example

```
Strategy: Momentum_v12

OOS Sharpe:          1.63
Profit Factor:       1.72
Max Drawdown:       -14.2%
Regime Robustness:   82%
Asset Robustness:    76%
Parameter Stability: 91%

Score: 84/100

STATUS: ACCEPT
```

## 16. Overfitting Detector

必须检测：

- Parameter count
- Number of experiments
- Number of backtests
- Sharpe inflation
- Train/OOS gap
- Performance concentration
- Asset concentration
- Time concentration

### 16.1 Experiment Database

每一次实验都必须保存。

- `experiment_id`
- `strategy_id`
- `hypothesis`
- `features`
- `parameters`
- `train_period`
- `validation_period`
- `test_period`
- `metrics`
- `timestamp`
- LLM_model
- LLM_prompt_hash
- `code_hash`
- `dataset_version`

这样才能知道：

> LLM 到底尝试过什么。

## 17. Strategy Registry

只有通过验证的策略才能进入 Registry。

```
Strategy Registry

├── ACTIVE
├── PAPER
├── LIVE
├── DEGRADED
├── RETIRED
└── REJECTED
```

## 18. Paper Trading

通过：

```
Real Market Data
        ↓
Strategy
        ↓
Signal
        ↓
Virtual Order
        ↓
Virtual Position
```

持续至少：`30 - 90 days`

根据策略频率决定。

## 19. Live Trading

Live Trading 必须和 Research Environment 隔离。

```
Research
    │
    X
    │
Live Trading
```

LLM 不应该拥有：`Direct Broker API`

推荐：

```
Strategy
 ↓
Signal
 ↓
Risk Engine
 ↓
Order Manager
 ↓
Broker
```

## 20. Risk Engine

Risk Engine 必须是 deterministic code。

### Risk Rules

- Max risk per trade
- Max daily loss
- Max portfolio drawdown
- Max position size
- Max sector exposure
- Max correlated exposure
- Max leverage
- Max number of positions

例如：

```
Risk per trade = 0.5%

Maximum portfolio drawdown = 15%

Maximum single position = 10%

Maximum sector exposure = 30%
```

## 21. Position Sizing

推荐支持：

- Fixed Fractional
- ATR Position Sizing
- Volatility Scaling
- Kelly Fraction
- Risk Parity

例如：

```
Risk = 0.5% portfolio

Stop distance = $5

Portfolio = $100,000

Risk amount = $500

Shares = 500 / 5

       = 100 shares
```

## 22. News-driven Strategy

本项目的重要特色。

### Pipeline

```
News
 ↓
LLM Event Extraction
 ↓
Event Classification
 ↓
Event Score
 ↓
Market Reaction
 ↓
Technical Confirmation
 ↓
ML Probability
 ↓
Entry
```

### 22.1 Example

```
Headline:

Company X Phase III trial
successfully meets primary endpoint.
```

LLM：

```
Event Type:
PHASE_III_SUCCESS

Materiality:
0.94

Surprise:
0.91

Confidence:
0.98
```

Market：

```
Premarket:
+42%

Volume:
15x average

RVOL:
18.4

Float:
low

Gap:
+35%
```

系统：

```
Event Score = 0.84

Market Reaction Score = 0.91

Technical Score = 0.87

Final Score = 0.89
```

进入策略模型。

## 23. ML Prediction Layer

LLM 不负责最终数值预测。

使用：

- LightGBM
- XGBoost
- CatBoost
- Logistic Regression
- Random Forest

### Input

- Technical Features
- Volume Features
- Volatility
- Market Regime
- News Event Score
- Market Reaction
- Sector Momentum
- Relative Strength

### Output

例如：

```
P(return > 3% within 1 day)
P(return > 5% within 3 days)
P(return < -3% within 1 day)
```

## 24. Signal Engine

最终信号：

```
LLM Event Score
        +
Technical Signal
        +
ML Probability
        +
Market Regime
        +
Risk Constraints
        ↓
Signal Engine
```

输出：

```json
{
  "ticker": "XYZ",
  "direction": "LONG",
  "confidence": 0.82,
  "entry": 105.2,
  "stop": 99.8,
  "target": 116.0,
  "position_size": 0.03
}
```

## 25. Strategy Evolution

这是系统的核心创新之一。

每次实验：

```
Strategy
 ↓
Backtest
 ↓
Evaluation
 ↓
Critic
 ↓
Improvement
 ↓
New Strategy
```

例如：

```
Momentum_v1

Sharpe = 0.82

        ↓

Momentum_v2

加入 Market Regime

Sharpe = 1.21

        ↓

Momentum_v3

加入 RVOL

Sharpe = 1.48

        ↓

Momentum_v4

增加 volatility filter

Sharpe = 1.52
```

但是：

> 只有 OOS performance 提升才允许晋级。

## 26. Multi-Agent Architecture

推荐最终使用：

```
                    ┌───────────────┐
                    │ Research Lead │
                    └───────┬───────┘
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
   Strategy Agent     Factor Agent      News Agent
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
                    ┌──────────────┐
                    │ Backtest     │
                    │ Agent        │
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │ Critic Agent │
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │ Risk Agent   │
                    └──────────────┘
```

## 27. LLM Agent Roles

### Research Lead

负责：

- 研究方向
- 实验规划
- 策略选择

### Strategy Agent

负责：`Trading Strategy`

### Factor Agent

负责：

- Factor discovery
- Feature engineering

### News Agent

负责：

- Event extraction
- Event classification

### Backtest Agent

负责：`Backtest analysis`

### Critic Agent

负责：

- Overfitting
- Bias
- Robustness

### Risk Agent

负责：`Risk assessment`

## 28. LLM Memory

系统必须保存：

- Research History
- Hypothesis
- Experiment
- Result
- Failure
- Successful Strategy
- Rejected Strategy

LLM 下一次研究时读取：`Previous experiments`

避免重复探索。

## 29. Research Knowledge Graph

可以建立：

```
Strategy
   │
   ├── uses → Feature
   │
   ├── works_in → Regime
   │
   ├── trades → Asset
   │
   ├── triggered_by → Event
   │
   └── failed_in → Regime
```

例如：

```
Momentum Strategy

works_in:
    Bull
    High Momentum

fails_in:
    Sideways
    Low Volume

best_assets:
    Technology
    Semiconductors
```

## 30. API Architecture

推荐：`FastAPI`

API：

- `GET /market/{ticker}`
- `GET /features/{ticker}`
- `GET /strategies`
- `GET /strategies/{id}`
- `POST /research/experiment`
- `POST /backtest`
- `GET /backtest/{id}`
- `GET /signals`
- `GET /positions`
- `GET /performance`

## 31. Task Queue

研究任务不应该同步执行。

使用：

```
Redis
+
Celery / Dramatiq
```

例如：

```
POST /research/experiment

        ↓

Queue

        ↓

Research Worker

        ↓

Backtest Worker

        ↓

Validation Worker

        ↓

Result
```

## 32. Scheduler

使用：

- Cron
- Airflow
- Prefect

任务：

- Daily Data Update
- Daily Feature Calculation
- Daily Strategy Scan
- Weekly Research
- Weekly Strategy Evaluation
- Monthly Strategy Retirement

## 33. 推荐技术栈

### Backend

- Python
- FastAPI
- Pydantic

### Data

- Polars
- Pandas
- DuckDB
- Parquet
- PostgreSQL
- Redis

### ML

- LightGBM
- XGBoost
- scikit-learn
- PyTorch

### LLM

支持：

- OpenAI
- Anthropic
- Google
- Local LLM

LLM 应通过统一接口调用。

### Backtesting

MVP：`VectorBT`

或者：`Custom Event-driven Backtester`

### Frontend

- React
- TypeScript
- Next.js
- TradingView Lightweight Charts

Dashboard：

- Market Monitor
- Strategy Monitor
- Backtest
- Signals
- Positions
- Research
- Experiments
- Performance

## 34. Database Schema

### strategies

- `id`
- `name`
- `version`
- `status`
- `strategy_type`
- `created_at`
- `updated_at`
- `code_hash`
- `config_hash`

### experiments

- `id`
- `strategy_id`
- `hypothesis`
- `prompt_hash`
- `dataset_version`
- `train_start`
- `train_end`
- `validation_start`
- `validation_end`
- `test_start`
- `test_end`
- `metrics`
- `status`
- `created_at`

### signals

- `id`
- `ticker`
- `strategy_id`
- `timestamp`
- `direction`
- `confidence`
- `entry`
- `stop`
- `target`
- `status`

### trades

- `id`
- `signal_id`
- `ticker`
- `side`
- `entry_time`
- `entry_price`
- `exit_time`
- `exit_price`
- `quantity`
- `pnl`
- `fees`
- `slippage`

### news_events

- `id`
- `ticker`
- `event_time`
- `available_time`
- `event_type`
- `materiality`
- `surprise`
- `confidence`
- `sentiment`
- `raw_text`

## 35. Observability

必须记录：

- LLM calls
- Token usage
- Latency
- Errors
- Backtest duration
- Strategy generation
- Model version
- Dataset version
- Trading decisions

使用：

- OpenTelemetry
- Prometheus
- Grafana

## 36. Security

LLM：

- NO direct broker access
- NO direct database write
- NO arbitrary shell access

LLM 只能：

- Read Research Data
- Generate DSL
- Generate Hypothesis
- Request Backtest
- Read Results

所有执行：`Sandbox`

## 37. MVP

第一阶段不要做全部功能。

### MVP v0.1

目标：

> 自动发现技术分析策略。

只支持：

- US Stocks
- Daily OHLCV
- Technical Indicators
- LLM Strategy Generator
- Backtester
- Walk Forward
- Strategy Ranking

Pipeline：

```
OHLCV
 ↓
Features
 ↓
LLM
 ↓
Strategy DSL
 ↓
Backtest
 ↓
Validation
 ↓
Ranking
```

## 38. MVP v0.2

增加：

- 5m / 15m data
- Volume
- Volatility
- Market Regime
- ML Model
- Paper Trading

## 39. MVP v0.3

增加：

- News
- LLM Event Extraction
- Event Strategy
- News-driven ML
- Real-time Signal Engine

## 40. MVP v0.4

增加：

- Multi-Agent
- Strategy Evolution
- Knowledge Graph
- Automated Research
- Strategy Retirement

## 41. 最终版本

最终系统：

```
             MARKET DATA
                  │
             NEWS / EVENTS
                  │
                  ▼
          ┌───────────────┐
          │ Feature Store │
          └───────┬───────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
 Market Regime          News Engine
        │                   │
        └─────────┬─────────┘
                  ▼
            Research LLM
                  │
          ┌───────┴───────┐
          ▼               ▼
      Factor Agent    Strategy Agent
          │               │
          └───────┬───────┘
                  ▼
             Backtester
                  │
                  ▼
             Critic Agent
                  │
                  ▼
          Robustness Tests
                  │
                  ▼
          Strategy Registry
                  │
          ┌───────┴────────┐
          ▼                ▼
     Paper Trading     Rejected
          │
          ▼
     Live Trading
          │
          ▼
     Performance
          │
          ▼
     Research Feedback
          │
          └───────────────→ Research LLM
```

## 42. 最重要的设计思想

这个系统最终不是：

```
LLM
 ↓
预测股票
 ↓
BUY
```

而是：

```
                    ┌─────────────┐
                    │ LLM         │
                    │ Researcher  │
                    └──────┬──────┘
                           │
                     Generate
                           ↓
                    ┌─────────────┐
                    │ Hypothesis  │
                    └──────┬──────┘
                           ↓
                    ┌─────────────┐
                    │ Experiment  │
                    └──────┬──────┘
                           ↓
                    ┌─────────────┐
                    │ Backtest    │
                    └──────┬──────┘
                           ↓
                    ┌─────────────┐
                    │ Validation  │
                    └──────┬──────┘
                           ↓
                     ┌─────┴─────┐
                     │           │
                   FAIL        PASS
                     │           │
                     ↓           ↓
                 Research     Paper
                               │
                               ↓
                              Live
                               │
                               ↓
                           Performance
                               │
                               ↓
                            Research
```

## 43. 核心成功标准

系统成功不应该定义为：

```
Backtest Sharpe > 2
```

而应该定义为：

```
OOS positive expectancy
        +
Multiple market regimes
        +
Multiple assets
        +
Stable parameters
        +
Low sensitivity to transaction costs
        +
Paper trading confirmation
        +
Live performance consistency
```

最终目标：

> **发现具有真实、稳定、可重复统计优势的交易策略，而不是发现历史数据中表现最好的策略。**

## 44. 推荐开发顺序

```
Phase 1
Data Pipeline
        ↓
Phase 2
Feature Engine
        ↓
Phase 3
Backtester
        ↓
Phase 4
Strategy DSL
        ↓
Phase 5
LLM Strategy Generator
        ↓
Phase 6
Walk Forward Validation
        ↓
Phase 7
Strategy Registry
        ↓
Phase 8
ML Prediction
        ↓
Phase 9
Paper Trading
        ↓
Phase 10
News/Event Engine
        ↓
Phase 11
Multi-Agent Research
        ↓
Phase 12
Live Trading
```

## 45. 最终目标

建立一个能够持续运行的：

### AI Quant Research Lab

它每天自动：

```
获取数据
   ↓
分析市场
   ↓
分析新闻
   ↓
识别 Regime
   ↓
寻找异常
   ↓
生成 Hypothesis
   ↓
生成策略
   ↓
回测
   ↓
验证
   ↓
淘汰过拟合策略
   ↓
保存优秀策略
   ↓
Paper Trading
   ↓
评估真实表现
   ↓
继续研究
```

最终形成：

> **一个能够持续进行 Quantitative Research，而不是单纯进行股票预测的 AI 系统。**
