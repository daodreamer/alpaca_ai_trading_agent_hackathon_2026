# AI Quant Researcher

> An AI-driven platform for quantitative strategy research, backtesting, validation and automated evolution.
>
> Goal: use LLM + Machine Learning + Quantitative Research to automatically discover trading strategies that hold a statistical edge and generalise across market regimes.
>
> Not an HFT system. It targets minute-, hour- and day-level strategies, plus positions held for several days.

## 1. Project goals

### 1.1 Core goal

Build an automated Quant Research Lab:

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

The core of the system is not to have the LLM predict stock prices directly. It is to:

1. Find latent regularities in the market
2. Generate trading hypotheses
3. Turn hypotheses into executable strategies
4. Run historical backtests automatically
5. Detect overfitting automatically
6. Run out-of-sample validation
7. Confirm in paper trading
8. Keep improving from real trading results

## 2. Core design principles

### 2.1 The LLM is not responsible for trading

The LLM does not directly perform:

- BUY
- SELL
- Position sizing
- Stop loss execution
- Order execution

The LLM is responsible for:

- Strategy research
- Hypothesis generation
- Feature discovery
- Event interpretation
- Market regime reasoning
- Strategy explanation
- Backtest analysis
- Research iteration

The final trading decision must be made by a deterministic Strategy Engine / ML Model / Risk Engine.

## 3. Overall system architecture

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

## 4. System modules

### 4.1 Data Layer

Responsible for fetching and storing all research data.

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

Main timeframes:

- 1m
- 5m
- 15m
- 30m
- 1h
- 4h
- 1D
- 1W

MVP:

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

Must be stored:

- `event_time`
- `ingestion_time`

To prevent look-ahead bias.

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

Recommended:

```
Raw Data
    ↓
Parquet
    ↓
DuckDB
```

For research.

Production: `PostgreSQL`

For:

- Strategy metadata
- Signals
- Orders
- Positions
- Performance
- Research experiments

Redis for:

- Cache
- Real-time state
- Event queue
- Signal state

## 5. Feature Engine

The Feature Engine does not use an LLM to compute base indicators.

Those must be computed with deterministic code.

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

This is a very important module in this project.

Goal:

> Turn unstructured news into structured Market Events.

### 6.1 LLM News Extraction

Input:

- Headline
- Article
- Company
- Time

Output:

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

Definition:

```
Event Score =
Materiality
× Surprise
× Confidence
× Market Relevance
```

For example:

```
FDA Approval
Materiality = 0.95
Surprise = 0.90
Confidence = 0.98
Relevance = 0.95

Event Score = 0.795
```

## 7. Market Regime Engine

Goal:

> Judge the current market environment, rather than predict a specific price.

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

This is the "brain" of the whole system.

But the LLM does not trade directly.

### 8.1 What the Research Agent does

1. Analyse past experiments
2. Find latent regularities
3. Propose a hypothesis
4. Design factors
5. Generate a strategy
6. Analyse why something failed
7. Revise the strategy
8. Generate a new experiment

### 8.2 Strategy Hypothesis

For example:

```
Hypothesis:

When the market is in a bull trend:

SPY > EMA200
ADX > 20

and at the same time:

price pulls back to EMA20
RSI > 40
RVOL > 1.2

then:

the probability of a rise over the next
3 trading days may be significantly
higher than baseline.
```

### 8.3 Strategy DSL

Do not let the LLM emit arbitrary Python.

Design your own DSL instead.

For example:

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

That gives you:

```
LLM
 ↓
DSL
 ↓
Validator
 ↓
Backtester
```

and avoids the LLM generating malicious or incorrect code.

## 9. Strategy Generator

The Strategy Generator turns the LLM's hypothesis into a standard strategy.

### A strategy must contain

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

The backtester is the core of the system.

It must simulate real trading.

### Must account for

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

## 11. Preventing look-ahead bias

This is one of the highest priorities in the system.

The wrong way:

```
using data from 2026-01-02 on 2026-01-01
```

It must hold that:

```
signal_time
<
available_time
```

Every piece of data must record:

- `event_time`
- `available_time`

## 12. Train / Validation / Test

Recommended:

```
Historical Data

├── Train       60%
├── Validation  20%
└── Test        20%
```

The key rule:

> The LLM is not allowed to see the test data.

## 13. Walk Forward Validation

A fixed train/test split is not enough.

Use:

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

Final metric: `Average OOS Performance`

## 14. Robustness Testing

A strategy must pass several tests.

### 14.1 Parameter Perturbation

For example:

```
EMA = 20
```

Test:

- 15
- 18
- 20
- 22
- 25

If:

```
20 → Sharpe 2.1
19 → Sharpe 0.4
21 → Sharpe 0.3
```

that is severe overfitting.

### 14.2 Market Test

Test:

- Bull Market
- Bear Market
- Sideways Market
- High Volatility
- Low Volatility

### 14.3 Asset Test

Do not test only `SPY`.

Test:

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

Apply to the trade sequence:

- Random shuffle
- Bootstrap
- Trade resampling

To estimate:

- Expected Drawdown
- Worst Case Drawdown
- Probability of Ruin

## 15. Strategy Evaluator

The evaluator gives every strategy a composite score.

For example:

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

Must detect:

- Parameter count
- Number of experiments
- Number of backtests
- Sharpe inflation
- Train/OOS gap
- Performance concentration
- Asset concentration
- Time concentration

### 16.1 Experiment Database

Every experiment must be stored.

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

That is the only way to know:

> what the LLM actually tried.

## 17. Strategy Registry

Only a validated strategy may enter the registry.

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

Through:

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

Run for at least `30 - 90 days`.

The exact length depends on the strategy's frequency.

## 19. Live Trading

Live trading must be isolated from the research environment.

```
Research
    │
    X
    │
Live Trading
```

The LLM must not hold a `Direct Broker API`.

Recommended:

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

The Risk Engine must be deterministic code.

### Risk Rules

- Max risk per trade
- Max daily loss
- Max portfolio drawdown
- Max position size
- Max sector exposure
- Max correlated exposure
- Max leverage
- Max number of positions

For example:

```
Risk per trade = 0.5%

Maximum portfolio drawdown = 15%

Maximum single position = 10%

Maximum sector exposure = 30%
```

## 21. Position Sizing

Recommended to support:

- Fixed Fractional
- ATR Position Sizing
- Volatility Scaling
- Kelly Fraction
- Risk Parity

For example:

```
Risk = 0.5% portfolio

Stop distance = $5

Portfolio = $100,000

Risk amount = $500

Shares = 500 / 5

       = 100 shares
```

## 22. News-driven Strategy

A distinctive part of this project.

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

LLM:

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

Market:

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

System:

```
Event Score = 0.84

Market Reaction Score = 0.91

Technical Score = 0.87

Final Score = 0.89
```

This then feeds the strategy model.

## 23. ML Prediction Layer

The LLM is not responsible for the final numeric prediction.

Use:

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

For example:

```
P(return > 3% within 1 day)
P(return > 5% within 3 days)
P(return < -3% within 1 day)
```

## 24. Signal Engine

The final signal:

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

Output:

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

One of the core innovations of the system.

Each experiment:

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

For example:

```
Momentum_v1

Sharpe = 0.82

        ↓

Momentum_v2

add Market Regime

Sharpe = 1.21

        ↓

Momentum_v3

add RVOL

Sharpe = 1.48

        ↓

Momentum_v4

add a volatility filter

Sharpe = 1.52
```

But:

> Only an improvement in OOS performance may be promoted.

## 26. Multi-Agent Architecture

Recommended as the eventual shape:

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

Responsible for:

- research direction
- experiment planning
- strategy selection

### Strategy Agent

Responsible for: `Trading Strategy`

### Factor Agent

Responsible for:

- Factor discovery
- Feature engineering

### News Agent

Responsible for:

- Event extraction
- Event classification

### Backtest Agent

Responsible for: `Backtest analysis`

### Critic Agent

Responsible for:

- Overfitting
- Bias
- Robustness

### Risk Agent

Responsible for: `Risk assessment`

## 28. LLM Memory

The system must store:

- Research History
- Hypothesis
- Experiment
- Result
- Failure
- Successful Strategy
- Rejected Strategy

The LLM reads `Previous experiments` at the start of the next research round,

so it does not explore the same ground twice.

## 29. Research Knowledge Graph

You can build:

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

For example:

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

Recommended: `FastAPI`

API:

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

Research tasks should not run synchronously.

Use:

```
Redis
+
Celery / Dramatiq
```

For example:

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

Use:

- Cron
- Airflow
- Prefect

Jobs:

- Daily Data Update
- Daily Feature Calculation
- Daily Strategy Scan
- Weekly Research
- Weekly Strategy Evaluation
- Monthly Strategy Retirement

## 33. Recommended tech stack

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

Supported:

- OpenAI
- Anthropic
- Google
- Local LLM

The LLM should be called through a single unified interface.

### Backtesting

MVP: `VectorBT`

Or: `Custom Event-driven Backtester`

### Frontend

- React
- TypeScript
- Next.js
- TradingView Lightweight Charts

Dashboard:

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

Must be recorded:

- LLM calls
- Token usage
- Latency
- Errors
- Backtest duration
- Strategy generation
- Model version
- Dataset version
- Trading decisions

Use:

- OpenTelemetry
- Prometheus
- Grafana

## 36. Security

The LLM:

- NO direct broker access
- NO direct database write
- NO arbitrary shell access

The LLM may only:

- Read Research Data
- Generate DSL
- Generate Hypothesis
- Request Backtest
- Read Results

All execution happens in a `Sandbox`.

## 37. MVP

Do not build every feature in the first phase.

### MVP v0.1

Goal:

> Automatically discover technical-analysis strategies.

Supports only:

- US Stocks
- Daily OHLCV
- Technical Indicators
- LLM Strategy Generator
- Backtester
- Walk Forward
- Strategy Ranking

Pipeline:

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

Adds:

- 5m / 15m data
- Volume
- Volatility
- Market Regime
- ML Model
- Paper Trading

## 39. MVP v0.3

Adds:

- News
- LLM Event Extraction
- Event Strategy
- News-driven ML
- Real-time Signal Engine

## 40. MVP v0.4

Adds:

- Multi-Agent
- Strategy Evolution
- Knowledge Graph
- Automated Research
- Strategy Retirement

## 41. Final version

The final system:

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

## 42. The most important design idea

In the end this system is not:

```
LLM
 ↓
predict the stock
 ↓
BUY
```

but:

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

## 43. Core success criteria

Success must not be defined as:

```
Backtest Sharpe > 2
```

It must be defined as:

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

The ultimate goal:

> **Discover trading strategies with a real, stable, repeatable statistical edge — not the strategy that performed best on historical data.**

## 44. Recommended development order

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

## 45. Ultimate goal

Build something that can keep running on its own:

### AI Quant Research Lab

Every day it automatically:

```
fetch data
   ↓
analyse the market
   ↓
analyse the news
   ↓
identify the regime
   ↓
look for anomalies
   ↓
generate a hypothesis
   ↓
generate a strategy
   ↓
backtest
   ↓
validate
   ↓
discard overfitted strategies
   ↓
keep the good ones
   ↓
paper trade
   ↓
evaluate real performance
   ↓
keep researching
```

What this adds up to:

> **An AI system that keeps doing quantitative research, rather than one that merely predicts stock prices.**
