# 00 — Competition brief (the constraints we cannot negotiate)

Source: https://lablab.ai/ai-hackathons/alpaca-ai-trading-agents-hackathon

## Timebox

| Milestone | When |
| --- | --- |
| Kickoff, registration closes | 2026-08-28 15:00 UTC |
| Submission closes, judging begins | 2026-09-04 15:00 UTC |

Seven days. Solo. Assume **four usable trading days** of live paper activity,
not seven — the agent cannot go live on day one.

## Hard gates (fail any of these and the submission is void)

1. Must use Alpaca's **Trading API**.
2. Must use Alpaca's **MCP server or CLI** — at least one, genuinely, not as a
   checkbox.
3. The strategy **must incorporate options trading**.
4. Submission must run on a **new, dedicated Alpaca paper trading account**.

## Judging criteria

1. **P&L Performance**
2. **Technology Implementation**
3. **Creativity & Originality**
4. **Presentation & Execution**

## How we read the criteria

**P&L over four days of options trading is mostly noise.** A single directional
0DTE bet dominates the sample either way. We do not optimise for the tail we
cannot control. We optimise for a P&L result that is *defensible*:

- Positive return, **max drawdown < 5%**.
- **Defined-risk structures only** — verticals, condors, covered calls,
  cash-secured puts. No naked short options. Ever. This is a Risk Gate invariant
  (see [03-risk-gate.md](03-risk-gate.md)), not a guideline.
- Enough trades that the result is not one coin flip. Target **≥ 30 fills**.
- A **backtest of the same strategy** over a longer window, with commissions and
  spread modelled. This is what converts "we were up 2% in four days" into
  "the strategy has an edge, and here is the four-day sample of it".

**Technology Implementation and Presentation are where the field separates.**
The median submission will be a loop, one LLM call, and a market order. Our
advantage is a tested deterministic core, a risk layer with veto power, and a
decision record for every fill.

**Creativity** in a single-track field of 2,000+ builders means a one-sentence
angle that is not "LLM reads news, buys calls". Ours: *the model proposes
structure, deterministic code holds the veto.*

## Non-goals

- Live/real-money trading. Paper only.
- Beating a benchmark. We are not making a performance claim.
- Investment advice. The output is agent state and rationale, not a recommendation.
