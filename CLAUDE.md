# CLAUDE.md — AlphaGate (Alpaca AI Trading Agents Hackathon, 2026-08-28 → 09-04)

## 1. Role

You are the AI software-engineering collaborator on this project. It is a 7-day
solo hackathon project built with SDD + TDD. Read `specs/` first, then write code.

## 2. What this file governs

**Sections 3 to 7 apply to `backend/` only.**

`ai_quant_researcher/` is a different project living in the same repository — the
equity strategy research system that implements
[specs/trading_strategy_architecture.md](specs/trading_strategy_architecture.md).
It has its own `pyproject.toml`, its own venv, its own tests and its own boundary
tests, and it **imports nothing from `alphagate`**.

It holds no money (so there is no `Decimal` requirement), places no orders (so
there is no Risk Gate) and never touches Alpaca. Its LLM boundary is enforced by
its own
[tests/test_boundaries.py](ai_quant_researcher/tests/test_boundaries.py).

The two projects answer different questions and are kept apart deliberately.
**Do not try to "unify" them**, and do not carry AlphaGate's invariants across —
in either direction. When changing `ai_quant_researcher/`, use its own commands
(see the README in that directory), not the ones in section 6.

**The interface between them is files, and nothing else.** There are three of
them today, all one-way (the research side writes, `backend/` reads):

| File | Written by | Carries |
| --- | --- | --- |
| `runs/target_books/*.json` | `aqr target-book` | the equity weight vector (specs/09 D0) |
| `runs/option_books/*.json` | `aqr option-book` | the option **rule**, deliberately without strikes (specs/07 D1) |
| `data-options-sealed/volatility_history/SPY.csv` | `aqr options-pull` | the implied-volatility history that `alphagate iv-seed` reads |

The absence of strikes in the option book is deliberate: a 5480 written down at
Tuesday's close is already wrong at Wednesday's open, so what travels is the
rule, and the execution side resolves it against the live chain itself — which is
also why `backend/` must have its own delta leg-selection code and cannot import
`aqr.options.chain`.

The third row is a file rather than a function call for the same reason:
`iv-seed` reads it with the standard-library `csv` module and hands the parsed
mapping to `IvHistoryStore.seed_from_vendor_history`, importing nothing from
`aqr`.

Neither side imports the other. Guard 9 in `backend/tests/test_boundaries.py`
enforces that in both directions, and `scripts/pipeline.py` only ever invokes the
two CLIs as subprocesses.

When you feel the urge to "just share one constant" or "just call one function on
the other side": that urge is exactly why this line exists.

## 3. Non-negotiable rules

1. The `core` / `options` / `risk` layers are pure: they depend only on the
   standard library and on each other. Enforced by
   `backend/tests/test_boundaries.py`, not by good intentions.
2. **Only `agent/` may call an LLM.** A model call inside the Risk Gate means the
   Gate is no longer a Gate.
3. **Every order that reaches Alpaca goes through the Risk Gate.** No bypass, no
   `force=True`, no debug switch. `execution/` accepts only a `GatedOrder`, and
   only `risk.gate` can construct a `GatedOrder`.
4. Money is `Decimal`, end to end. Greeks and IV are `float` (they are estimates,
   not money).
5. All times are tz-aware UTC. The Gate and the domain never read the clock; time
   is always passed in as an argument.
6. **No naked short options.** The structure is unrepresentable at the type level
   (specs/02 D3).
7. Domain computations are deterministic: the same inputs and the same
   configuration must give the same result, including the order of the checks.
8. No look-ahead bias. Backtest and live trading run the same code path; the only
   difference is the clock.
9. No real money, no investment advice. The output is agent state and reasoning,
   not buy/sell recommendations.
10. Never leak an API key or account id into logs, the journal, the dashboard or
    the demo video.

## 4. TDD

RED → GREEN → REFACTOR. `options/` and `risk/` are the correctness surface, so
tests come first there. `agent/`, `interface/` and the dashboard can be moved
pragmatically — do not spend hackathon time on type stubs for the dashboard.

Never weaken an invariant to make a test pass. Never use a mock to paper over
real behaviour.

## 5. Time discipline (this is a hackathon, not a product)

- Commit before the end of each day, and keep `main` runnable.
- Any direction that goes more than 2 hours without an outcome: stop and
  re-evaluate.
- After Sep 3, only demo work, documentation and bug fixes — no new features.
- Get trading running early: target ≥ 30 fills, live by D3 at the latest.

## 6. Common commands

Run these from the repository root. The CLI's default paths are anchored to the
repository root, so no arguments are needed.

| Purpose | Command |
| --- | --- |
| tests | `uv run --directory backend --extra dev pytest` |
| lint | `uv run --directory backend --extra dev ruff check .` |
| type check | `uv run --directory backend --extra dev mypy` |
| pre-open health check | `uv run --directory backend python -m alphagate preflight` |
| one cycle (does not trade by default) | `uv run --directory backend python -m alphagate once` |
| run a whole day | `uv run --directory backend python -m alphagate run [--dry-run]` |
| current state | `uv run --directory backend python -m alphagate status` |
| one day's log | `uv run --directory backend python -m alphagate show [-v] [--day YYYY-MM-DD]` |
| dashboard | `uv run --directory backend python -m alphagate serve` |

Equity side (executing the one strategy `ai_quant_researcher` validated, see
specs/09):

| Purpose | Command |
| --- | --- |
| run only the equity chain | `python scripts/pipeline.py --only equity` |
| rehearse only, no orders | `python scripts/pipeline.py --only equity --dry-run` |
| pre-open health check (book + account) | `uv run --directory backend python -m alphagate equity-preflight` |
| one rebalance, through the Gate but no orders | `uv run --directory backend python -m alphagate equity-plan` |
| one rebalance, orders for real | `uv run --directory backend python -m alphagate equity-rebalance` |
| run a whole day (heartbeat + one rebalance) | `uv run --directory backend python -m alphagate equity-run [--dry-run]` |
| current holdings and drift | `uv run --directory backend python -m alphagate equity-status` |

Options side (executing the one option rule `ai_quant_researcher` validated, see
specs/07 D1 and specs/10):

| Purpose | Command |
| --- | --- |
| run only the options chain | `python scripts/pipeline.py --only options` |
| rehearse only, no orders | `python scripts/pipeline.py --only options --dry-run` |
| run both chains (equity first) | `python scripts/pipeline.py` |
| top up the IV history (the input to `iv_rank`) | `uv run --directory backend python -m alphagate iv-seed` |

Both `ALPHAGATE_STRATEGY_FINGERPRINT` and `ALPHAGATE_OPTION_FINGERPRINT` must be
pinned in `.env.local`. This is the only checkable place where "we execute only
the strategy the research side validated" means anything — a book whose
fingerprint does not match is rejected by name by `load_target_book` /
`load_option_book`, and **the absence of a default is deliberate**. The two pins
are separate: the two sleeves execute two different rules, each validated against
a different sealed window, and a single pin covering both would make "which rule
was running at the time" unanswerable from the moment they diverged — and they
have already diverged.

`iv-seed` has no equity-side counterpart, and it is not an optimisation. The
entry condition of the researched rule is `iv_rank() < 15`, and `iv_rank` needs a
year of implied-volatility history, which Alpaca will not give you without an
OPRA subscription (see `agent/iv_store.py`). Without seeding, the rule is not
"false" but **undecidable**, and the agent stands aside every cycle — which from
the outside looks exactly like a quiet market. So it runs once every trading day.

## 6b. How the two sleeves split the money

A $100,000 account, split once and then held fixed rather than drifting with
market value:

| Sleeve | Allocation | Constant |
| --- | --- | --- |
| Equity | $90,000 | `EQUITY_SLEEVE_ALLOCATION` in `equity/policy.py` |
| Options | $10,000 | `OPTIONS_SLEEVE_ALLOCATION` in `risk/limits.py` |

The two must sum to the account total, and a test enforces it — Alpaca has one
buying-power pool and knows nothing about sleeves, so this split only means
anything while it adds up.

The options side is $10,000 rather than something smaller for a computed reason,
not a chosen one: the rule sells the 0.16-delta SPY put and buys the 0.08-delta
wing, and the measured max loss of one contract is $1,389. A $5,000 sleeve gives
a per-trade budget of only $1,000, `agent/sizing.py` floors the contract count to
**0**, and the rule can never open a position — and in the logs that looks
exactly like "no opportunity in the market". At $10,000 the per-trade budget is
exactly $2,000, matching the sizing the research side ran (2% of $100,000,
specs/10 D8a).

Frontend (the dashboard's Live page, Vite + React + shadcn/ui):

| Purpose | Command |
| --- | --- |
| build into the Python package | `cd frontend && npm run build` |
| dev mode | `cd frontend && npm run dev` |
| checks | `cd frontend && npx eslint . && npm run typecheck` |

**`run` without `--dry-run` places real orders.** `once` does not trade by
default, `run` does — a command you use for debugging should not casually place
orders, and running a whole day is meant to trade in the first place.

## 7. Definition of Done

- the spec exists and is unambiguous
- covered by positive, boundary and determinism tests
- no look-ahead
- error paths are handled and observable
- pytest / ruff / mypy all green (plus eslint / tsc green if `frontend/` changed)
