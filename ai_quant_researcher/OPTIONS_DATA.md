# Option chains, end of day, free

`aqr options-pull` caches historical option chains from DoltHub's public
[`post-no-preference/options`](https://www.dolthub.com/repositories/post-no-preference/options)
database. No key, no account, no subscription.

```
uv run aqr options-pull                              # both tables, the whole thing
uv run aqr options-pull --table volatility_history   # the small one: ~1 minute
uv run aqr options-pull --skip-download              # re-split what is already on disk
```

Output:

```
data-options/_raw/option_chain.csv.gz            the vendor file, as downloaded
data-options/_raw/volatility_history.csv.gz
data-options/option_chain/<symbol>.csv           filtered to the universe
data-options/volatility_history/<symbol>.csv
```

Nothing here is in git. `aqr options-pull` rebuilds all of it from a public
source, and the raw chain table alone is measured in gigabytes.

## Why this source and not a broker

Neither broker this project already talks to will serve it:

| | Historical option chains |
|---|---|
| **Alpaca** | `GET /v1beta1/options/bars` → `403 OPRA agreement is not signed`. Signing it means Algo Trader Plus at $99/month, and even then the endpoint returns close and volume — no bid/ask, which is the entire economics of a spread. |
| **IBKR** | Nothing. Probed against a live TWS: `reqHistoricalData` on a listed SPY put returns **zero bars** for TRADES, MIDPOINT and BID_ASK alike, with `No data of type EODChart is available ... security type 'Option'`. Contract *definitions* (`reqSecDefOptParams`) are free and complete — 33 expirations and 491 strikes for SPY — but they carry no prices. |

DoltHub carries bid, ask, implied volatility and all five greeks, for free.

## What is in it

| Table | Columns |
|---|---|
| `option_chain` | `date, act_symbol, expiration, strike, call_put, bid, ask, vol, delta, gamma, theta, vega, rho` |
| `volatility_history` | `date, act_symbol, iv_current, iv_week_ago, iv_month_ago, iv_year_high, iv_year_low, hv_current, …` |

**`volatility_history` is the higher-value table per byte**, because IV rank
falls straight out of it and no live feed on this account can produce it:

```
iv_rank = (iv_current − iv_year_low) / (iv_year_high − iv_year_low) × 100
```

Measured on the cache, session 2026-08-28:

| Symbol | IV | year low | year high | IV rank |
|---|---|---|---|---|
| XOM | 0.2675 | 0.1816 | 0.3639 | 47.1 |
| AAPL | 0.2386 | 0.1748 | 0.3246 | 42.6 |
| JPM | 0.1983 | 0.1843 | 0.3670 | 7.7 |
| TSLA | 0.3790 | 0.3717 | 0.6453 | 2.7 |
| NVDA | 0.3248 | 0.3200 | 0.5493 | 2.1 |
| SPY | 0.1158 | 0.1090 | 0.2648 | **4.4** |

## Coverage, measured

Window **2019-02-09 → 2026-08-28**, still updated. The vendor carries about
2,322 symbols; the universe file names 682.

* **637 of 682 covered.** The 45 that are not — ACE, AET, ANDV, ARG, BCR,
  BHI and the rest — were acquired or delisted. A point-in-time universe is
  meant to contain them, and a vendor keyed on live listings is not.
* **QQQ and IWM are absent entirely.** SPY, DIA and XLF are present. Anything
  that assumes the three big index ETFs are interchangeable here will find a
  hole. Confirmed by point query against the live database, not inferred from a
  missing file.
* The ETFs are added by `--symbols`, which defaults to `SPY,QQQ,IWM`, because a
  membership file cannot contain them.

## What this data is not

**End of day only.** One snapshot per session. `1D` options research is
possible; `1h` and `4h` are not, and no free source offers them. Intraday
option data starts around $40/month elsewhere.

**A subset of each chain.** Roughly 200 contracts per symbol per session,
against the 13,514 CBOE lists for SPY on the same day. The strike ladder near
the money is there; the far wings and most expirations are not. Adequate for
delta-targeted structures, inadequate for surface work — and, importantly, the
difference is invisible from inside a row.

## Two phases, and why

The CSV endpoint generates its answer on the fly. Confirmed against the live
service: it reports no `content-length`, `HEAD` times out, and it honours no
`Range` header. A transfer that dies at minute fifty therefore cannot be
resumed.

So phase one only moves bytes, to `_raw/`, writing a `.part` file it renames
only on a clean finish — an interrupted transfer cannot be mistaken for a
complete one. Phase two splits that local file and may be re-run freely with
`--skip-download`. The download is hours; the split is a minute, and a change
to the splitting logic must not cost a second transfer.

**The SQL API is not an alternative.** Aggregates (`count`, `min`, `group by`)
exceed the server's query deadline on a table this size, and a plain `select`
silently returns **zero rows** above roughly a thousand — not an error, an
empty success. A puller built on it would produce a short cache indistinguishable
from a complete one. Point lookups for one symbol on one date do work, and are
what a coverage probe should use.

## What it does not yet do

There is no options research in `aqr` — only the data. Three things stand
between this cache and an LLM proposing option strategies, and none of them is
data:

1. **The DSL cannot express an option.** `Universe(symbols, timeframe)`,
   `signal`/`portfolio`, stop, target, sizing. No strike, expiration, right,
   multiplier or second leg — so there is no field in which a model could write
   "sell the 16-delta put at 7 DTE".
2. **The feature registry has no options features.** Every entry is derived
   from a bar. `iv_rank` is now one division away, but term structure, skew and
   anything surface-shaped are not there at all.
3. **The backtest engine models share fills.** Decision at bar `t`'s close,
   fill at `t + 1`'s open, positions counted in shares, exits on price. An
   options engine needs multi-leg entry and exit at each leg's own bid/ask,
   expiry settlement, assignment, and the fact that the instrument itself ages
   — a 7-DTE option is a 6-DTE option tomorrow. That is a different engine, not
   a parameter.

The cache makes those buildable. It does not make them built.
