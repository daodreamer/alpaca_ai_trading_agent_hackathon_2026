"""Pre-registration, one-shot enforcement, and the ancestry taint check.

The embargo apparatus is only worth its complexity if spending it is governed.
Three things have to be true before the sealed years may be read for a
candidate, and each of them is a way the check could otherwise be skipped
without anyone noticing:

1.  The candidate was **declared before** the data was read, with the selection
    rule written down. A hypothesis chosen after seeing the answer is not a
    hypothesis.
2.  The sealed run happens **once**. A second attempt is refused rather than
    overwritten, because "we re-ran it" and "we re-ran it until it worked" are
    indistinguishable from the outside once the first result is gone.
3.  Nothing in the candidate's ancestry read the embargoed years during the
    search. A rule selected under a tainted seal was selected with knowledge of
    the answer, whatever the sealed run then reports.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from aqr.dsl.loader import loads
from aqr.dsl.schema import StrategySpec
from aqr.registry.db import ExperimentRecord, PreregistrationError, Registry
from aqr.seal import Seal

SPEC = """
strategy:
  name: prereg_probe
  hypothesis: Something falsifiable.
  universe: {symbols: [SPY], timeframe: 1D}
  entry: close > ema(20)
  exit:
    stop_loss: {type: atr, multiplier: 2.0, period: 14}
    take_profit: {type: risk_reward, ratio: 2.0}
    max_holding_bars: 20
  sizing: {risk_per_trade: 0.01, max_position_pct: 0.25}
"""

OTHER = SPEC.replace("prereg_probe", "prereg_other").replace("ema(20)", "ema(50)")

RULE = "highest score in campaign 07; declared before any sealed bar was read"


@pytest.fixture
def registry(tmp_path: Path) -> Iterator[Registry]:
    with Registry(tmp_path / "research.sqlite") as reg:
        yield reg


def _spec(text: str = SPEC) -> StrategySpec:
    return loads(text)


def _experiment(
    fingerprint: str,
    *,
    seal: dict[str, Any] | None = None,
    name: str = "prereg_probe",
) -> ExperimentRecord:
    return ExperimentRecord(
        fingerprint=fingerprint,
        strategy_name=name,
        symbols=("SPY",),
        timeframe="1D",
        data_start="2010-01-01",
        data_end="2024-01-01",
        dataset_version="synthetic-v1",
        verdict="ACCEPT",
        seal=seal,
    )


def _certificate(*, tainted: bool = False, run_id: str = "campaign-a") -> dict[str, Any]:
    return {
        "digest": "abc",
        "tainted": tainted,
        "phase": "research",
        "embargo_start": 0,
        "max_event_time": 0,
        "loads": 1,
        "run_id": run_id,
        "knowledge_exposure": None,
    }


# --------------------------------------------------------------------------
# 3.1 Pre-registration


def test_a_declaration_is_recorded_with_its_rule_and_the_seal_it_was_made_under(
    registry: Registry,
) -> None:
    spec = _spec()
    fingerprint = registry.upsert_strategy(spec)
    seal = Seal()

    declared = registry.preregister(
        fingerprint, selection_rule=RULE, seal_digest=seal.digest
    )

    assert declared.fingerprint == fingerprint
    assert declared.selection_rule == RULE
    assert declared.seal_digest == seal.digest
    assert declared.declared_at
    assert registry.preregistration(fingerprint) == declared


def test_the_selection_rule_may_not_be_blank(registry: Registry) -> None:
    """An unstated rule is indistinguishable from one invented after the fact."""
    fingerprint = registry.upsert_strategy(_spec())
    with pytest.raises(PreregistrationError):
        registry.preregister(fingerprint, selection_rule="   ", seal_digest="d")


def test_an_unknown_strategy_cannot_be_preregistered(registry: Registry) -> None:
    with pytest.raises(PreregistrationError):
        registry.preregister("not-a-fingerprint", selection_rule=RULE, seal_digest="d")


def test_declaring_twice_is_refused(registry: Registry) -> None:
    """A second declaration is a rewritten hypothesis wearing the first timestamp."""
    fingerprint = registry.upsert_strategy(_spec())
    registry.preregister(fingerprint, selection_rule=RULE, seal_digest="d")
    with pytest.raises(PreregistrationError):
        registry.preregister(fingerprint, selection_rule="a better rule", seal_digest="d")


def test_a_candidate_with_no_preregistration_cannot_be_sealed_run(
    registry: Registry,
) -> None:
    fingerprint = registry.upsert_strategy(_spec())
    with pytest.raises(PreregistrationError):
        registry.record_sealed_run(fingerprint, result={"alpha": 0.1})


def test_the_sealed_run_is_recorded_once_and_can_be_read_back(registry: Registry) -> None:
    fingerprint = registry.upsert_strategy(_spec())
    registry.preregister(fingerprint, selection_rule=RULE, seal_digest="d")

    registry.record_sealed_run(fingerprint, result={"alpha": 0.1013, "t_alpha": 2.02})

    stored = registry.sealed_run(fingerprint)
    assert stored is not None
    assert stored["result"]["t_alpha"] == 2.02
    assert stored["sealed_run_at"]


def test_a_second_sealed_run_is_refused_not_overwritten(registry: Registry) -> None:
    """The whole value of one shot is that the first shot cannot be discarded."""
    fingerprint = registry.upsert_strategy(_spec())
    registry.preregister(fingerprint, selection_rule=RULE, seal_digest="d")
    registry.record_sealed_run(fingerprint, result={"t_alpha": -0.4})

    with pytest.raises(PreregistrationError):
        registry.record_sealed_run(fingerprint, result={"t_alpha": 2.9})

    stored = registry.sealed_run(fingerprint)
    assert stored is not None
    assert stored["result"]["t_alpha"] == -0.4


def test_preregistering_after_the_seal_was_spent_is_refused(registry: Registry) -> None:
    fingerprint = registry.upsert_strategy(_spec())
    registry.preregister(fingerprint, selection_rule=RULE, seal_digest="d")
    registry.record_sealed_run(fingerprint, result={})
    with pytest.raises(PreregistrationError):
        registry.preregister(fingerprint, selection_rule=RULE, seal_digest="d")


def test_two_candidates_keep_separate_declarations(registry: Registry) -> None:
    a = registry.upsert_strategy(_spec())
    b = registry.upsert_strategy(_spec(OTHER))
    registry.preregister(a, selection_rule="rule a", seal_digest="da")
    registry.record_sealed_run(a, result={})

    # b is untouched by the seal a spent.
    registry.preregister(b, selection_rule="rule b", seal_digest="db")
    registry.record_sealed_run(b, result={})
    assert registry.sealed_run(a) is not None
    assert registry.sealed_run(b) is not None


# --------------------------------------------------------------------------
# 3.2 Ancestry taint


def test_a_clean_ancestry_passes(registry: Registry) -> None:
    fingerprint = registry.upsert_strategy(_spec())
    registry.record_experiment(_experiment(fingerprint, seal=_certificate()))
    report = registry.ancestry_taint(fingerprint)
    assert report.clean
    assert report.tainted == ()
    assert report.experiments == 1


def test_one_tainted_ancestor_disqualifies_the_candidate(registry: Registry) -> None:
    fingerprint = registry.upsert_strategy(_spec())
    registry.record_experiment(_experiment(fingerprint, seal=_certificate()))
    registry.record_experiment(_experiment(fingerprint, seal=_certificate(tainted=True)))

    report = registry.ancestry_taint(fingerprint)

    assert not report.clean
    assert len(report.tainted) == 1


def test_taint_anywhere_in_the_campaign_disqualifies_the_candidate(
    registry: Registry,
) -> None:
    """The campaign is the unit of contamination, not the individual backtest.

    A process that read the embargoed years while evaluating hypothesis 12 was
    contaminated when it evaluated hypothesis 13, whatever the ledger of
    hypothesis 13 says. The ``run_id`` of the seal is what ties them together:
    one process, one seal, one campaign.
    """
    candidate = registry.upsert_strategy(_spec())
    sibling = registry.upsert_strategy(_spec(OTHER))
    registry.record_experiment(_experiment(candidate, seal=_certificate(run_id="c7")))
    registry.record_experiment(
        _experiment(
            sibling, seal=_certificate(tainted=True, run_id="c7"), name="prereg_other"
        )
    )

    report = registry.ancestry_taint(candidate)

    assert not report.clean
    assert "c7" in report.campaigns


def test_an_unrelated_campaign_taint_does_not_disqualify(registry: Registry) -> None:
    candidate = registry.upsert_strategy(_spec())
    unrelated = registry.upsert_strategy(_spec(OTHER))
    registry.record_experiment(_experiment(candidate, seal=_certificate(run_id="clean")))
    registry.record_experiment(
        _experiment(
            unrelated,
            seal=_certificate(tainted=True, run_id="dirty"),
            name="prereg_other",
        )
    )

    report = registry.ancestry_taint(candidate)

    assert report.clean


def test_experiments_with_no_seal_are_reported_as_unrecorded_not_as_clean(
    registry: Registry,
) -> None:
    """Silence is not evidence of innocence, and must not be scored as such.

    Every experiment written before the seal was recorded in the registry looks
    exactly like a clean one from here. Counting it clean would let the check
    pass on a database where it never actually ran.
    """
    fingerprint = registry.upsert_strategy(_spec())
    registry.record_experiment(_experiment(fingerprint, seal=None))
    report = registry.ancestry_taint(fingerprint)
    assert report.clean
    assert report.unrecorded == 1


def test_the_seal_certificate_survives_a_round_trip(registry: Registry) -> None:
    fingerprint = registry.upsert_strategy(_spec())
    seal = Seal()
    registry.record_experiment(_experiment(fingerprint, seal=seal.certificate()))
    row = registry.experiments(fingerprint=fingerprint)[0]
    assert json.loads(row["seal"])["run_id"] == seal.run_id


def test_the_taint_check_reads_a_database_written_before_the_column_existed(
    tmp_path: Path,
) -> None:
    """Opening an old research log must migrate it, not crash on it.

    18MB of campaign history predates these tables. A schema change that
    required starting over would delete the multiple-comparisons denominator,
    which is the one number this project cannot reconstruct.
    """
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE strategies (
            fingerprint TEXT PRIMARY KEY, name TEXT NOT NULL, version INTEGER NOT NULL,
            parent TEXT, status TEXT NOT NULL, hypothesis TEXT NOT NULL DEFAULT '',
            spec_yaml TEXT NOT NULL, score REAL, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL);
        CREATE TABLE experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT, fingerprint TEXT NOT NULL,
            strategy_name TEXT NOT NULL, hypothesis TEXT NOT NULL DEFAULT '',
            symbols TEXT NOT NULL, timeframe TEXT NOT NULL, data_start TEXT NOT NULL,
            data_end TEXT NOT NULL, dataset_version TEXT NOT NULL, train_metrics TEXT,
            oos_metrics TEXT, robustness TEXT, overfitting TEXT, evaluation TEXT,
            verdict TEXT, score REAL, backtests_run INTEGER NOT NULL DEFAULT 1,
            llm_model TEXT, prompt_hash TEXT, code_hash TEXT, error TEXT,
            created_at TEXT NOT NULL);
        """
    )
    conn.execute(
        "INSERT INTO experiments (fingerprint, strategy_name, symbols, timeframe, "
        "data_start, data_end, dataset_version, created_at) "
        "VALUES ('old-fp', 'old', 'SPY', '1D', 'a', 'b', 'v', 'then')"
    )
    conn.commit()
    conn.close()

    with Registry(path) as reg:
        report = reg.ancestry_taint("old-fp")
        assert report.clean
        assert report.unrecorded == 1
        assert reg.preregistration("old-fp") is None
