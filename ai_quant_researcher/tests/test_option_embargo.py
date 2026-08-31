"""The embargo, over option chains.

`test_embargo_providers.py` covers the same three locks for bars. What is
pinned here is the part that is genuinely new: a chain row never becomes
`Bars`, so `Bars.__post_init__` — the sensor the whole seal rests on — never
fires for it, and a cache built without a replacement would be a route into the
process the seal cannot see.
"""

from __future__ import annotations

import csv
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from aqr.data.option_embargo import (
    CHAIN_TABLE,
    VOLATILITY_TABLE,
    OptionChain,
    audit_option_root,
    load_research,
    load_sealed,
    split_at_embargo,
    write_option_canary,
)
from aqr.data.options_chain import CHAIN_COLUMNS
from aqr.seal import CANARY_SYMBOL, EMBARGO_START, Phase, Seal, scope

BEFORE = "2024-08-30"
AFTER = "2024-09-03"


def chain_csv(path: Path, days: tuple[str, ...], symbol: str = "SPY") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CHAIN_COLUMNS)
        for day in days:
            writer.writerow(
                [day, symbol, day, "550.00", "Put", "1.10", "1.15", "0.15", "-0.16"]
                + ["0.01", "-0.05", "0.09", "0.02"]
            )
    return path


def epochs(*days: str) -> np.ndarray:
    return np.array(
        [int(datetime.fromisoformat(d).replace(tzinfo=UTC).timestamp()) for d in days],
        dtype=np.int64,
    )


def a_chain(symbol: str, *days: str) -> OptionChain:
    return OptionChain(
        symbol=symbol,
        table=CHAIN_TABLE,
        columns=("date", "act_symbol"),
        rows=tuple((d, symbol) for d in days),
        session=epochs(*days),
    )


# ------------------------------------------------------------------ #
# the sensor is on the type
# ------------------------------------------------------------------ #


class TestSensor:
    def test_a_pre_embargo_chain_leaves_the_seal_clean(self) -> None:
        with scope(Seal()) as seal:
            a_chain("SPY", BEFORE, "2024-08-31")
            assert not seal.tainted

    def test_a_post_embargo_chain_taints(self) -> None:
        """The whole point. A chain row never becomes `Bars`, so without this
        the sealed sessions would enter the process unobserved."""
        with scope(Seal()) as seal:
            a_chain("SPY", BEFORE, AFTER)
            assert seal.tainted

    def test_the_canary_taints_whatever_its_dates_say(self) -> None:
        """Its timestamps are not consulted: the failure it exists to catch is
        one where the timestamps are wrong too."""
        with scope(Seal()) as seal:
            a_chain(CANARY_SYMBOL, BEFORE)
            assert seal.tainted

    def test_an_empty_chain_is_not_a_peek(self) -> None:
        with scope(Seal()) as seal:
            OptionChain(
                symbol="SPY",
                table=CHAIN_TABLE,
                columns=("date",),
                rows=(),
                session=np.array([], dtype=np.int64),
            )
            assert not seal.tainted

    def test_sessions_and_rows_must_agree(self) -> None:
        with pytest.raises(ValueError, match="sessions for"):
            OptionChain(
                symbol="SPY",
                table=CHAIN_TABLE,
                columns=("date",),
                rows=((BEFORE,),),
                session=epochs(BEFORE, AFTER),
            )

    def test_the_sealed_phase_reads_without_tainting(self) -> None:
        """Taint is a research-phase concept: the sealed run is *supposed* to
        read those sessions. What makes it honest is that it cannot search."""
        seal = Seal()
        seal.enter_sealed()
        with scope(seal):
            a_chain("SPY", AFTER)
            assert not seal.tainted


# ------------------------------------------------------------------ #
# the physical lock
# ------------------------------------------------------------------ #


class TestSplit:
    def test_the_research_root_stops_before_the_embargo(self, tmp_path: Path) -> None:
        source = chain_csv(tmp_path / "src.csv", (BEFORE, "2024-08-31", AFTER, "2026-08-28"))
        result = split_at_embargo(
            source,
            symbol="SPY",
            table=CHAIN_TABLE,
            research_root=tmp_path / "research",
            sealed_root=tmp_path / "sealed",
        )
        assert result.research_rows == 2
        assert result.sealed_rows == 4
        assert result.research_last == "2024-08-31"
        assert result.sealed_last == "2026-08-28"

    def test_the_embargo_day_itself_is_sealed(self, tmp_path: Path) -> None:
        """Strictly before. The boundary belongs to the reserved window, so a
        run that stops *at* it has still not read it."""
        day = EMBARGO_START.date().isoformat()
        source = chain_csv(tmp_path / "src.csv", (BEFORE, day))
        result = split_at_embargo(
            source,
            symbol="SPY",
            table=CHAIN_TABLE,
            research_root=tmp_path / "research",
            sealed_root=tmp_path / "sealed",
        )
        assert result.research_rows == 1
        assert result.sealed_rows == 2

    def test_splitting_does_not_taint_the_run_that_does_it(self, tmp_path: Path) -> None:
        """Arranging not to see the sealed years must not be the peek.

        So the splitter reads with `csv` and never builds an `OptionChain`.
        """
        source = chain_csv(tmp_path / "src.csv", (BEFORE, AFTER))
        with scope(Seal()) as seal:
            split_at_embargo(
                source,
                symbol="SPY",
                table=CHAIN_TABLE,
                research_root=tmp_path / "research",
                sealed_root=tmp_path / "sealed",
            )
            assert not seal.tainted


# ------------------------------------------------------------------ #
# the procedural lock
# ------------------------------------------------------------------ #


class TestLoaders:
    def test_research_loads_from_the_truncated_root(self, tmp_path: Path) -> None:
        chain_csv(tmp_path / CHAIN_TABLE / "SPY.csv", (BEFORE,))
        with scope(Seal()) as seal:
            chain = load_research("SPY", root=tmp_path)
            assert len(chain) == 1
            assert not seal.tainted

    def test_the_sealed_root_is_refused_in_the_research_phase(self, tmp_path: Path) -> None:
        chain_csv(tmp_path / CHAIN_TABLE / "SPY.csv", (AFTER,))
        with scope(Seal()), pytest.raises(PermissionError, match="sealed phase"):
            load_sealed("SPY", root=tmp_path)

    def test_the_sealed_root_opens_in_the_sealed_phase(self, tmp_path: Path) -> None:
        chain_csv(tmp_path / CHAIN_TABLE / "SPY.csv", (BEFORE, AFTER))
        seal = Seal()
        seal.enter_sealed()
        with scope(seal):
            assert len(load_sealed("SPY", root=tmp_path)) == 2
            assert seal.phase is Phase.SEALED

    def test_a_missing_cache_says_which_file(self, tmp_path: Path) -> None:
        with scope(Seal()), pytest.raises(FileNotFoundError, match="SPY"):
            load_research("SPY", root=tmp_path)


# ------------------------------------------------------------------ #
# the tripwire, and the audit that must not trip it
# ------------------------------------------------------------------ #


class TestCanary:
    def test_it_is_written_past_the_embargo(self, tmp_path: Path) -> None:
        path = write_option_canary(tmp_path, CHAIN_TABLE)
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        assert all(date.fromisoformat(r["date"]) >= EMBARGO_START.date() for r in rows)
        assert all(r["act_symbol"] == CANARY_SYMBOL for r in rows)

    def test_it_is_idempotent(self, tmp_path: Path) -> None:
        """A cache rebuild must re-arm it byte for byte, or a diff is noise."""
        first = write_option_canary(tmp_path, CHAIN_TABLE).read_bytes()
        second = write_option_canary(tmp_path, CHAIN_TABLE).read_bytes()
        assert first == second

    def test_it_is_armed_for_both_tables(self, tmp_path: Path) -> None:
        write_option_canary(tmp_path, CHAIN_TABLE)
        write_option_canary(tmp_path, VOLATILITY_TABLE)
        audit = audit_option_root(tmp_path)
        assert set(audit.canary_tables) == {CHAIN_TABLE, VOLATILITY_TABLE}

    def test_loading_it_taints(self, tmp_path: Path) -> None:
        write_option_canary(tmp_path, CHAIN_TABLE)
        with scope(Seal()) as seal:
            load_research(CANARY_SYMBOL, root=tmp_path)
            assert seal.tainted


class TestAudit:
    def test_a_clean_root_reports_its_latest_session(self, tmp_path: Path) -> None:
        chain_csv(tmp_path / CHAIN_TABLE / "SPY.csv", (BEFORE, "2024-08-31"))
        audit = audit_option_root(tmp_path)
        assert audit.clean
        assert audit.latest == date(2024, 8, 31)
        assert audit.rows == 2

    def test_a_contaminated_root_names_the_symbol(self, tmp_path: Path) -> None:
        chain_csv(tmp_path / CHAIN_TABLE / "SPY.csv", (BEFORE, AFTER))
        audit = audit_option_root(tmp_path)
        assert not audit.clean
        assert audit.offenders == ("SPY",)

    def test_the_canary_is_reported_not_counted_as_contamination(self, tmp_path: Path) -> None:
        """It is *supposed* to hold embargoed rows. A tripwire placed where a
        peek cannot happen catches nothing."""
        chain_csv(tmp_path / CHAIN_TABLE / "SPY.csv", (BEFORE,))
        write_option_canary(tmp_path, CHAIN_TABLE)
        audit = audit_option_root(tmp_path)
        assert audit.clean
        assert audit.canary_present

    def test_auditing_is_not_itself_a_peek(self, tmp_path: Path) -> None:
        """Otherwise nobody could ever check whether the root was clean."""
        chain_csv(tmp_path / CHAIN_TABLE / "SPY.csv", (BEFORE, AFTER))
        write_option_canary(tmp_path, CHAIN_TABLE)
        with scope(Seal()) as seal:
            audit_option_root(tmp_path)
            assert not seal.tainted

    def test_a_custom_embargo_moves_the_line(self, tmp_path: Path) -> None:
        chain_csv(tmp_path / CHAIN_TABLE / "SPY.csv", (BEFORE, AFTER))
        later = EMBARGO_START + timedelta(days=365)
        assert audit_option_root(tmp_path, embargo=later).clean
