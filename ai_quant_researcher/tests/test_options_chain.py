"""The option-chain cache. Offline and deterministic, like every other test here.

No socket is opened. ``download_table`` takes a stream opener, so the transfer
is exercised against bytes this file wrote, and ``split_table`` reads a gzip
file this file built. What is being pinned is the part that would fail
silently: a schema that moved, a symbol filter that let the wrong rows through,
a re-run that doubled a cache instead of replacing it.
"""

from __future__ import annotations

import csv
import gzip
from contextlib import contextmanager
from pathlib import Path

import pytest

from aqr.data.options_chain import (
    CHAIN_COLUMNS,
    VOLATILITY_COLUMNS,
    csv_url,
    download_table,
    read_header,
    split_table,
)

CHAIN_HEADER = ",".join(CHAIN_COLUMNS)

CHAIN_BODY = "\n".join(
    [
        CHAIN_HEADER,
        "2019-02-09,AAPL,2019-02-15,170.00,Call,0.50,0.55,0.2705,0.12,0.01,-0.05,0.03,0.01",
        "2019-02-09,SPY,2019-02-15,270.00,Put,1.10,1.15,0.1502,-0.31,0.02,-0.07,0.09,0.02",
        "2019-02-09,LLTC,2019-02-15,60.00,Call,0.20,0.25,0.3100,0.09,0.01,-0.02,0.01,0.00",
        "2026-08-28,AAPL,2026-09-11,250.00,Put,2.00,2.10,0.2200,-0.16,0.01,-0.04,0.08,0.01",
        "",
    ]
)


def gzipped(path: Path, text: str) -> Path:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write(text)
    return path


def rows_in(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class StubStream:
    """A response that yields the bytes it was given, in small chunks."""

    def __init__(self, payload: bytes, chunk: int = 16) -> None:
        self.payload = payload
        self.chunk = chunk

    def iter_bytes(self):  # noqa: ANN201 - matches the ByteStream Protocol
        for start in range(0, len(self.payload), self.chunk):
            yield self.payload[start : start + self.chunk]


def opener_for(payload: bytes, seen: list[str] | None = None):  # noqa: ANN201
    @contextmanager
    def open_stream(url: str):  # noqa: ANN202
        if seen is not None:
            seen.append(url)
        yield StubStream(payload)

    return open_stream


# ------------------------------------------------------------------ #
# the URL
# ------------------------------------------------------------------ #


def test_the_url_names_the_public_repo() -> None:
    assert (
        csv_url("option_chain")
        == "https://www.dolthub.com/csv/post-no-preference/options/master/option_chain"
    )


# ------------------------------------------------------------------ #
# phase one -- moving bytes
# ------------------------------------------------------------------ #


class TestDownload:
    def test_it_writes_what_it_was_given(self, tmp_path: Path) -> None:
        dest = tmp_path / "option_chain.csv.gz"
        result = download_table(
            "option_chain", dest, open_stream=opener_for(CHAIN_BODY.encode())
        )
        assert result.bytes_read == len(CHAIN_BODY.encode())
        with gzip.open(dest, "rt", encoding="utf-8") as handle:
            assert handle.read() == CHAIN_BODY

    def test_a_partial_transfer_leaves_no_usable_file(self, tmp_path: Path) -> None:
        """The service honours no `Range`, so there is no resume.

        A half file that the splitter would happily read is therefore the
        dangerous outcome, and the rename is what rules it out.
        """
        dest = tmp_path / "option_chain.csv.gz"

        @contextmanager
        def dies(url: str):  # noqa: ANN202
            class Failing:
                def iter_bytes(self):  # noqa: ANN202
                    yield b"date,act_symbol\n"
                    raise OSError("connection reset")

            yield Failing()

        with pytest.raises(OSError, match="connection reset"):
            download_table("option_chain", dest, open_stream=dies)
        assert not dest.exists()
        assert list(tmp_path.glob("*.part"))

    def test_progress_is_reported_by_bytes(self, tmp_path: Path) -> None:
        seen: list[int] = []
        download_table(
            "option_chain",
            tmp_path / "x.csv.gz",
            open_stream=opener_for(b"x" * 500),
            on_progress=lambda read, _sec: seen.append(read),
            progress_every=100,
        )
        assert seen and seen == sorted(seen)


# ------------------------------------------------------------------ #
# phase two -- splitting
# ------------------------------------------------------------------ #


class TestSplit:
    def test_it_keeps_only_the_universe(self, tmp_path: Path) -> None:
        source = gzipped(tmp_path / "chain.csv.gz", CHAIN_BODY)
        root = tmp_path / "out"
        result = split_table(
            source, root, symbols=["AAPL", "SPY"], required_columns=CHAIN_COLUMNS
        )
        assert result.rows_read == 4
        assert result.rows_kept == 3
        assert result.symbols == {"AAPL", "SPY"}
        assert sorted(p.name for p in root.glob("*.csv")) == ["AAPL.csv", "SPY.csv"]

    def test_a_symbol_the_universe_did_not_name_is_counted_not_dropped_quietly(
        self, tmp_path: Path
    ) -> None:
        """LLTC is delisted, so its absence from a live universe is correct.

        But "the vendor covers less than we hoped" and "our universe spells
        symbols differently" produce the same small file count and want
        opposite responses, so the skipped names are reported.
        """
        source = gzipped(tmp_path / "chain.csv.gz", CHAIN_BODY)
        result = split_table(
            source, tmp_path / "out", symbols=["AAPL", "SPY"], required_columns=CHAIN_COLUMNS
        )
        assert result.unknown_symbols == {"LLTC"}

    def test_the_header_travels_with_every_symbol(self, tmp_path: Path) -> None:
        source = gzipped(tmp_path / "chain.csv.gz", CHAIN_BODY)
        root = tmp_path / "out"
        split_table(source, root, symbols=["AAPL"], required_columns=CHAIN_COLUMNS)
        rows = rows_in(root / "AAPL.csv")
        assert len(rows) == 2
        assert rows[0]["strike"] == "170.00"
        assert rows[1]["delta"] == "-0.16"

    def test_the_window_is_reported(self, tmp_path: Path) -> None:
        source = gzipped(tmp_path / "chain.csv.gz", CHAIN_BODY)
        result = split_table(
            source, tmp_path / "out", symbols=["AAPL"], required_columns=CHAIN_COLUMNS
        )
        assert (result.first_date, result.last_date) == ("2019-02-09", "2026-08-28")

    def test_rerunning_replaces_rather_than_doubles(self, tmp_path: Path) -> None:
        """A cache that grows every time it is refreshed is a cache that lies."""
        source = gzipped(tmp_path / "chain.csv.gz", CHAIN_BODY)
        root = tmp_path / "out"
        for _ in range(2):
            split_table(source, root, symbols=["AAPL"], required_columns=CHAIN_COLUMNS)
        assert len(rows_in(root / "AAPL.csv")) == 2

    def test_a_small_buffer_still_produces_one_file_per_symbol(self, tmp_path: Path) -> None:
        """Flushing mid-stream must append, not start the file over."""
        source = gzipped(tmp_path / "chain.csv.gz", CHAIN_BODY)
        root = tmp_path / "out"
        split_table(
            source,
            root,
            symbols=["AAPL", "SPY"],
            required_columns=CHAIN_COLUMNS,
            flush_rows=1,
        )
        assert len(rows_in(root / "AAPL.csv")) == 2
        assert len(rows_in(root / "SPY.csv")) == 1

    def test_a_moved_schema_is_refused_rather_than_misread(self, tmp_path: Path) -> None:
        """The failure this check exists for is silent.

        A column inserted between two existing ones shifts every later field one
        place, and every number that results is the right shape and the wrong
        value.
        """
        body = "date,act_symbol,strike\n2019-02-09,AAPL,170.00\n"
        source = gzipped(tmp_path / "chain.csv.gz", body)
        with pytest.raises(ValueError, match="schema has moved"):
            split_table(
                source, tmp_path / "out", symbols=["AAPL"], required_columns=CHAIN_COLUMNS
            )

    def test_a_truncated_line_is_counted_not_fatal(self, tmp_path: Path) -> None:
        body = CHAIN_BODY + "2026-08-28,AAPL,2026-09-11\n"
        source = gzipped(tmp_path / "chain.csv.gz", body)
        result = split_table(
            source, tmp_path / "out", symbols=["AAPL"], required_columns=CHAIN_COLUMNS
        )
        assert result.malformed_rows == 1
        assert result.rows_kept == 2

    def test_an_empty_file_is_not_an_empty_universe(self, tmp_path: Path) -> None:
        source = gzipped(tmp_path / "chain.csv.gz", "")
        with pytest.raises(ValueError, match="empty"):
            split_table(
                source, tmp_path / "out", symbols=["AAPL"], required_columns=CHAIN_COLUMNS
            )

    def test_symbols_are_matched_case_insensitively(self, tmp_path: Path) -> None:
        source = gzipped(tmp_path / "chain.csv.gz", CHAIN_BODY)
        result = split_table(
            source, tmp_path / "out", symbols=["aapl"], required_columns=CHAIN_COLUMNS
        )
        assert result.symbols == {"AAPL"}


# ------------------------------------------------------------------ #
# volatility_history -- the table IV rank comes from
# ------------------------------------------------------------------ #


VOL_BODY = "\n".join(
    [
        "date,act_symbol,hv_current,iv_current,iv_year_high,iv_year_low",
        "2026-08-28,SPY,0.0910,0.1158,0.2648,0.1090",
        "2026-08-28,AAPL,0.2100,0.2400,0.5000,0.1500",
        "",
    ]
)


def test_volatility_history_splits_and_carries_the_iv_rank_inputs(tmp_path: Path) -> None:
    """`iv_current` against its own year range is IV rank, which specs/07 D3
    conditions on and which no live feed on this account can produce."""
    source = gzipped(tmp_path / "vol.csv.gz", VOL_BODY)
    root = tmp_path / "vol"
    result = split_table(
        source, root, symbols=["SPY"], required_columns=VOLATILITY_COLUMNS
    )
    assert result.rows_kept == 1
    row = rows_in(root / "SPY.csv")[0]
    low, high, now = (
        float(row["iv_year_low"]),
        float(row["iv_year_high"]),
        float(row["iv_current"]),
    )
    assert round((now - low) / (high - low) * 100, 1) == 4.4


def test_read_header_does_not_consume_the_body(tmp_path: Path) -> None:
    source = gzipped(tmp_path / "chain.csv.gz", CHAIN_BODY)
    assert read_header(source) == list(CHAIN_COLUMNS)
    result = split_table(
        source, tmp_path / "out", symbols=["AAPL"], required_columns=CHAIN_COLUMNS
    )
    assert result.rows_kept == 2
