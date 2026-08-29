"""Users and watchlists — ADR 0012's spec delta to specs/04-domain-model.md."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from alphagate.core.accounts import User, Watchlist, WatchlistItem
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import UserId, WatchlistId, ticker

MOMENT = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
USER: UserId = UserId("u-1")
AAPL = ticker("AAPL")
MSFT = ticker("MSFT")


class TestUser:
    def test_a_user_carries_a_creation_instant_in_utc(self) -> None:
        assert User(id=USER, display_name="Jo", created_at=MOMENT).created_at.tzinfo is not None

    def test_a_naive_timestamp_is_refused(self) -> None:
        with pytest.raises(InvariantViolation):
            User(id=USER, display_name="Jo", created_at=datetime(2026, 8, 13, 12, 0))  # noqa: DTZ001

    @pytest.mark.parametrize("name", ["", "   "])
    def test_a_user_needs_a_name(self, name: str) -> None:
        with pytest.raises(InvariantViolation, match="display_name"):
            User(id=USER, display_name=name, created_at=MOMENT)

    def test_a_user_needs_an_id(self) -> None:
        with pytest.raises(InvariantViolation, match="id"):
            User(id=UserId(""), display_name="Jo", created_at=MOMENT)


class TestWatchlist:
    def test_items_come_back_in_the_users_order(self) -> None:
        watchlist = Watchlist(
            id=WatchlistId("w-1"),
            user_id=USER,
            name="Core",
            items=(
                WatchlistItem(symbol=MSFT, position=1),
                WatchlistItem(symbol=AAPL, position=0),
            ),
        )
        assert watchlist.symbols == (AAPL, MSFT)

    def test_a_repeated_symbol_is_refused(self) -> None:
        with pytest.raises(InvariantViolation, match="twice"):
            Watchlist(
                id=WatchlistId("w-1"),
                user_id=USER,
                name="Core",
                items=(
                    WatchlistItem(symbol=AAPL, position=0),
                    WatchlistItem(symbol=AAPL, position=1),
                ),
            )

    def test_two_items_at_one_position_are_refused(self) -> None:
        with pytest.raises(InvariantViolation, match="position"):
            Watchlist(
                id=WatchlistId("w-1"),
                user_id=USER,
                name="Core",
                items=(
                    WatchlistItem(symbol=AAPL, position=0),
                    WatchlistItem(symbol=MSFT, position=0),
                ),
            )

    @pytest.mark.parametrize("name", ["", " "])
    def test_a_watchlist_needs_a_name(self, name: str) -> None:
        with pytest.raises(InvariantViolation, match="name"):
            Watchlist(id=WatchlistId("w-1"), user_id=USER, name=name)

    def test_a_negative_position_is_refused(self) -> None:
        with pytest.raises(InvariantViolation, match="position"):
            WatchlistItem(symbol=AAPL, position=-1)

    def test_an_empty_watchlist_is_valid(self) -> None:
        assert Watchlist(id=WatchlistId("w-1"), user_id=USER, name="Core").symbols == ()
