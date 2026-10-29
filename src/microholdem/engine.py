"""No-Limit Hold'em hand engine.

One :func:`play_hand` call plays a complete hand for 2 to 10 seats: blinds and
antes, four betting rounds with full raise and short all-in semantics, uncalled
bet return, side pots, and showdown with the C++ evaluator. Every bot action
is timed with ``perf_counter_ns`` and a bot that exceeds the per-action budget
(1 ms by default, matching the competition rule) is folded, or checked when
checking is free. Exceptions raised by a bot are handled the same way.

The engine is deterministic for a given ``random.Random`` instance.
"""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol

from . import _core

DEFAULT_TIME_LIMIT_NS = 1_000_000


class ActionKind(IntEnum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    RAISE = 3  # bet or raise; ``amount`` is the total bet for the street ("raise to")


class Street(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    amount: int = 0

    @staticmethod
    def fold() -> Action:
        return FOLD

    @staticmethod
    def check() -> Action:
        return CHECK

    @staticmethod
    def call() -> Action:
        return CALL

    @staticmethod
    def raise_to(amount: int) -> Action:
        return Action(ActionKind.RAISE, int(amount))

    def __str__(self) -> str:
        if self.kind == ActionKind.RAISE:
            return f"raise_to {self.amount}"
        return self.kind.name.lower()


FOLD = Action(ActionKind.FOLD)
CHECK = Action(ActionKind.CHECK)
CALL = Action(ActionKind.CALL)


@dataclass(slots=True)
class Observation:
    """Everything a bot may see when asked to act."""

    seat: int
    hole: tuple[int, int]
    board: tuple[int, ...]
    street: int
    pot: int  # all chips committed by everyone so far, including this street
    to_call: int  # chips needed to call (already capped by the stack)
    min_raise_to: int  # smallest legal total bet when raising
    max_raise_to: int  # bet + stack
    can_raise: bool
    stacks: tuple[int, ...]  # chips behind, per seat
    bets: tuple[int, ...]  # chips committed this street, per seat
    in_hand: tuple[bool, ...]
    button: int
    small_blind: int
    big_blind: int
    ante: int
    n_in_hand: int
    hand_number: int
    history: tuple[tuple[int, int, int, int], ...]  # (street, seat, kind, amount); see ActionRecord.amount
    payouts: tuple[float, ...]  # tournament payout structure for ICM, empty if none

    @property
    def stack(self) -> int:
        return self.stacks[self.seat]

    @property
    def bet(self) -> int:
        return self.bets[self.seat]

    @property
    def n_seats(self) -> int:
        return len(self.stacks)

    @property
    def effective_stack(self) -> int:
        """Smallest of hero's stack and the largest opposing stack (chips behind + bets)."""
        mine = self.stacks[self.seat] + self.bets[self.seat]
        others = [self.stacks[i] + self.bets[i] for i in range(len(self.stacks)) if i != self.seat and self.in_hand[i]]
        return min(mine, max(others)) if others else mine


class Bot(Protocol):
    """Interface every bot implements. ``act`` must return within the time limit."""

    name: str

    def reset(self, seat: int, n_seats: int, seed: int) -> None: ...

    def act(self, obs: Observation) -> Action: ...


@dataclass(slots=True)
class ActionRecord:
    """One legalized action. ``amount`` is the raise-to total for raises, the chips put in for calls, else 0."""

    street: int
    seat: int
    kind: int
    amount: int
    latency_ns: int
    timed_out: bool
    errored: bool


@dataclass(slots=True)
class HandResult:
    hand_number: int
    button: int
    hole: list[tuple[int, int] | None]
    board: list[int]
    start_stacks: list[int]
    end_stacks: list[int]
    actions: list[ActionRecord]
    showdown: bool
    winners: list[int]
    sb_seat: int | None = None  # None when the small blind was dead
    bb_seat: int = -1

    @property
    def net(self) -> list[int]:
        return [e - s for s, e in zip(self.start_stacks, self.end_stacks, strict=True)]


@dataclass(slots=True)
class BotStats:
    actions: int = 0
    timeouts: int = 0
    errors: int = 0
    illegal: int = 0
    latencies_ns: list[int] = field(default_factory=list)


@dataclass(slots=True)
class HandConfig:
    small_blind: int
    big_blind: int
    ante: int = 0
    time_limit_ns: int = DEFAULT_TIME_LIMIT_NS
    payouts: tuple[float, ...] = ()
    record_latency: bool = True


_IN_HAND_ANY = object()


def next_live(seat: int, alive: list[bool]) -> int:
    n = len(alive)
    i = (seat + 1) % n
    while not alive[i]:
        i = (i + 1) % n
    return i


def play_hand(
    bots: Sequence[Bot],
    stacks: list[int],
    button: int,
    cfg: HandConfig,
    rng: random.Random,
    hand_number: int,
    stats: list[BotStats] | None = None,
    *,
    sb_seat: int | None = None,
    bb_seat: int | None = None,
) -> HandResult:
    """Play one hand in place (``stacks`` is updated) and return the record.

    By default the blinds are derived from ``button``, which must be a live seat. Tournament
    callers may pass ``bb_seat`` (and ``sb_seat``) explicitly for the dead-button rule: the button
    may then sit on a busted seat and ``sb_seat`` may be ``None`` when the small blind is dead.
    """
    n = len(stacks)
    alive = [s > 0 for s in stacks]
    players = [i for i in range(n) if alive[i]]
    if len(players) < 2:
        raise ValueError("a hand needs at least two players with chips")
    if not 0 <= button < n:
        raise ValueError("button out of range")
    if bb_seat is None:
        if not alive[button]:
            raise ValueError("button must be on a live seat")
    elif not alive[bb_seat] or (sb_seat is not None and not alive[sb_seat]) or sb_seat == bb_seat:
        raise ValueError("blind seats must be distinct live seats")
    start_stacks = list(stacks)

    in_hand = list(alive)
    all_in = [False] * n
    bet = [0] * n
    total = [0] * n
    hist: list[tuple[int, int, int, int]] = []
    records: list[ActionRecord] = []

    def post(i: int, amount: int) -> None:
        a = min(amount, stacks[i])
        stacks[i] -= a
        bet[i] += a
        total[i] += a
        if stacks[i] == 0:
            all_in[i] = True

    # Blinds and antes.
    if bb_seat is None:
        if len(players) == 2:
            sb_seat, bb_seat = button, next_live(button, alive)
        else:
            sb_seat = next_live(button, alive)
            bb_seat = next_live(sb_seat, alive)
    if cfg.ante > 0:
        for i in players:
            post(i, cfg.ante)
    if sb_seat is not None:
        post(sb_seat, cfg.small_blind)
    post(bb_seat, cfg.big_blind)

    # Deal.
    deck = list(range(52))
    rng.shuffle(deck)
    pos = 0
    hole: list[tuple[int, int] | None] = [None] * n
    for i in players:
        hole[i] = (deck[pos], deck[pos + 1])
        pos += 2
    board_cards = deck[pos : pos + 5]
    board: list[int] = []

    n_in_hand = len(players)
    # A short big blind does not lower the amount the others must call (Robert's Rules, blinds).
    current_bet = max(max(bet), cfg.ante + cfg.big_blind)
    min_raise_inc = cfg.big_blind
    acted_at: dict[int, int] = {}  # bet level at which each seat last acted since the last full raise
    payouts = cfg.payouts
    time_limit = cfg.time_limit_ns

    def observe(i: int, street: int) -> Observation:
        to_call = min(current_bet - bet[i], stacks[i])
        # A player who already acted may raise again only if the bet has grown by at least a full
        # raise since then (a short all-in alone does not reopen the action; TDA Rule 49-A).
        can_raise = stacks[i] > to_call and (i not in acted_at or current_bet - acted_at[i] >= min_raise_inc)
        max_to = bet[i] + stacks[i]
        min_to = min(current_bet + min_raise_inc, max_to)
        return Observation(
            seat=i,
            hole=hole[i],  # type: ignore[arg-type]
            board=tuple(board),
            street=street,
            pot=sum(total),
            to_call=to_call,
            min_raise_to=min_to if can_raise else 0,
            max_raise_to=max_to if can_raise else 0,
            can_raise=can_raise,
            stacks=tuple(stacks),
            bets=tuple(bet),
            in_hand=tuple(in_hand),
            button=button,
            small_blind=cfg.small_blind,
            big_blind=cfg.big_blind,
            ante=cfg.ante,
            n_in_hand=n_in_hand,
            hand_number=hand_number,
            history=tuple(hist),
            payouts=payouts,
        )

    def ask(i: int, obs: Observation) -> tuple[Action, int, bool, bool]:
        bot = bots[i]
        t0 = time.perf_counter_ns()
        errored = False
        try:
            action = bot.act(obs)
            # A return value that is not a well-formed Action is an error: the bot is folded.
            if not isinstance(action, Action) or not isinstance(action.kind, ActionKind):
                raise TypeError("act() must return an Action")
            if action.kind == ActionKind.RAISE:
                action = Action(ActionKind.RAISE, int(action.amount))
        except Exception:
            action = FOLD
            errored = True
        dt = time.perf_counter_ns() - t0
        timed_out = dt > time_limit
        if timed_out:
            action = FOLD
        return action, dt, timed_out, errored

    def legalize(i: int, obs: Observation, action: Action) -> tuple[Action, bool]:
        """Map any action onto a legal one. Returns (action, was_illegal)."""
        kind = action.kind
        if kind == ActionKind.FOLD:
            return (CHECK, False) if obs.to_call == 0 else (FOLD, False)
        if kind == ActionKind.CHECK:
            return (CHECK, False) if obs.to_call == 0 else (FOLD, True)
        if kind == ActionKind.CALL:
            return (CHECK, False) if obs.to_call == 0 else (CALL, False)
        # RAISE
        if not obs.can_raise:
            return (CHECK, True) if obs.to_call == 0 else (CALL, True)
        amt = int(action.amount)
        if amt <= current_bet:
            return (CHECK, True) if obs.to_call == 0 else (CALL, True)
        lo, hi = obs.min_raise_to, obs.max_raise_to
        if amt > hi:
            return Action(ActionKind.RAISE, hi), True
        if amt < lo:
            return Action(ActionKind.RAISE, lo), True
        return Action(ActionKind.RAISE, amt), False

    def betting_round(first: int, street: int) -> None:
        nonlocal current_bet, min_raise_inc, n_in_hand
        pending = {i for i in range(n) if in_hand[i] and not all_in[i]}
        # Nobody has to act if no live player faces a bet and at most one can act.
        if len(pending) <= 1:
            only = next(iter(pending), None)
            if only is None or bet[only] >= current_bet:
                return
        cursor = first
        while pending and n_in_hand > 1:
            while cursor not in pending:
                cursor = (cursor + 1) % n
            i = cursor
            # Nothing to decide once everyone else is all-in or out and hero already covers every bet.
            if not any(in_hand[j] and not all_in[j] for j in range(n) if j != i) and bet[i] >= max(
                (bet[j] for j in range(n) if j != i), default=0
            ):
                break
            obs = observe(i, street)
            raw, dt, timed_out, errored = ask(i, obs)
            action, illegal = legalize(i, obs, raw)
            if stats is not None:
                st = stats[i]
                st.actions += 1
                st.timeouts += timed_out
                st.errors += errored
                st.illegal += illegal
                if cfg.record_latency:
                    st.latencies_ns.append(dt)
            kind = action.kind
            if kind == ActionKind.FOLD:
                in_hand[i] = False
                n_in_hand -= 1
            elif kind == ActionKind.CALL:
                post(i, obs.to_call)
            elif kind == ActionKind.RAISE:
                target = action.amount
                inc = target - current_bet
                post(i, target - bet[i])
                current_bet = bet[i]
                if inc >= min_raise_inc:
                    min_raise_inc = inc
                    pending = {j for j in range(n) if in_hand[j] and not all_in[j] and j != i}
                    acted_at.clear()
                else:
                    # Short all-in: players who already acted must respond but may only raise once
                    # the short raises since their action add up to a full raise (see observe()).
                    pending = {j for j in range(n) if in_hand[j] and not all_in[j] and j != i and bet[j] < current_bet}
            acted_at[i] = current_bet
            pending.discard(i)
            amount = action.amount if kind == ActionKind.RAISE else (obs.to_call if kind == ActionKind.CALL else 0)
            hist.append((street, i, int(kind), amount))
            records.append(ActionRecord(street, i, int(kind), amount, dt, timed_out, errored))
            cursor = (cursor + 1) % n

    def return_uncalled() -> None:
        top = max(range(n), key=lambda j: bet[j])
        second = max((bet[j] for j in range(n) if j != top), default=0)
        extra = bet[top] - second
        if extra > 0:
            bet[top] -= extra
            total[top] -= extra
            stacks[top] += extra
            if stacks[top] > 0:
                all_in[top] = False

    def new_street() -> None:
        nonlocal current_bet, min_raise_inc
        for j in range(n):
            bet[j] = 0
        current_bet = 0
        min_raise_inc = cfg.big_blind
        acted_at.clear()

    # Preflop.
    first = next_live(bb_seat, alive)  # heads-up this is the small blind, which is the button
    betting_round(first, Street.PREFLOP)
    return_uncalled()
    showdown = False
    winners: list[int] = []
    streets = ((Street.FLOP, 3), (Street.TURN, 4), (Street.RIVER, 5))
    for street, n_board in streets:
        if n_in_hand <= 1:
            break
        new_street()
        board = board_cards[:n_board]
        live_actors = sum(1 for j in range(n) if in_hand[j] and not all_in[j])
        if live_actors >= 2:
            betting_round(next_live(button, in_hand), street)
            return_uncalled()
    board = board_cards[:5] if n_in_hand > 1 else board

    # Award pots.
    if n_in_hand == 1:
        w = next(j for j in range(n) if in_hand[j])
        stacks[w] += sum(total)
        winners = [w]
    else:
        showdown = True
        contenders = [j for j in range(n) if in_hand[j]]
        values = {j: _core.evaluate([*hole[j], *board]) for j in contenders}  # type: ignore[misc]
        levels = sorted({total[j] for j in contenders})
        prev = 0
        order = [(button + 1 + k) % n for k in range(n)]
        for level in levels:
            pot = sum(max(0, min(total[j], level) - prev) for j in range(n))
            eligible = [j for j in contenders if total[j] >= level]
            best = max(values[j] for j in eligible)
            pot_winners = [j for j in order if j in eligible and values[j] == best]
            share, rem = divmod(pot, len(pot_winners))
            for k, j in enumerate(pot_winners):
                stacks[j] += share + (1 if k < rem else 0)
            prev = level
            for j in pot_winners:
                if j not in winners:
                    winners.append(j)

    if sum(stacks) != sum(start_stacks):
        raise AssertionError("chip conservation violated")
    return HandResult(
        hand_number=hand_number,
        button=button,
        hole=hole,
        board=list(board),
        start_stacks=start_stacks,
        end_stacks=list(stacks),
        actions=records,
        showdown=showdown,
        winners=winners,
        sb_seat=sb_seat,
        bb_seat=bb_seat,
    )
