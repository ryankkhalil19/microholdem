"""Heads-up agent with a per-opponent frequency model.

:class:`HeadsUpAgent` plays two-seat tables; with more seats it defers to :class:`MicroAgent`.
It keeps the parent's preflop opening and defending ranges and changes three things.

Pot odds use only the chips that can be won.
    :func:`callable_view` caps every bet at what the acting seat can match, so a shove that
    covers it is priced on the callable part, not on the whole stack pushed forward.

A re-raise is answered by price.
    The parent folds almost every hand to a re-raise. Here the re-raiser is given the top slice
    of hands matching how often this opponent has re-raised (:class:`OpponentModel`), and the
    agent continues when ``equity * realisation >= pot odds``.

Postflop decisions condition on the opponent's own frequencies.
    The opponent's range is laid out by strength and reweighted by what it did this hand: a bet
    from a seat that bets a fraction ``a`` of the time keeps the top of its range and scales the
    rest down, a check does the reverse, and a call of our bet drops the hands that would have
    folded. The agent's equity against the whole range becomes a win probability against that
    reweighted range, compared with the pot odds. It therefore calls a frequent bettor lightly
    and a rare bettor tightly, and sizes its own value bets and bluffs by how often the
    opponent has folded.

Every decision is a few table reads, a 50-cell loop and at most one C++ equity call.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import numpy.typing as npt

from . import _core
from .agent import MicroAgent
from .cards import class_index
from .engine import CALL, CHECK, FOLD, Action, ActionKind, Observation, Street
from .equity import Range

__all__ = ["HeadsUpAgent", "OpponentModel", "callable_view"]

History = Sequence[tuple[int, int, int, int]]  # (street, seat, kind, amount), as in Observation.history

_RAISE = int(ActionKind.RAISE)
_CALL = int(ActionKind.CALL)
_FOLD = int(ActionKind.FOLD)

SHORT_BB = 25.0  # at or below this effective stack a re-raise is a shove
REALISATION_IP = 0.92  # share of preflop equity realised when calling in position
REALISATION_OOP = 0.76  # ... and out of position
RERAISE_EQUITY = 0.60  # re-raise for value at this equity against the raiser's range
JAM_OVERSIZE_BB = 16.0  # a preflop raise to at least this many big blinds is treated as a shove
MARGIN_FLOP = 0.04  # extra equity needed to call a flop bet: more bets may follow
MARGIN_TURN = 0.02
_BINS = 50  # resolution of the range-strength grid in HeadsUpAgent._standing


def callable_view(obs: Observation) -> Observation:
    """``obs`` with every bet capped at what the acting seat can match.

    ``Observation.pot`` counts an all-in in full, including the part a shorter stack cannot call
    and the bettor gets back. Pot odds taken from it make calling a covering shove look far
    cheaper than it is.
    """
    limit = obs.bet + obs.stack
    excess = sum(max(0, b - limit) for b in obs.bets)
    if excess == 0:
        return obs
    return replace(obs, pot=obs.pot - excess, bets=tuple(min(b, limit) for b in obs.bets))


class OpponentModel:
    """Smoothed frequencies of one heads-up opponent, updated once per finished hand.

    Each statistic is ``(prior_mean * prior_weight + hits) / (prior_weight + opportunities)``:
    with no data it is the prior, and the prior fades as hands accumulate.
    """

    # statistic: (prior mean, prior weight in opportunities)
    PRIORS: dict[str, tuple[float, float]] = {
        "sb_open": (0.70, 8.0),  # raises first in from the small blind
        "sb_limp": (0.12, 8.0),  # limps first in from the small blind
        "bb_fold": (0.35, 8.0),  # folds the big blind to an open
        "bb_3bet": (0.12, 10.0),  # re-raises an open, shoves included
        "bb_jam_deep": (0.03, 12.0),  # shoves over an open with more than 25 big blinds
        "bb_jam_short": (0.15, 8.0),  # shoves over an open with 25 big blinds or fewer
        "fold_to_3bet": (0.50, 6.0),  # folds its open to a re-raise
        "bet": (0.50, 5.0),  # bets postflop when first to act or checked to
        "raise": (0.12, 5.0),  # raises a postflop bet
        "fold_to_bet": (0.42, 5.0),  # folds to a postflop bet or raise
    }

    def __init__(self) -> None:
        self.hits: dict[str, float] = dict.fromkeys(self.PRIORS, 0.0)
        self.seen: dict[str, float] = dict.fromkeys(self.PRIORS, 0.0)
        self.hands = 0

    def rate(self, stat: str) -> float:
        mean, weight = self.PRIORS[stat]
        return (mean * weight + self.hits[stat]) / (weight + self.seen[stat])

    def _count(self, stat: str, hit: bool) -> None:
        self.seen[stat] += 1.0
        if hit:
            self.hits[stat] += 1.0

    def observe_hand(self, history: History, hero: int, button: int, big_blind: int, effective_stack: int) -> None:
        """Count one finished heads-up hand. ``history`` is every action; ``button`` posted the small blind."""
        if big_blind <= 0:
            return
        self.hands += 1
        villain = 1 - hero
        shove_size = min(JAM_OVERSIZE_BB * big_blind, 0.8 * effective_stack)
        pre = [(seat, kind, amount) for st, seat, kind, amount in history if st == 0]
        if pre:
            seat, kind, amount = pre[0]
            jam = kind == _RAISE and amount >= shove_size
            if seat == villain and button == villain and not jam:
                self._count("sb_open", kind == _RAISE)
                self._count("sb_limp", kind == _CALL)
                if kind == _RAISE and len(pre) > 2 and pre[1][:2] == (hero, _RAISE):
                    self._count("fold_to_3bet", pre[2][1] == _FOLD)
            elif seat == hero and button == hero and kind == _RAISE and not jam and len(pre) > 1:
                _s, reply, reply_amount = pre[1]
                self._count("bb_fold", reply == _FOLD)
                self._count("bb_3bet", reply == _RAISE)
                short = effective_stack <= SHORT_BB * big_blind
                self._count("bb_jam_short" if short else "bb_jam_deep", reply == _RAISE and reply_amount >= shove_size)
        for street in (1, 2, 3):
            bettor = -1  # seat whose bet or raise is outstanding
            for st, seat, kind, _amount in history:
                if st != street:
                    continue
                if seat == villain:
                    if bettor == hero:
                        self._count("fold_to_bet", kind == _FOLD)
                        self._count("raise", kind == _RAISE)
                    elif bettor == -1:
                        self._count("bet", kind == _RAISE)
                if kind == _RAISE:
                    bettor = seat


class HeadsUpAgent(MicroAgent):
    """See the module docstring. Feed finished hands to :meth:`observe_hand` to adapt to the opponent."""

    name = "micro-hu"

    def __init__(self, seed: int = 0, **kwargs: object) -> None:
        super().__init__(seed, **kwargs)  # type: ignore[arg-type]
        self.model = OpponentModel()
        self._slices: dict[tuple[int, int], tuple[_core.CompiledRange, npt.NDArray[np.float64]]] = {}

    def new_opponent(self) -> None:
        """Forget what was learned; call when the seat across the table changes."""
        self.model = OpponentModel()

    def observe_hand(self, history: History, hero: int, button: int, big_blind: int, effective_stack: int) -> None:
        self.model.observe_hand(history, hero, button, big_blind, effective_stack)

    # ------------------------------------------------------------------
    # ranges
    # ------------------------------------------------------------------
    def _slice(self, lo: float, hi: float) -> tuple[_core.CompiledRange, npt.NDArray[np.float64]]:
        """Hands ranked between the top ``lo`` and the top ``hi`` fraction, on a 5% grid."""
        a = max(0, min(19, round(lo * 20)))
        b = max(a + 1, min(20, round(hi * 20)))
        found = self._slices.get((a, b))
        if found is None:
            order = [int(x) for x in self.tb.order_hu]
            weights = Range.top_fraction(order, b / 20).weights - Range.top_fraction(order, a / 20).weights
            weights = np.clip(weights, 0.0, 1.0)
            found = (Range(weights).compile(), weights)
            self._slices[(a, b)] = found
        return found

    def _villain_slice(self, obs: Observation) -> tuple[float, float]:
        """The opponent's preflop range as a slice of the ranking, from its line this hand."""
        m = self.model
        hero = obs.seat
        villain = 1 - hero
        pre = [(seat, kind) for st, seat, kind, _a in obs.history if st == 0]
        villain_raises = sum(1 for seat, kind in pre if seat == villain and kind == _RAISE)
        hero_raises = sum(1 for seat, kind in pre if seat == hero and kind == _RAISE)
        open_rate = min(1.0, max(0.15, m.rate("sb_open")))
        if obs.button == villain:  # the opponent was the small blind
            if pre and pre[0] == (villain, _CALL):  # it limped
                if villain_raises:
                    return 0.0, 0.08
                top = min(0.9, open_rate)
                return top, min(1.0, top + max(0.1, m.rate("sb_limp")) * (0.7 if hero_raises else 1.0))
            if villain_raises >= 2:
                return 0.0, 0.05
            if hero_raises:
                return 0.0, max(0.1, open_rate * (1.0 - m.rate("fold_to_3bet")))  # called a re-raise
            return 0.0, open_rate
        three = min(0.6, max(0.03, m.rate("bb_3bet")))
        if villain_raises >= 2:
            return 0.0, 0.05
        if villain_raises == 1:
            return 0.0, three
        if hero_raises == 0:
            return 0.0, 1.0  # checked its option
        return three, max(three + 0.1, 1.0 - m.rate("bb_fold"))  # called an open

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------
    def act(self, obs: Observation) -> Action:
        if len(obs.stacks) != 2:
            return super().act(obs)
        t0 = time.perf_counter_ns()
        self.decisions += 1
        obs = callable_view(obs)
        if obs.street != Street.PREFLOP:
            action = self._postflop_hu(obs, t0)
        elif any(st == 0 and seat == obs.seat and kind == _RAISE for st, seat, kind, _a in obs.history):
            action = self._facing_reraise(obs)
        else:
            action = self._preflop(obs, t0)
        self.last_latency_ns = time.perf_counter_ns() - t0
        return action

    # ------------------------------------------------------------------
    # preflop: our raise was re-raised
    # ------------------------------------------------------------------
    def _facing_reraise(self, obs: Observation) -> Action:
        m = self.model
        bb = obs.big_blind
        villain = 1 - obs.seat
        cls = class_index(obs.hole[0], obs.hole[1])
        eff_bb = obs.effective_stack / bb
        raises = [(seat, amount) for st, seat, kind, amount in obs.history if st == 0 and kind == _RAISE]
        on_button = obs.button == obs.seat
        all_in = obs.stacks[villain] == 0 or obs.to_call >= obs.stack
        shove = all_in or raises[-1][1] >= JAM_OVERSIZE_BB * bb
        opened = on_button and raises[0][0] == obs.seat and len(raises) == 2
        if not opened:
            fraction = 0.05  # a limp-raise, or a raise over our re-raise
        elif shove:
            fraction = max(0.03, m.rate("bb_jam_short" if eff_bb <= SHORT_BB else "bb_jam_deep"))
        else:
            fraction = min(0.6, max(0.04, m.rate("bb_3bet")))
        eq = self.tb.equity_vs_class_range(cls, self._slice(0.0, fraction)[1])
        need = obs.to_call / (obs.pot + obs.to_call)
        if all_in:
            # Heads-up to the last chip, chips are equity: call whenever the price is there.
            return CALL if eq >= need + 0.01 else FOLD

        short = eff_bb <= SHORT_BB
        if obs.can_raise and eq >= (RERAISE_EQUITY - 0.04 if short else RERAISE_EQUITY):
            if short or shove:
                return self._shove(obs)
            action = self._raise_multiple(obs, 2.6 if on_button else 3.3)
            # A raise that puts in a third of the stack leaves no room to fold: shove instead.
            return self._shove(obs) if action.amount - obs.bet >= 0.33 * obs.stack else action
        if eq * (REALISATION_IP if on_button else REALISATION_OOP) >= need:
            if obs.to_call >= 0.33 * obs.stack:  # calling commits the stack: shove or fold
                return self._shove(obs) if eq >= 0.5 else FOLD
            return CALL
        return FOLD

    # ------------------------------------------------------------------
    # postflop
    # ------------------------------------------------------------------
    def _standing(self, obs: Observation, eq: float) -> float:
        """Win probability against the opponent's range as reweighted by this hand's postflop actions.

        The opponent's range is laid out by strength on a grid, weakest first; ``eq``, the equity
        against the whole range, is read as the share of it the agent beats. A seat that bets with
        frequency ``a`` is modelled as betting the top ``a * (1 - bluffs)`` of its range for value
        every time and anything weaker as a bluff with a fixed probability, so each bet scales the
        weaker part down and repeated bets compound. A check scales the value part down instead,
        and a call of our bet removes the weakest hands, which would have folded.
        """
        m = self.model
        hero = obs.seat
        bet_rate = min(0.95, max(0.08, m.rate("bet")))
        raise_rate = min(0.6, max(0.03, m.rate("raise")))
        fold_rate = min(0.9, max(0.05, m.rate("fold_to_bet")))
        bet_value = bet_rate * (1.0 - min(0.5, 0.10 + 0.45 * bet_rate))
        bet_bluff = (bet_rate - bet_value) / (1.0 - bet_value)
        raise_value = raise_rate * 0.8
        raise_bluff = (raise_rate - raise_value) / (1.0 - raise_value)
        w = [1.0] * _BINS
        street = 0
        bettor = -1
        for st, seat, kind, _a in obs.history:
            if st == 0:
                continue
            if st != street:
                street, bettor = st, -1
            if seat != hero:
                if kind == _RAISE:
                    value, bluff = (raise_value, raise_bluff) if bettor == hero else (bet_value, bet_bluff)
                    for i in range(int((1.0 - value) * _BINS)):
                        w[i] *= bluff
                elif kind == _CALL and bettor == hero:
                    drop = fold_rate * sum(w)
                    for i in range(_BINS):
                        if w[i] >= drop:
                            w[i] -= drop
                            break
                        drop -= w[i]
                        w[i] = 0.0
                elif bettor == -1:  # checked with the option to bet
                    cut = int((1.0 - bet_value) * _BINS)
                    for i in range(_BINS):
                        w[i] *= (1.0 - bet_bluff) if i < cut else 0.3
            if kind == _RAISE:
                bettor = seat
        total = sum(w)
        if total <= 0.0:
            return eq
        x = min(1.0, max(0.0, eq)) * _BINS
        whole = int(x)
        below = sum(w[:whole]) + (w[whole] * (x - whole) if whole < _BINS else 0.0)
        return below / total

    def _postflop_hu(self, obs: Observation, t0: int) -> Action:
        lo, hi = self._villain_slice(obs)
        eq = self._equity(obs, t0, [self._slice(lo, hi)[0]])
        q = self._standing(obs, eq)
        fold_rate = self.model.rate("fold_to_bet")
        river = obs.street == Street.RIVER

        if obs.to_call == 0:
            if not obs.can_raise:
                return CHECK
            # Value: ahead of half the hands that call. A seat that rarely folds calls with more.
            if q >= min(0.80, max(0.56, 1.0 - (1.0 - fold_rate) / 2.0)):
                return self._bet_pot_fraction(obs, 0.75 if fold_rate < 0.30 else 0.62)
            # Bluff: a half-pot bet shows a profit once the opponent folds more than a third of the time.
            weak = q < (0.20 if river else 0.35)
            if weak and self.rng.random() < min(0.9, max(0.0, 3.0 * (fold_rate - 0.30))):
                return self._bet_pot_fraction(obs, 0.5)
            # A middling hand can take the pot too, more often the more the opponent folds.
            if not weak and not river and self.rng.random() < min(0.85, max(0.0, 2.5 * (fold_rate - 0.33))):
                return self._bet_pot_fraction(obs, 0.4)
            return CHECK

        need = obs.to_call / (obs.pot + obs.to_call)
        if obs.can_raise and q >= 0.80 and eq >= 0.85:
            action = self._raise_multiple(obs, 2.7)
            return self._shove(obs) if action.amount - obs.bet >= 0.45 * obs.stack else action
        margin = 0.0 if river else (MARGIN_FLOP if obs.street == Street.FLOP else MARGIN_TURN)
        if obs.to_call >= 0.5 * obs.stack:
            margin += 0.02
        return CALL if q >= need + margin else FOLD
