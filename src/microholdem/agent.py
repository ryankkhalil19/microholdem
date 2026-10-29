"""The sub-millisecond decision agent.

Decision structure
    preflop, short effective stacks (<= 20 BB)
        first-in: CFR+ push/fold tables; with a payout structure the jam is
        re-evaluated in ICM dollars using the solved calling ranges.
        facing an all-in: equity versus the solved jam range (class matrix),
        chip EV or ICM EV depending on whether payouts are known.
    preflop, deep
        position-indexed opening, three-bet and calling thresholds over the
        class ranking (rank fraction by combo mass).
    postflop
        Monte Carlo equity against per-opponent weighted ranges inferred from
        their preflop line; sample count is chosen from the remaining time
        budget using a calibration made at construction time.
        heads-up river: bucketed CFR strategy (check/bet/raise mixes).
        otherwise: pot-odds thresholds with street margins and an ICM risk
        premium for stack-committing calls.

Every path is table lookups plus at most one C++ call, so the per-action
latency is dominated by Python overhead and the bounded equity sample.
"""

from __future__ import annotations

import gc
import random
import time
from typing import Any

import numpy as np
import numpy.typing as npt

from . import _core
from .cards import class_index
from .engine import CALL, CHECK, FOLD, Action, ActionKind, Observation, Street
from .equity import Range
from .tables import Tables, tables

RANGE_FRACTIONS: tuple[float, ...] = (0.04, 0.07, 0.10, 0.15, 0.20, 0.30, 0.45, 0.65, 1.0)
PUSHFOLD_MAX_BB = 20.0

# Opening thresholds (rank fraction, 0 = best) indexed by players behind the opener.
OPEN_BY_BEHIND: dict[int, float] = {7: 0.12, 6: 0.14, 5: 0.17, 4: 0.20, 3: 0.25, 2: 0.33, 1: 0.42}
OPEN_SB_MULTIWAY = 0.38
OPEN_HU = 0.72
CALL_VS_OPEN_IP = 0.20
CALL_VS_OPEN_BLINDS = 0.16
CALL_VS_OPEN_BB = 0.27
CALL_VS_OPEN_HU_BB = 0.55
THREEBET_IP = 0.075
THREEBET_OOP = 0.06
THREEBET_HU_BB = 0.12
FOURBET = 0.03
CALL_VS_3BET = 0.08
FIVEBET = 0.02
STREET_MARGIN: tuple[float, float, float, float] = (0.0, 0.06, 0.04, 0.0)


class MicroAgent:
    """Decides in well under one millisecond per action. See the module docstring."""

    name = "micro"

    def __init__(
        self,
        seed: int = 0,
        time_budget_ns: int = 1_000_000,
        equity_budget_ns: int = 100_000,
        max_samples: int = 800,
        min_samples: int = 48,
        tb: Tables | None = None,
        gc_freeze: bool = False,
    ) -> None:
        self.tb = tb or tables()
        self.rng = random.Random(seed)
        self.seat = -1
        self.n_seats = 0
        self.time_budget_ns = time_budget_ns
        self.equity_budget_ns = equity_budget_ns
        self.max_samples = max_samples
        self.min_samples = min_samples
        self._ranges: dict[float, _core.CompiledRange] = {
            f: Range.top_fraction([int(x) for x in self.tb.order_multi], f).compile() for f in RANGE_FRACTIONS
        }
        self._range_random = self._ranges[1.0]
        self._class_ranges: dict[float, npt.NDArray[np.float64]] = {
            f: Range.top_fraction([int(x) for x in self.tb.order_multi], f).weights for f in RANGE_FRACTIONS
        }
        self.ns_base, self.ns_per_opp = self._calibrate()
        self.last_samples = 0
        self.last_latency_ns = 0
        self.decisions = 0
        if gc_freeze:
            # Move everything allocated so far (tables, compiled ranges, interpreter state)
            # into the permanent generation so later collections do not traverse it.
            gc.collect()
            gc.freeze()

    # ------------------------------------------------------------------
    def _calibrate(self) -> tuple[float, float]:
        hero: list[int] = [48, 51]
        board: list[int] = []
        best1 = best4 = float("inf")
        for _ in range(3):
            t0 = time.perf_counter_ns()
            _core.equity_vs_compiled(hero, board, [self._range_random], 2000, 7)
            best1 = min(best1, (time.perf_counter_ns() - t0) / 2000.0)
            t0 = time.perf_counter_ns()
            _core.equity_vs_compiled(hero, board, [self._range_random] * 4, 2000, 7)
            best4 = min(best4, (time.perf_counter_ns() - t0) / 2000.0)
        per_opp = max(5.0, (best4 - best1) / 3.0)
        base = max(10.0, best1 - per_opp)
        return base, per_opp

    def reset(self, seat: int, n_seats: int, seed: int) -> None:
        self.seat = seat
        self.n_seats = n_seats
        self.rng = random.Random(seed)

    def _range(self, fraction: float) -> _core.CompiledRange:
        best = min(RANGE_FRACTIONS, key=lambda f: abs(f - fraction))
        return self._ranges[best]

    def _class_range(self, fraction: float) -> npt.NDArray[np.float64]:
        best = min(RANGE_FRACTIONS, key=lambda f: abs(f - fraction))
        return self._class_ranges[best]

    # ------------------------------------------------------------------
    def act(self, obs: Observation) -> Action:
        t0 = time.perf_counter_ns()
        self.decisions += 1
        action = self._preflop(obs, t0) if obs.street == Street.PREFLOP else self._postflop(obs, t0)
        self.last_latency_ns = time.perf_counter_ns() - t0
        return action

    # ------------------------------------------------------------------
    # sizing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clamp_raise(obs: Observation, total: int) -> Action:
        if not obs.can_raise:
            return CALL if obs.to_call > 0 else CHECK
        total = max(obs.min_raise_to, min(obs.max_raise_to, int(total)))
        return Action(ActionKind.RAISE, total)

    def _bet_pot_fraction(self, obs: Observation, frac: float) -> Action:
        pot_after_call = obs.pot + obs.to_call
        target = obs.bet + obs.to_call + int(frac * pot_after_call)
        if target - obs.bet >= 0.6 * obs.stack:
            target = obs.max_raise_to
        return self._clamp_raise(obs, target)

    def _raise_multiple(self, obs: Observation, mult: float) -> Action:
        top = max(obs.bets)
        target = int(top * mult)
        if target - obs.bet >= 0.55 * obs.stack:
            target = obs.max_raise_to
        return self._clamp_raise(obs, target)

    @staticmethod
    def _shove(obs: Observation) -> Action:
        return Action(ActionKind.RAISE, obs.max_raise_to) if obs.can_raise else (CALL if obs.to_call else CHECK)

    # ------------------------------------------------------------------
    # ICM helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _icm(stacks: list[float], payouts: tuple[float, ...], hero: int) -> float:
        return float(_core.icm_equity(stacks, list(payouts))[hero])

    def _icm_jam_vs_fold(self, obs: Observation, cls: int, eff_bb: float, k: int) -> bool:
        """ICM expected value of a first-in jam versus folding (no-overcall model)."""
        hero = obs.seat
        n = len(obs.stacks)
        tot = [float(obs.stacks[i] + obs.bets[i]) for i in range(n)]
        bets = [float(b) for b in obs.bets]
        others = [i for i in range(n) if i != hero and obs.in_hand[i]]
        payouts = obs.payouts
        # Fold: hero keeps the stack behind; the posted chips go to the last player to act.
        fold_stacks = list(tot)
        fold_stacks[hero] = float(obs.stacks[hero])
        pot_now = sum(bets)
        fold_stacks[others[-1]] += pot_now
        ev_fold = self._icm(fold_stacks, payouts, hero)
        calls = self.tb.call_ranges(eff_bb, k + 1)
        survive = 1.0
        ev_jam = 0.0
        dead_others = sum(bets[i] for i in others)
        for idx, i in enumerate(others[-calls.shape[0] :]):
            rng = calls[idx]
            p_call = self.tb.range_mass(cls, rng)
            eq = self.tb.equity_vs_class_range(cls, rng)
            risk = min(tot[hero], tot[i])
            win = list(tot)
            lose = list(tot)
            for j in others:
                if j != i:
                    win[j] -= bets[j]
                    lose[j] -= bets[j]
            win[hero] = tot[hero] + risk + (dead_others - bets[i])
            win[i] = tot[i] - risk
            lose[hero] = tot[hero] - risk
            lose[i] = tot[i] + risk + (dead_others - bets[i])
            ev_call = eq * self._icm(win, payouts, hero) + (1.0 - eq) * self._icm(lose, payouts, hero)
            ev_jam += survive * p_call * ev_call
            survive *= 1.0 - p_call
        nocall = list(tot)
        for j in others:
            nocall[j] -= bets[j]
        nocall[hero] = tot[hero] + dead_others
        ev_jam += survive * self._icm(nocall, payouts, hero)
        return ev_jam > ev_fold

    def _icm_call_allin(self, obs: Observation, eq: float, villain: int) -> bool:
        """Call an all-in (or a stack-committing bet) if ICM EV(call) > ICM EV(fold)."""
        hero = obs.seat
        n = len(obs.stacks)
        tot = [float(obs.stacks[i] + obs.bets[i]) for i in range(n)]
        payouts = obs.payouts
        fold_stacks = list(tot)
        fold_stacks[hero] = float(obs.stacks[hero])
        fold_stacks[villain] += float(obs.pot) - float(obs.bets[hero])  # villain scoops what is there
        for j in range(n):
            if j not in (hero, villain):
                fold_stacks[j] = float(obs.stacks[j])
        ev_fold = self._icm(fold_stacks, payouts, hero)
        risk = min(tot[hero], tot[villain])
        dead = float(obs.pot) - float(obs.bets[hero]) - float(obs.bets[villain])
        win = list(tot)
        lose = list(tot)
        for j in range(n):
            if j not in (hero, villain):
                win[j] = float(obs.stacks[j])
                lose[j] = float(obs.stacks[j])
        win[hero] = tot[hero] + risk + dead
        win[villain] = tot[villain] - risk
        lose[hero] = tot[hero] - risk
        lose[villain] = tot[villain] + risk + dead
        ev_call = eq * self._icm(win, payouts, hero) + (1.0 - eq) * self._icm(lose, payouts, hero)
        return ev_call > ev_fold

    # ------------------------------------------------------------------
    # preflop
    # ------------------------------------------------------------------
    def _preflop(self, obs: Observation, t0: int) -> Action:
        bb = obs.big_blind
        cls = class_index(obs.hole[0], obs.hole[1])
        n_in = obs.n_in_hand
        hero = obs.seat
        hu = n_in == 2
        eff_bb = obs.effective_stack / bb
        raises = [(seat, amt) for (st, seat, kind, amt) in obs.history if st == 0 and kind == ActionKind.RAISE]
        n_raises = len(raises)
        rank = self.tb.rank_hu[cls] if hu else self.tb.rank_multi[cls]
        n = len(obs.stacks)
        acted = {seat for (st, seat, _k, _a) in obs.history if st == 0}
        behind = [i for i in range(n) if obs.in_hand[i] and i != hero and i not in acted]
        k_behind = len(behind)
        is_bb = obs.to_call == 0 and obs.bet == bb and n_raises == 0
        is_sb = obs.bet == obs.small_blind and obs.to_call > 0 and n_raises == 0

        if n_raises == 0:
            # Unopened pot or limped pot.
            k = n_in - 1  # everybody else still in the hand must get through our raise
            if eff_bb <= PUSHFOLD_MAX_BB:
                if obs.payouts:
                    jam = self._icm_jam_vs_fold(obs, cls, eff_bb, k)
                else:
                    jam = self.rng.random() < self.tb.pushfold_jam(eff_bb, cls, k + 1)
                if jam:
                    return self._shove(obs)
                return CHECK if obs.to_call == 0 else FOLD
            # Deep: open-raise by position.
            if hu:
                threshold = OPEN_HU if not is_bb else 0.30
            elif is_bb:
                threshold = 0.18  # isolate limpers from the big blind
            elif is_sb:
                threshold = OPEN_SB_MULTIWAY
            else:
                threshold = OPEN_BY_BEHIND.get(max(1, min(7, k_behind)), 0.15)
            if rank <= threshold and obs.can_raise:
                limpers = sum(1 for (st, _s, kind, _a) in obs.history if st == 0 and kind == ActionKind.CALL)
                size = int(bb * (2.5 + limpers)) if not hu else int(bb * 2.5)
                return self._clamp_raise(obs, max(size, obs.min_raise_to))
            return CHECK if obs.to_call == 0 else FOLD

        # Facing at least one raise.
        raiser = raises[-1][0]
        raiser_total = obs.stacks[raiser] + obs.bets[raiser]
        raiser_bb = raiser_total / bb
        facing_allin = obs.stacks[raiser] == 0 or obs.to_call >= obs.stack
        if facing_allin or eff_bb <= PUSHFOLD_MAX_BB:
            jam_rng: npt.NDArray[np.floating[Any]]
            if n_raises == 1:
                jam_rng = self.tb.jam_range(min(raiser_bb, eff_bb) if facing_allin else eff_bb, n_in + len(acted) - 1)
            else:
                jam_rng = self._class_range(0.10)
            eq = self.tb.equity_vs_class_range(cls, jam_rng)
            if obs.payouts:
                call = self._icm_call_allin(obs, eq, raiser)
            else:
                po = obs.to_call / (obs.pot + obs.to_call)
                call = eq > po + 0.02 * max(0, k_behind)
            if not call:
                return FOLD
            if not facing_allin and obs.can_raise:
                return self._shove(obs)  # re-jam over a non-all-in raise
            if obs.can_raise and obs.stack > obs.to_call and k_behind > 0 and eq > 0.55:
                return self._shove(obs)  # isolate: shove over the all-in with a strong hand
            return CALL

        # Deep stacks facing a raise.
        in_position = self._has_position(obs, raiser)
        if n_raises == 1:
            if hu and is_bb_vs_open(obs):
                three, call_thr = THREEBET_HU_BB, CALL_VS_OPEN_HU_BB
            elif obs.bet == bb:
                three, call_thr = THREEBET_OOP, CALL_VS_OPEN_BB
            elif obs.bet == obs.small_blind:
                three, call_thr = THREEBET_OOP, CALL_VS_OPEN_BLINDS
            else:
                three, call_thr = (THREEBET_IP, CALL_VS_OPEN_IP) if in_position else (THREEBET_OOP, CALL_VS_OPEN_BLINDS)
            if rank <= three and obs.can_raise:
                return self._raise_multiple(obs, 3.0 if in_position else 3.5)
            if rank <= call_thr and obs.to_call <= 5 * bb:
                return CALL
            return FOLD
        if n_raises == 2:
            if rank <= FOURBET and obs.can_raise:
                return self._raise_multiple(obs, 2.3)
            if rank <= CALL_VS_3BET and eff_bb > 40 and obs.to_call <= 12 * bb:
                return CALL
            return FOLD
        if rank <= FIVEBET:
            return self._shove(obs)
        return FOLD

    @staticmethod
    def _has_position(obs: Observation, villain: int) -> bool:
        """True if hero acts after ``villain`` postflop (closer to the button clockwise)."""
        n = len(obs.stacks)
        d_hero = (obs.seat - obs.button - 1) % n
        d_vill = (villain - obs.button - 1) % n
        return d_hero > d_vill

    # ------------------------------------------------------------------
    # postflop
    # ------------------------------------------------------------------
    def _opponent_ranges(self, obs: Observation) -> list[_core.CompiledRange]:
        n = len(obs.stacks)
        pre: dict[int, list[int]] = {i: [] for i in range(n)}
        post_aggr: dict[int, int] = {i: 0 for i in range(n)}
        for st, seat, kind, _amt in obs.history:
            if st == 0:
                pre[seat].append(kind)
            elif kind == ActionKind.RAISE:
                post_aggr[seat] += 1
        n_raises = sum(1 for (st, _s, kind, _a) in obs.history if st == 0 and kind == ActionKind.RAISE)
        out: list[_core.CompiledRange] = []
        for i in range(n):
            if i == obs.seat or not obs.in_hand[i]:
                continue
            kinds = pre[i]
            if ActionKind.RAISE in kinds:
                frac = {1: 0.15, 2: 0.07}.get(n_raises, 0.04)
            elif ActionKind.CALL in kinds:
                frac = 0.30 if n_raises >= 1 else 0.45
            else:
                frac = 1.0
            if post_aggr[i]:
                frac = max(0.04, frac * (0.6 ** post_aggr[i]))
            out.append(self._range(frac))
        return out or [self._range_random]

    def _equity(self, obs: Observation, t0: int, ranges: list[_core.CompiledRange]) -> float:
        elapsed = time.perf_counter_ns() - t0
        remaining = self.equity_budget_ns - elapsed
        cost = self.ns_base + self.ns_per_opp * len(ranges)
        n = int(remaining / cost) if remaining > 0 else self.min_samples
        n = max(self.min_samples, min(self.max_samples, n))
        self.last_samples = n
        seed = self.rng.getrandbits(62)
        return float(_core.equity_vs_compiled(list(obs.hole), list(obs.board), ranges, n, seed))

    def _postflop(self, obs: Observation, t0: int) -> Action:
        n_opp = max(1, obs.n_in_hand - 1)
        ranges = self._opponent_ranges(obs)
        eq = self._equity(obs, t0, ranges)
        if obs.street == Street.RIVER and n_opp == 1:
            action = self._river_cfr(obs, eq)
            if action is not None:
                return action
        pot_after = obs.pot + obs.to_call
        po = obs.to_call / pot_after if obs.to_call > 0 else 0.0
        extra = 0.03 * (n_opp - 1)
        was_aggressor = self._was_aggressor(obs)

        if obs.to_call == 0:
            value_thr = 1.0 / (n_opp + 1) + 0.15 + extra
            if eq >= value_thr and obs.can_raise:
                return self._bet_pot_fraction(obs, 0.66)
            if obs.can_raise and n_opp == 1 and obs.street != Street.RIVER:
                if was_aggressor and 0.33 <= eq < value_thr and self.rng.random() < 0.55:
                    return self._bet_pot_fraction(obs, 0.5)
                if self.rng.random() < 0.08:
                    return self._bet_pot_fraction(obs, 0.5)
            return CHECK

        # Facing a bet.
        raise_thr = 0.78 + extra
        committing = obs.to_call >= 0.5 * obs.stack
        if obs.payouts and committing:
            villain = max(
                (i for i in range(len(obs.stacks)) if i != obs.seat and obs.in_hand[i]), key=lambda i: obs.bets[i]
            )
            return CALL if self._icm_call_allin(obs, eq, villain) else FOLD
        if eq >= raise_thr and obs.can_raise:
            return self._raise_multiple(obs, 2.6)
        need = po + STREET_MARGIN[obs.street] + extra
        if eq >= need:
            return CALL
        return FOLD

    @staticmethod
    def _was_aggressor(obs: Observation) -> bool:
        last_raiser = -1
        for st, seat, kind, _amt in obs.history:
            if st < obs.street and kind == ActionKind.RAISE:
                last_raiser = seat
        return last_raiser == obs.seat

    def _river_cfr(self, obs: Observation, eq: float) -> Action | None:
        hero = obs.seat
        river = [(seat, kind, amt) for (st, seat, kind, amt) in obs.history if st == Street.RIVER]
        n = len(obs.stacks)
        first_postflop = next((obs.button + 1 + d) % n for d in range(n) if obs.in_hand[(obs.button + 1 + d) % n])
        hero_oop = hero == first_postflop
        street_bets = sum(obs.bets)
        pot0 = obs.pot - street_bets
        if pot0 <= 0:
            return None
        key_parts: list[str] = []
        last_bet_ratio = 0.0
        for _seat, kind, amt in river:
            if kind == ActionKind.CHECK:
                key_parts.append("x")
            elif kind == ActionKind.RAISE:
                if not key_parts or key_parts[-1] == "x":
                    last_bet_ratio = amt / pot0
                    key_parts.append("b")
                else:
                    key_parts.append("r")
            else:
                return None  # a call or fold on the river ends the street
        if len(key_parts) > 3:
            return None
        if (len(key_parts) == 0) != hero_oop:
            return None
        stack_pots = obs.effective_stack / pot0
        for size_key in self._size_keys(last_bet_ratio):
            key = " ".join("b" + size_key if p == "b" else p for p in key_parts)
            if any(p == "b" for p in key_parts) and size_key == "a":
                key = " ".join("a" if p == "b" else p for p in key_parts)
            found = self.tb.river_lookup(stack_pots, key)
            if found is not None:
                acts, strat = found
                bucket = min(self.tb.river_buckets - 1, max(0, int(eq * self.tb.river_buckets)))
                probs = strat[bucket]
                r = self.rng.random()
                acc = 0.0
                chosen = acts[-1]
                for a, p in zip(acts, probs, strict=True):
                    acc += float(p)
                    if r < acc:
                        chosen = a
                        break
                return self._translate_river_action(obs, chosen)
        return None

    @staticmethod
    def _size_keys(ratio: float) -> list[str]:
        if ratio <= 0.0:
            return ["0.5"]
        order = sorted(["0.5", "1"], key=lambda s: abs(float(s) - ratio))
        return [*order, "a"]

    def _translate_river_action(self, obs: Observation, tag: str) -> Action:
        if tag == "x":
            return CHECK if obs.to_call == 0 else CALL
        if tag == "f":
            return FOLD if obs.to_call > 0 else CHECK
        if tag == "c":
            return CALL if obs.to_call > 0 else CHECK
        if tag == "a":
            return self._shove(obs)
        if tag == "r":
            return self._raise_multiple(obs, 3.0)
        frac = float(tag[1:])
        return self._bet_pot_fraction(obs, frac)


def is_bb_vs_open(obs: Observation) -> bool:
    return obs.bet == obs.big_blind


def describe(agent: MicroAgent) -> dict[str, Any]:
    return {
        "ns_base": round(agent.ns_base, 1),
        "ns_per_opponent_sample": round(agent.ns_per_opp, 1),
        "equity_budget_ns": agent.equity_budget_ns,
        "max_samples": agent.max_samples,
    }
