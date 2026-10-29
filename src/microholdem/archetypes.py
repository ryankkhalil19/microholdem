"""Opponent archetypes for the simulator.

Each archetype is a deliberately simple, legible policy. They exist to give
the main agent a population to be measured against, not to be strong. All of
them are deterministic given their seed.
"""

from __future__ import annotations

import random

from . import _core
from .cards import class_index
from .engine import CALL, CHECK, FOLD, Action, Observation, Street
from .tables import Tables, tables

__all__ = [
    "Blitz",
    "Granite",
    "Jammer",
    "Oracle",
    "Passenger",
    "Scatter",
    "Sponge",
    "all_archetypes",
    "make_archetype",
]


class _Base:
    name = "base"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.seat = -1
        self.n_seats = 0
        self.tb: Tables = tables()

    def reset(self, seat: int, n_seats: int, seed: int) -> None:
        self.seat = seat
        self.n_seats = n_seats
        self.rng = random.Random(seed)

    def act(self, obs: Observation) -> Action:  # pragma: no cover - overridden
        raise NotImplementedError

    # helpers -------------------------------------------------------------
    def strength(self, obs: Observation) -> float:
        """Preflop: equity of the class vs one random hand. Postflop: quick Monte Carlo."""
        if obs.street == Street.PREFLOP:
            return float(self.tb.preflop_equity[0][class_index(*obs.hole)])
        n_opp = max(1, obs.n_in_hand - 1)
        return float(_core.equity_vs_random(list(obs.hole), list(obs.board), n_opp, 200, self.rng.getrandbits(62)))

    @staticmethod
    def pot_odds(obs: Observation) -> float:
        if obs.to_call == 0:
            return 0.0
        return obs.to_call / (obs.pot + obs.to_call)

    @staticmethod
    def raise_pot(obs: Observation, frac: float) -> Action:
        target = obs.bets[obs.seat] + obs.to_call + int(frac * (obs.pot + obs.to_call))
        return Action.raise_to(max(obs.min_raise_to, min(obs.max_raise_to, target)))

    @staticmethod
    def shove(obs: Observation) -> Action:
        return Action.raise_to(obs.max_raise_to)


class Passenger(_Base):
    """Checks whenever it can, folds to any bet. The floor of the population."""

    name = "passenger"

    def act(self, obs: Observation) -> Action:
        return CHECK if obs.to_call == 0 else FOLD


class Scatter(_Base):
    """Uniformly random legal action; raises pick a random legal size."""

    name = "scatter"

    def act(self, obs: Observation) -> Action:
        options: list[Action] = [CHECK if obs.to_call == 0 else FOLD]
        if obs.to_call > 0:
            options.append(CALL)
        if obs.can_raise:
            options.append(Action.raise_to(self.rng.randint(obs.min_raise_to, obs.max_raise_to)))
        return self.rng.choice(options)


class Granite(_Base):
    """Tight and passive: plays only strong hands, rarely raises, folds to pressure."""

    name = "granite"

    def act(self, obs: Observation) -> Action:
        s = self.strength(obs)
        if obs.street == Street.PREFLOP:
            if s >= 0.66:  # premium pairs and AK-ish
                return self.raise_pot(obs, 1.0) if obs.can_raise else CALL
            if s >= 0.58:
                return CALL if obs.to_call <= 3 * obs.big_blind else FOLD
            return CHECK if obs.to_call == 0 else FOLD
        if s >= 0.85:
            return self.raise_pot(obs, 0.75) if obs.can_raise else CALL
        if obs.to_call == 0:
            return CHECK
        return CALL if s >= self.pot_odds(obs) + 0.15 else FOLD


class Blitz(_Base):
    """Loose and aggressive: raises most hands, bluffs often, calls light."""

    name = "blitz"

    def act(self, obs: Observation) -> Action:
        s = self.strength(obs)
        r = self.rng.random()
        if obs.can_raise and (s >= 0.5 or r < 0.35):
            if obs.street == Street.PREFLOP:
                return self.raise_pot(obs, 1.0) if r < 0.85 else self.shove(obs)
            return self.raise_pot(obs, 0.8)
        if obs.to_call == 0:
            return CHECK
        return CALL if s >= self.pot_odds(obs) - 0.05 else FOLD


class Sponge(_Base):
    """Calling station: calls almost everything, raises only with monsters."""

    name = "sponge"

    def act(self, obs: Observation) -> Action:
        s = self.strength(obs)
        if obs.can_raise and s >= 0.9 and obs.street != Street.PREFLOP:
            return self.raise_pot(obs, 1.0)
        if obs.to_call == 0:
            return CHECK
        if obs.to_call >= obs.stack and s < 0.4:
            return FOLD
        return CALL if s >= 0.22 else FOLD


class Oracle(_Base):
    """Pure equity player: bets and calls strictly on Monte Carlo equity vs pot odds."""

    name = "oracle"

    def act(self, obs: Observation) -> Action:
        s = self.strength(obs)
        n_opp = max(1, obs.n_in_hand - 1)
        edge = s - 1.0 / (n_opp + 1)
        if obs.can_raise and edge > 0.18:
            return self.raise_pot(obs, 0.7)
        if obs.to_call == 0:
            return CHECK
        return CALL if s > self.pot_odds(obs) else FOLD


class Jammer(_Base):
    """Short-stack specialist: Nash push/fold preflop, otherwise fit-or-fold."""

    name = "jammer"

    def act(self, obs: Observation) -> Action:
        cls = class_index(*obs.hole)
        bb = obs.big_blind
        eff_bb = obs.effective_stack / bb
        if obs.street == Street.PREFLOP:
            facing_shove = obs.to_call >= obs.stack * 0.6 and obs.to_call > 0
            if facing_shove:
                p_call = self.tb.pushfold_call(eff_bb, cls, n_players=obs.n_in_hand)
                return CALL if self.rng.random() < p_call else FOLD
            if obs.can_raise and eff_bb <= 20 and obs.to_call <= bb:
                p_jam = self.tb.pushfold_jam(eff_bb, cls, n_players=obs.n_in_hand)
                return self.shove(obs) if self.rng.random() < p_jam else (CHECK if obs.to_call == 0 else FOLD)
            s = float(self.tb.preflop_equity[0][cls])
            if obs.can_raise and s >= 0.62:
                return self.raise_pot(obs, 1.0)
            return CHECK if obs.to_call == 0 else (CALL if s >= 0.55 and obs.to_call <= 2 * bb else FOLD)
        s = self.strength(obs)
        if obs.can_raise and s >= 0.7:
            return self.shove(obs) if eff_bb <= 15 else self.raise_pot(obs, 0.75)
        if obs.to_call == 0:
            return CHECK
        return CALL if s >= self.pot_odds(obs) + 0.05 else FOLD


_REGISTRY: dict[str, type[_Base]] = {c.name: c for c in (Passenger, Scatter, Granite, Blitz, Sponge, Oracle, Jammer)}


def all_archetypes() -> list[str]:
    return list(_REGISTRY)


def make_archetype(name: str, seed: int = 0) -> _Base:
    try:
        return _REGISTRY[name](seed)
    except KeyError as exc:
        raise ValueError(f"unknown archetype {name!r}; choose from {all_archetypes()}") from exc
