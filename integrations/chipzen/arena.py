"""Local arena in Chipzen's bot-vs-bot match format.

A match is a heads-up freezeout: both seats start with 10,000 chips, the blinds start at 50/100 and
step up every 20 hands, and play continues until one seat has every chip. Hands are dealt by
microholdem's engine; a ``chipzen.Bot`` is seated through :class:`ChipzenSeat`, which hands it the
same ``match_start``, ``turn_request`` and ``round_result`` payloads the platform sends, including
the fact that ``match_start`` only ever names the opening blinds.

Compare two bot files (each defines one ``chipzen.Bot`` subclass) over mirrored matches:

    python arena.py --a bot.py --b /path/to/older_bot.py --matches 2000
    python arena.py --a bot.py --pool --matches 500          # against the built-in opponents
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import multiprocessing
import os
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from chipzen import Bot as ChipzenBot
from chipzen import GameState

from microholdem import _core
from microholdem.archetypes import make_archetype
from microholdem.cards import card_str, class_index
from microholdem.engine import CALL, CHECK, FOLD, ActionKind, HandConfig, HandResult, Observation, Street, play_hand
from microholdem.engine import Action as EngineAction
from microholdem.engine import Bot as EngineBot
from microholdem.tables import tables

STARTING_STACK = 10_000
HANDS_PER_LEVEL = 20
# The first six levels are the ones seen in Chipzen replays; later levels continue the pattern.
BLIND_LEVELS: tuple[tuple[int, int], ...] = (
    (50, 100), (75, 150), (100, 200), (150, 300), (200, 400), (300, 600),
    (400, 800), (600, 1200), (800, 1600), (1200, 2400), (1600, 3200), (2400, 4800),
)  # fmt: skip
PHASES = ("preflop", "flop", "turn", "river")
NAMES = {
    int(ActionKind.FOLD): "fold",
    int(ActionKind.CHECK): "check",
    int(ActionKind.CALL): "call",
    int(ActionKind.RAISE): "raise",
}


def blinds_for_hand(hand_index: int) -> tuple[int, int]:
    return BLIND_LEVELS[min(len(BLIND_LEVELS) - 1, hand_index // HANDS_PER_LEVEL)]


def _entry(seat: int, action: str, amount: int, phase: str = "preflop") -> dict[str, Any]:
    return {"seat": seat, "action": action, "amount": amount, "phase": phase, "is_timeout": False}


class ChipzenSeat:
    """Seats a ``chipzen.Bot`` at microholdem's engine, speaking the Chipzen payloads to it."""

    def __init__(self, bot: ChipzenBot, name: str = "", agent_seed: int | None = None) -> None:
        self.bot = bot
        self.name = name or type(bot).__name__
        self.agent_seed = agent_seed  # when set, the bot's agent draws from this seed in every match
        self.seat = -1
        self.blinds: list[dict[str, Any]] = []
        self.illegal = 0

    def reset(self, seat: int, n_seats: int, seed: int) -> None:
        self.seat = seat
        sb, bb = BLIND_LEVELS[0]
        self.bot.on_match_start(
            {
                "type": "match_start",
                "your_seat": seat,
                "game_config": {
                    "variant": "nlhe",
                    "starting_stack": STARTING_STACK,
                    "small_blind": sb,
                    "big_blind": bb,
                    "ante": 0,
                    "num_players": n_seats,
                },
            }
        )
        agent = getattr(self.bot, "agent", None)
        if self.agent_seed is not None and agent is not None:
            # match_start reseeds the bot's agent from the system RNG; a seeded run overrides that.
            agent.rng = random.Random(self.agent_seed)

    def start_hand(self, hand_number: int, button: int, stacks: list[int], blinds: list[dict[str, Any]]) -> None:
        self.blinds = blinds
        state = {"hand_number": hand_number, "dealer_seat": button, "your_hole_cards": [], "stacks": list(stacks)}
        self.bot.on_round_start({"type": "round_start", "round_number": hand_number, "state": state})

    def _history(self, records: tuple[tuple[int, int, int, int], ...]) -> list[dict[str, Any]]:
        return [*self.blinds, *(_entry(seat, NAMES[kind], amount, PHASES[st]) for st, seat, kind, amount in records)]

    def act(self, obs: Observation) -> EngineAction:
        valid = (["check"] if obs.to_call == 0 else ["fold", "call"]) + (["raise"] if obs.can_raise else [])
        message = {
            "type": "turn_request",
            "seat": obs.seat,
            "request_id": "arena",
            "valid_actions": valid,
            "state": {
                "hand_number": obs.hand_number,
                "phase": PHASES[obs.street],
                "board": [card_str(c) for c in obs.board],
                "your_hole_cards": [card_str(c) for c in obs.hole],
                "pot": obs.pot,
                "your_stack": obs.stacks[obs.seat],
                "opponent_stacks": [s for i, s in enumerate(obs.stacks) if i != obs.seat],
                "to_call": obs.to_call,
                "min_raise": obs.min_raise_to,
                "max_raise": obs.max_raise_to,
                "action_history": self._history(obs.history),
            },
        }
        action = self.bot.decide(GameState.from_turn_request(message, your_seat=obs.seat, dealer_seat=obs.button))
        if action.action not in valid:
            self.illegal += 1
            return CHECK if obs.to_call == 0 else FOLD
        if action.action == "raise":
            return EngineAction.raise_to(action.amount)
        return {"fold": FOLD, "check": CHECK, "call": CALL}[action.action]

    def end_hand(self, result: HandResult) -> None:
        showdown = (
            [{"seat": s, "hole_cards": [card_str(c) for c in h]} for s, h in enumerate(result.hole) if h is not None]
            if result.showdown
            else []
        )
        history = self._history(tuple((a.street, a.seat, a.kind, a.amount) for a in result.actions))
        self.bot.on_round_result(
            {
                "type": "round_result",
                "result": {
                    "hand_number": result.hand_number,
                    "winner_seats": result.winners,
                    "showdown": showdown,
                    "action_history": history,
                    "stacks": result.end_stacks,
                },
            }
        )


class Profile:
    """A parameterised heads-up opponent; each preset copies the frequencies of a bot met on Chipzen.

    Preflop it plays fixed fractions of the hand ranking; postflop it bets, calls and folds on its
    equity against a random hand. Deliberately simple: it is something to be measured against.
    """

    def __init__(
        self,
        name: str,
        seed: int,
        *,
        open_frac: float,
        limp_frac: float,
        open_size: float,
        threebet_frac: float,
        jam3_frac: float,
        defend_frac: float,
        value_bet: float,
        bluff_freq: float,
        bet_size: float,
        call_margin: float,
    ) -> None:
        self.name = name
        self.rng = random.Random(seed)
        self.tb = tables()
        self.open_frac, self.limp_frac, self.open_size = open_frac, limp_frac, open_size
        self.threebet_frac, self.jam3_frac, self.defend_frac = threebet_frac, jam3_frac, defend_frac
        self.value_bet, self.bluff_freq, self.bet_size, self.call_margin = value_bet, bluff_freq, bet_size, call_margin

    def reset(self, seat: int, n_seats: int, seed: int) -> None:
        self.rng = random.Random(seed)

    def act(self, obs: Observation) -> EngineAction:
        bb = obs.big_blind
        if obs.street == Street.PREFLOP:
            rank = float(self.tb.rank_hu[class_index(*obs.hole)])
            raises = sum(1 for st, _s, kind, _a in obs.history if st == 0 and kind == ActionKind.RAISE)
            if obs.effective_stack <= 10 * bb and raises == 0 and obs.can_raise:
                jam = self.rng.random() < self.tb.pushfold_jam(obs.effective_stack / bb, class_index(*obs.hole), 2)
                return EngineAction.raise_to(obs.max_raise_to) if jam else (CHECK if obs.to_call == 0 else FOLD)
            if raises == 0:
                if obs.to_call > 0:  # small blind, first in
                    if rank <= self.open_frac and obs.can_raise:
                        return EngineAction.raise_to(
                            max(obs.min_raise_to, min(obs.max_raise_to, int(self.open_size * bb)))
                        )
                    return CALL if rank <= self.open_frac + self.limp_frac else FOLD
                if rank <= 0.25 and obs.can_raise:  # big blind after a limp
                    return EngineAction.raise_to(max(obs.min_raise_to, min(obs.max_raise_to, 3 * bb)))
                return CHECK
            if raises == 1 and obs.bet <= bb:  # facing an open (or a raise after limping)
                if rank <= self.jam3_frac and obs.can_raise:
                    return EngineAction.raise_to(obs.max_raise_to)
                if rank <= self.threebet_frac and obs.can_raise:
                    top = max(obs.bets)
                    return EngineAction.raise_to(max(obs.min_raise_to, min(obs.max_raise_to, int(3.5 * top))))
                if obs.to_call >= obs.stack:
                    return CALL if rank <= 0.12 else FOLD
                return CALL if rank <= self.defend_frac and obs.to_call <= 4 * bb else FOLD
            if rank <= 0.04 and obs.can_raise:
                return EngineAction.raise_to(obs.max_raise_to)
            committed = obs.to_call >= 0.4 * obs.stack
            return CALL if rank <= (0.06 if committed else 0.14) else FOLD
        eq = float(_core.equity_vs_random(list(obs.hole), list(obs.board), 1, 300, self.rng.getrandbits(62)))
        if obs.to_call == 0:
            if obs.can_raise and (eq >= self.value_bet or self.rng.random() < self.bluff_freq):
                target = obs.bet + int(self.bet_size * obs.pot)
                return EngineAction.raise_to(max(obs.min_raise_to, min(obs.max_raise_to, target)))
            return CHECK
        if eq >= 0.85 and obs.can_raise:
            return EngineAction.raise_to(max(obs.min_raise_to, min(obs.max_raise_to, int(2.8 * max(obs.bets)))))
        return CALL if eq >= obs.to_call / (obs.pot + obs.to_call) + self.call_margin else FOLD


# Frequencies read from the Season 16 replays (6BP9, Phineas-IV) and earlier challenges (glambot).
PRESETS: dict[str, dict[str, float]] = {
    "minraiser": dict(open_frac=0.80, limp_frac=0.0, open_size=2.0, threebet_frac=0.05, jam3_frac=0.0,
                      defend_frac=0.95, value_bet=0.58, bluff_freq=0.35, bet_size=0.5, call_margin=0.25),
    "barreler": dict(open_frac=0.56, limp_frac=0.40, open_size=2.0, threebet_frac=0.19, jam3_frac=0.08,
                     defend_frac=0.60, value_bet=0.55, bluff_freq=0.45, bet_size=0.55, call_margin=0.05),
    "station": dict(open_frac=0.0, limp_frac=1.0, open_size=2.0, threebet_frac=0.04, jam3_frac=0.0,
                    defend_frac=1.0, value_bet=0.70, bluff_freq=0.05, bet_size=0.5, call_margin=-0.15),
    "regular": dict(open_frac=0.72, limp_frac=0.0, open_size=2.5, threebet_frac=0.12, jam3_frac=0.0,
                    defend_frac=0.65, value_bet=0.60, bluff_freq=0.25, bet_size=0.66, call_margin=0.04),
}  # fmt: skip
ARCHETYPES = ("oracle", "blitz", "granite", "jammer", "sponge")
POOL = (*PRESETS, *ARCHETYPES)


def make_opponent(name: str, seed: int) -> EngineBot:
    if name in PRESETS:
        return Profile(name, seed, **PRESETS[name])
    return make_archetype(name, seed)


@dataclass(slots=True)
class MatchResult:
    winner: int  # seat index, or -1 if the hand cap was reached with equal stacks
    hands: int
    illegal: tuple[int, int]


def play_match(bots: list[EngineBot], seed: int, first_button: int = 0, max_hands: int = 1000) -> MatchResult:
    """One freezeout. ``bots`` are engine bots; wrap a Chipzen bot in :class:`ChipzenSeat`."""
    rng = random.Random(seed)
    stacks = [STARTING_STACK, STARTING_STACK]
    for i, b in enumerate(bots):
        b.reset(i, 2, rng.getrandbits(32))
    hands = 0
    while min(stacks) > 0 and hands < max_hands:
        sb, bb = blinds_for_hand(hands)
        button = (first_button + hands) % 2  # heads-up: the button posts the small blind
        blinds = [
            _entry(button, "post_small_blind", min(sb, stacks[button])),
            _entry(1 - button, "post_big_blind", min(bb, stacks[1 - button])),
        ]
        for b in bots:
            if isinstance(b, ChipzenSeat):
                b.start_hand(hands + 1, button, stacks, blinds)
        cfg = HandConfig(small_blind=sb, big_blind=bb, time_limit_ns=10**12, record_latency=False)
        result = play_hand(bots, stacks, button, cfg, rng, hands + 1)
        hands += 1
        for b in bots:
            if isinstance(b, ChipzenSeat):
                b.end_hand(result)
    winner = 0 if stacks[0] > stacks[1] else (1 if stacks[1] > stacks[0] else -1)
    illegal = tuple(b.illegal if isinstance(b, ChipzenSeat) else 0 for b in bots)
    return MatchResult(winner, hands, (illegal[0], illegal[1]))


# ----------------------------------------------------------------------------
# Running many matches
# ----------------------------------------------------------------------------
def load_bot_class(path: str) -> type[ChipzenBot]:
    """The ``chipzen.Bot`` subclass defined in the file at ``path``."""
    name = "arena_bot_" + "".join(ch if ch.isalnum() else "_" for ch in os.path.abspath(path))
    if name in sys.modules:
        module = sys.modules[name]
    else:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    for value in vars(module).values():
        if isinstance(value, type) and issubclass(value, ChipzenBot) and value.__module__ == name:
            return value
    raise ValueError(f"no chipzen.Bot subclass in {path}")


def make_seat(spec: str, seed: int, samples: int) -> EngineBot:
    """``spec`` is a pool name or the path of a bot file."""
    if spec in POOL:
        return make_opponent(spec, seed)
    bot = load_bot_class(spec)()
    agent = getattr(bot, "agent", None)
    if agent is not None and samples > 0:
        # A fixed Monte Carlo sample makes a match depend only on its seed and keeps runs fast.
        agent.samples = samples
        agent.max_samples = max(agent.max_samples, samples)
        return ChipzenSeat(bot, os.path.basename(spec), agent_seed=seed)
    return ChipzenSeat(bot, os.path.basename(spec))


def _pair(job: tuple[str, str, int, int]) -> tuple[int, int, int, int]:
    """Two matches on one seed with the seats swapped. Returns (wins for a, matches, hands, illegal by a)."""
    a, b, seed, samples = job
    wins = hands = illegal = 0
    for a_seat in (0, 1):
        bots: list[EngineBot] = [make_seat(a, seed * 2 + 1, samples), make_seat(b, seed * 2 + 2, samples)]
        if a_seat == 1:
            bots.reverse()
        r = play_match(bots, seed)
        wins += r.winner == a_seat
        hands += r.hands
        illegal += r.illegal[a_seat]
    return wins, 2, hands, illegal


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def run(
    a: str, b: str, matches: int, samples: int, seed: int, workers: int, map_fn: Callable[..., Any] | None = None
) -> dict[str, Any]:
    jobs = [(a, b, seed + i, samples) for i in range(matches // 2)]
    if map_fn is None:
        with multiprocessing.Pool(workers) as pool:
            results = pool.map(_pair, jobs, chunksize=max(1, len(jobs) // (workers * 8)))
    else:
        results = list(map_fn(_pair, jobs))
    wins = sum(r[0] for r in results)
    n = sum(r[1] for r in results)
    lo, hi = wilson(wins, n)
    return {
        "a": os.path.basename(a), "b": os.path.basename(b), "matches": n, "wins": wins,
        "win_rate": wins / n if n else 0.0, "ci95": (lo, hi),
        "hands_per_match": sum(r[2] for r in results) / n if n else 0.0, "illegal": sum(r[3] for r in results),
    }  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="bot file under test")
    ap.add_argument("--b", action="append", default=[], help="opponent: a bot file or a pool name (repeatable)")
    ap.add_argument("--pool", action="store_true", help=f"also play every built-in opponent: {', '.join(POOL)}")
    ap.add_argument("--matches", type=int, default=1000, help="matches per opponent (played as mirrored pairs)")
    ap.add_argument(
        "--samples", type=int, default=600, help="fixed Monte Carlo samples per decision (0: the bot's own budget)"
    )
    ap.add_argument("--seed", type=int, default=20261125)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args(argv)
    opponents = [*args.b, *(POOL if args.pool else ())]
    if not opponents:
        ap.error("give --b or --pool")
    total_w = total_n = 0
    for opp in opponents:
        r = run(args.a, opp, args.matches, args.samples, args.seed, args.workers)
        total_w += r["wins"]
        total_n += r["matches"]
        print(
            f"{r['a']:>14} vs {r['b']:<14} {r['wins']:5d}/{r['matches']:<5d} = {100 * r['win_rate']:5.1f}%"
            f"  (95% CI {100 * r['ci95'][0]:.1f} to {100 * r['ci95'][1]:.1f})"
            f"  {r['hands_per_match']:5.1f} hands/match  illegal={r['illegal']}"
        )
    if len(opponents) > 1:
        lo, hi = wilson(total_w, total_n)
        print(
            f"{'overall':>14}    {'':<14} {total_w:5d}/{total_n:<5d} = {100 * total_w / total_n:5.1f}%"
            f"  (95% CI {100 * lo:.1f} to {100 * hi:.1f})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
