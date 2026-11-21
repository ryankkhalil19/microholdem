"""Tests for the Chipzen adapter in bot.py.

Run from this folder, in an environment with microholdem and chipzen-bot installed:

    python -m pytest -q test_bot.py
"""

from __future__ import annotations

import random
from typing import Any

import pytest
from bot import MicroholdemBot, to_chipzen, to_observation
from chipzen import GameState

from microholdem.agent import MicroAgent
from microholdem.archetypes import make_archetype
from microholdem.cards import card_str
from microholdem.engine import CALL, CHECK, FOLD, ActionKind, HandConfig, Observation, play_hand
from microholdem.engine import Action as AgentAction
from microholdem.engine import Bot as EngineBot

PHASES = ("preflop", "flop", "turn", "river")
NAMES = {
    int(ActionKind.FOLD): "fold",
    int(ActionKind.CHECK): "check",
    int(ActionKind.CALL): "call",
    int(ActionKind.RAISE): "raise",
}


def entry(seat: int, action: str, amount: int, phase: str = "preflop") -> dict[str, Any]:
    return {"seat": seat, "action": action, "amount": amount, "phase": phase, "is_timeout": False}


def turn_request(
    *,
    seat: int,
    dealer: int,
    phase: str,
    board: list[str],
    hole: list[str],
    pot: int,
    stacks: list[int],
    to_call: int,
    min_raise: int,
    max_raise: int,
    valid: list[str],
    history: list[dict[str, Any]],
    hand_number: int = 1,
) -> GameState:
    """A GameState parsed from a two-layer turn_request, as the SDK builds it."""
    message = {
        "type": "turn_request",
        "seat": seat,
        "request_id": "req_test",
        "valid_actions": valid,
        "state": {
            "hand_number": hand_number,
            "phase": phase,
            "board": board,
            "your_hole_cards": hole,
            "pot": pot,
            "your_stack": stacks[seat],
            "opponent_stacks": [s for i, s in enumerate(stacks) if i != seat],
            "to_call": to_call,
            "min_raise": min_raise,
            "max_raise": max_raise,
            "action_history": history,
        },
    }
    return GameState.from_turn_request(message, your_seat=seat, dealer_seat=dealer)


# ----------------------------------------------------------------------------
# The full-hand example in POKER-GAME-STATE-PROTOCOL section 4
# ----------------------------------------------------------------------------
PREFLOP = [entry(0, "post_small_blind", 5), entry(1, "post_big_blind", 10)]


def test_protocol_example_preflop_small_blind() -> None:
    state = turn_request(  # Message 4
        seat=0, dealer=0, phase="preflop", board=[], hole=["As", "Kh"], pot=15, stacks=[995, 990],
        to_call=5, min_raise=20, max_raise=1000, valid=["fold", "call", "raise"], history=PREFLOP,
    )  # fmt: skip
    obs = to_observation(state, 5, 10, 0)
    assert obs.bets == (5, 10)
    assert obs.stacks == (995, 990)
    assert obs.to_call == 5
    assert (obs.min_raise_to, obs.max_raise_to, obs.can_raise) == (20, 1000, True)
    assert obs.history == ()
    assert obs.in_hand == (True, True)
    assert [card_str(c) for c in obs.hole] == ["As", "Kh"]


def test_protocol_example_flop_facing_a_raise() -> None:
    history = [
        *PREFLOP,
        entry(0, "raise", 30),
        entry(1, "call", 20),
        entry(1, "check", 0, "flop"),
        entry(0, "raise", 40, "flop"),
    ]
    state = turn_request(  # Message 17
        seat=1, dealer=0, phase="flop", board=["Qs", "7h", "3d"], hole=["Jd", "Tc"], pot=100, stacks=[930, 970],
        to_call=40, min_raise=80, max_raise=970, valid=["fold", "call", "raise"], history=history,
    )  # fmt: skip
    obs = to_observation(state, 5, 10, 0)
    assert obs.street == 1
    assert obs.bets == (40, 0)
    assert obs.pot == 100
    assert obs.to_call == 40
    assert obs.history == ((0, 0, 3, 30), (0, 1, 2, 20), (1, 1, 1, 0), (1, 0, 3, 40))
    assert (obs.button, obs.small_blind, obs.big_blind) == (0, 5, 10)
    assert [card_str(c) for c in obs.board] == ["Qs", "7h", "3d"]


# ----------------------------------------------------------------------------
# Multiway tables, busted seats, short blinds
# ----------------------------------------------------------------------------
def test_multiway_with_busted_folded_and_all_in_seats() -> None:
    # Six chairs, seats 1 and 5 busted. Seat 2 posts the small blind and later calls all-in;
    # seat 4 raises, seat 0 folds, and the big blind (seat 3) must act.
    history = [
        entry(2, "post_small_blind", 50),
        entry(3, "post_big_blind", 100),
        entry(4, "raise", 300),
        entry(0, "fold", 0),
        entry(2, "call", 250),
    ]
    state = turn_request(
        seat=3, dealer=0, phase="preflop", board=[], hole=["Qh", "Qd"], pot=700, stacks=[1000, 0, 0, 900, 700, 0],
        to_call=200, min_raise=500, max_raise=1000, valid=["fold", "call", "raise"], history=history,
    )  # fmt: skip
    obs = to_observation(state, 50, 100, 0)
    assert obs.bets == (0, 0, 300, 100, 300, 0)
    assert obs.in_hand == (False, False, True, True, True, False)
    assert obs.n_in_hand == 3
    assert obs.to_call == 200


def test_blinds_read_from_the_hand_when_config_is_unknown() -> None:
    state = turn_request(
        seat=0, dealer=0, phase="preflop", board=[], hole=["2c", "7d"], pot=15, stacks=[995, 990],
        to_call=5, min_raise=20, max_raise=1000, valid=["fold", "call", "raise"], history=PREFLOP,
    )  # fmt: skip
    obs = to_observation(state)
    assert (obs.small_blind, obs.big_blind) == (5, 10)


def test_short_all_in_big_blind_keeps_the_posted_amount() -> None:
    # Protocol 5.1: the big blind posts 3 all-in, but to_call is still measured against the full 10.
    history = [entry(0, "post_small_blind", 5), entry(1, "post_big_blind", 3)]
    state = turn_request(
        seat=0, dealer=0, phase="preflop", board=[], hole=["Ah", "Ad"], pot=8, stacks=[995, 0],
        to_call=5, min_raise=20, max_raise=1000, valid=["fold", "call"], history=history,
    )  # fmt: skip
    obs = to_observation(state, 5, 10, 0)
    assert obs.bets == (5, 3)
    assert obs.to_call == 5
    assert obs.in_hand == (True, True)
    assert obs.big_blind == 10


def test_antes_stay_out_of_the_street_bets() -> None:
    history = [*PREFLOP, entry(0, "post_ante", 1), entry(1, "post_ante", 1)]
    state = turn_request(
        seat=0, dealer=0, phase="preflop", board=[], hole=["9s", "9h"], pot=17, stacks=[994, 989],
        to_call=5, min_raise=20, max_raise=999, valid=["fold", "call", "raise"], history=history,
    )  # fmt: skip
    obs = to_observation(state, 5, 10, 1)
    assert obs.bets == (5, 10)
    assert obs.ante == 1
    assert obs.history == ()


def test_a_state_without_blind_entries_still_shows_a_bet_to_match() -> None:
    # The SDK validator's smoke test sends a preflop state with an empty action_history.
    state = turn_request(
        seat=0, dealer=0, phase="preflop", board=[], hole=["Ah", "Kd"], pot=150, stacks=[9900, 9850],
        to_call=50, min_raise=200, max_raise=9900, valid=["fold", "call", "raise"], history=[],
    )  # fmt: skip
    obs = to_observation(state)
    assert max(obs.bets) == obs.bet + 50
    assert obs.big_blind == 100


# ----------------------------------------------------------------------------
# Mapping the agent's action onto valid_actions
# ----------------------------------------------------------------------------
FLOP_HISTORY = [*PREFLOP, entry(0, "raise", 30), entry(1, "call", 20), entry(1, "check", 0, "flop")]


def flop_state(valid: list[str], to_call: int = 0) -> GameState:
    return turn_request(
        seat=0, dealer=0, phase="flop", board=["Qs", "7h", "3d"], hole=["As", "Kh"], pot=60, stacks=[970, 970],
        to_call=to_call, min_raise=10, max_raise=970, valid=valid, history=FLOP_HISTORY,
    )  # fmt: skip


@pytest.mark.parametrize(
    ("agent_action", "valid", "to_call", "expected"),
    [
        (FOLD, ["check", "raise"], 0, ("check", 0)),  # a free check is never folded
        (CHECK, ["fold", "call", "raise"], 20, ("fold", 0)),  # checking is not possible facing a bet
        (CALL, ["check", "raise"], 0, ("check", 0)),
        (AgentAction.raise_to(5000), ["check", "raise"], 0, ("raise", 970)),  # clamped to the all-in
        (AgentAction.raise_to(20), ["fold", "call", "raise"], 20, ("call", 0)),  # not above the current bet
        (AgentAction.raise_to(40), ["fold", "call"], 20, ("call", 0)),  # raising is not offered
    ],
)
def test_action_mapping(agent_action: AgentAction, valid: list[str], to_call: int, expected: tuple[str, int]) -> None:
    state = flop_state(valid, to_call)
    action = to_chipzen(agent_action, to_observation(state, 5, 10, 0), state)
    assert (action.action, action.amount) == expected


def test_raise_below_the_minimum_is_raised_to_the_minimum() -> None:
    state = flop_state(["check", "raise"])
    obs = to_observation(state, 5, 10, 0)
    action = to_chipzen(AgentAction(ActionKind.RAISE, 8), obs, state)
    assert (action.action, action.amount) == ("raise", 10)


def test_bot_returns_a_legal_action_on_every_example() -> None:
    bot = MicroholdemBot()
    bot.on_match_start(
        {"your_seat": 0, "game_config": {"small_blind": 5, "big_blind": 10, "ante": 0, "num_players": 2}}
    )
    for valid, to_call in ((["check", "raise"], 0), (["fold", "call", "raise"], 20)):
        state = flop_state(valid, to_call)
        action = bot.decide(state)
        assert action.action in valid
        if action.action == "raise":
            assert min(state.min_raise, state.max_raise) <= action.amount <= state.max_raise


# ----------------------------------------------------------------------------
# Round trip against microholdem's own engine
# ----------------------------------------------------------------------------
class Recorder:
    """Plays like ``inner`` and keeps every Observation the engine hands it."""

    def __init__(self, inner: EngineBot) -> None:
        self.inner = inner
        self.name = inner.name
        self.seen: list[Observation] = []

    def reset(self, seat: int, n_seats: int, seed: int) -> None:
        self.inner.reset(seat, n_seats, seed)

    def act(self, obs: Observation) -> AgentAction:
        self.seen.append(obs)
        return self.inner.act(obs)


def as_chipzen_state(obs: Observation, blinds: list[dict[str, Any]]) -> GameState:
    """The turn_request a Chipzen server would send for the engine's Observation."""
    history = [
        *blinds,
        *(entry(seat, NAMES[kind], amount, PHASES[street]) for street, seat, kind, amount in obs.history),
    ]
    valid = (["check"] if obs.to_call == 0 else ["fold", "call"]) + (["raise"] if obs.can_raise else [])
    return turn_request(
        seat=obs.seat,
        dealer=obs.button,
        phase=PHASES[obs.street],
        board=[card_str(c) for c in obs.board],
        hole=[card_str(c) for c in obs.hole],
        pot=obs.pot,
        stacks=list(obs.stacks),
        to_call=obs.to_call,
        min_raise=obs.min_raise_to,
        max_raise=obs.max_raise_to,
        valid=valid,
        history=history,
        hand_number=obs.hand_number,
    )


def test_every_engine_observation_survives_the_round_trip() -> None:
    rng = random.Random(20261121)
    sb, bb = 5, 10
    pool: list[EngineBot] = [MicroAgent(seed=1, samples=64), MicroAgent(seed=2, samples=64)]
    pool += [
        make_archetype(name, seed=i)
        for i, name in enumerate(["granite", "jammer", "oracle", "blitz", "sponge", "scatter"])
    ]
    checked = 0
    for hand_number in range(1, 401):
        n = rng.randint(2, 6)
        stacks = [rng.choice([0, rng.randint(1, 25), rng.randint(25, 300), rng.randint(300, 3000)]) for _ in range(n)]
        for i in range(2):
            stacks[i] = stacks[i] or rng.randint(5, 2000)
        rng.shuffle(stacks)
        button = rng.choice([i for i in range(n) if stacks[i] > 0])
        bots = [Recorder(rng.choice(pool)) for _ in range(n)]
        cfg = HandConfig(small_blind=sb, big_blind=bb, time_limit_ns=10**9, record_latency=False)
        result = play_hand(bots, stacks, button, cfg, rng, hand_number)

        blinds = []
        if result.sb_seat is not None:
            blinds.append(entry(result.sb_seat, "post_small_blind", min(sb, result.start_stacks[result.sb_seat])))
        blinds.append(entry(result.bb_seat, "post_big_blind", min(bb, result.start_stacks[result.bb_seat])))
        for bot in bots:
            for obs in bot.seen:
                assert to_observation(as_chipzen_state(obs, blinds), sb, bb, 0) == obs
                checked += 1
    assert checked > 1500
