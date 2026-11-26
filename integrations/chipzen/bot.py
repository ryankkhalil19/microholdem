"""microholdem as a Chipzen bot.

A ``chipzen.Bot`` that plays No-Limit Hold'em with :class:`microholdem.headsup.HeadsUpAgent`, which
adapts to the opponent at two-seat tables and plays as ``MicroAgent`` at larger ones. Each
``turn_request`` is rebuilt into the ``Observation`` microholdem's own engine would give the acting
seat: per-seat stacks and street bets, who is still in the hand, the blinds and the action history
are all replayed from the hand's ``action_history`` (Chipzen POKER-GAME-STATE-PROTOCOL sections 3.3
and 3.4). The agent's action is then mapped onto ``valid_actions``, and every finished hand is fed
back to the agent's opponent model.

Hosted on Chipzen (the platform sets ``CHIPZEN_WS_URL`` and ``CHIPZEN_TOKEN``):  python bot.py
Remote play from your own machine:                                            chipzen run-external bot.py
"""

from __future__ import annotations

import asyncio
import os
import random
import sys
from typing import Any

from chipzen import Action, Bot, GameState
from chipzen.client import run_bot

from microholdem.cards import parse_card
from microholdem.engine import Action as AgentAction
from microholdem.engine import ActionKind, Observation
from microholdem.headsup import HeadsUpAgent, callable_view

# Chipzen allows 5 s per decision; microholdem was tuned for 1 ms. Some of the slack buys a larger
# Monte Carlo sample: at most 20,000 samples or 20 ms, a few milliseconds per postflop decision.
EQUITY_BUDGET_NS = 20_000_000
MAX_SAMPLES = 20_000
MIN_SAMPLES = 200

STREETS = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
PLAYER_ACTIONS = {
    "fold": ActionKind.FOLD,
    "check": ActionKind.CHECK,
    "call": ActionKind.CALL,
    "raise": ActionKind.RAISE,
}
BLIND_POSTS = ("post_small_blind", "post_big_blind")
# Fallback order on the wire for each agent action; a free check is never folded.
FALLBACKS = {
    ActionKind.RAISE: ("raise", "call", "check", "fold"),
    ActionKind.CALL: ("call", "check", "fold"),
    ActionKind.CHECK: ("check", "fold", "call"),
    ActionKind.FOLD: ("check", "fold", "call"),
}


def _int(value: object, default: int = 0) -> int:
    """A wire integer, or ``default`` for anything else."""
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def to_observation(state: GameState, small_blind: int = 0, big_blind: int = 0, ante: int = 0) -> Observation:
    """Rebuild the ``Observation`` microholdem's engine would give the acting seat.

    The blinds are read from the hand's own ``post_small_blind`` and ``post_big_blind`` entries,
    because Chipzen raises them during a match and ``match_start.game_config`` only names the
    opening level. ``small_blind`` and ``big_blind`` are lower bounds (the last level seen), which
    matter when a short stack posts less than the blind. Antes are dead money on Chipzen, so they
    stay out of the street bets.
    """
    n = len(state.opponent_stacks) + 1
    hero = state.your_seat
    opponents = iter(state.opponent_stacks)
    stacks = [state.your_stack if seat == hero else next(opponents) for seat in range(n)]
    street = STREETS.get(state.phase, 0)

    bets = [0] * n
    folded = [False] * n
    acted = [False] * n
    posted = dict.fromkeys(BLIND_POSTS, 0)
    history: list[tuple[int, int, int, int]] = []
    for entry in state.action_history:
        seat = _int(entry.get("seat"), -1)
        if not 0 <= seat < n:
            continue
        acted[seat] = True
        name = str(entry.get("action", ""))
        amount = _int(entry.get("amount"))
        entry_street = STREETS.get(str(entry.get("phase", "preflop")), 0)
        if name in posted:
            posted[name] = max(posted[name], amount)
            if entry_street == street:
                bets[seat] += amount
            continue
        kind = PLAYER_ACTIONS.get(name)
        if kind is None:  # post_ante: already in the pot, not part of the bet to match
            continue
        if kind == ActionKind.FOLD:
            folded[seat] = True
        elif entry_street == street and kind == ActionKind.CALL:
            bets[seat] += amount
        elif entry_street == street and kind == ActionKind.RAISE:
            bets[seat] = amount  # raises carry the raise-to total
        history.append((entry_street, seat, int(kind), amount))

    # A busted seat keeps its chair with a stack of 0 and has no entry this hand (protocol 5.9).
    in_hand = [not folded[s] and (s == hero or acted[s] or stacks[s] > 0) for s in range(n)]

    if not any(posted.values()) and max(bets) < bets[hero] + state.to_call:
        # No blind entries means an incomplete state (the SDK validator's smoke test sends one).
        # Give the bet to match to an opponent so the agent still sees a consistent table.
        rivals = [s for s in range(n) if s != hero and in_hand[s]] or [s for s in range(n) if s != hero]
        if rivals:
            bets[max(rivals, key=lambda s: bets[s])] = bets[hero] + state.to_call

    # A short stack posts less than the blind, so the level is the largest of what was posted,
    # twice the small blind, and the last level seen. With no blind entries, half the minimum raise.
    bb = max(big_blind, posted["post_big_blind"], 2 * posted["post_small_blind"]) or max(1, state.min_raise // 2)
    sb = max(small_blind, posted["post_small_blind"]) or max(1, bb // 2)
    can_raise = "raise" in state.valid_actions
    return Observation(
        seat=hero,
        hole=(parse_card(str(state.hole_cards[0])), parse_card(str(state.hole_cards[1]))),
        board=tuple(parse_card(str(card)) for card in state.board),
        street=street,
        pot=state.pot,
        to_call=min(state.to_call, state.your_stack),  # Chipzen caps a call at the stack itself
        min_raise_to=min(state.min_raise, state.max_raise) if can_raise else 0,
        max_raise_to=state.max_raise if can_raise else 0,
        can_raise=can_raise,
        stacks=tuple(stacks),
        bets=tuple(bets),
        in_hand=tuple(in_hand),
        button=state.dealer_seat,
        small_blind=sb,
        big_blind=bb,
        ante=ante,
        n_in_hand=sum(in_hand),
        hand_number=state.hand_number,
        history=tuple(history),
        payouts=(),  # chip EV: no payout structure for ICM
    )


def to_chipzen(action: AgentAction, obs: Observation, state: GameState) -> Action:
    """Map the agent's action onto ``valid_actions``."""
    kind = action.kind
    if kind == ActionKind.RAISE and action.amount <= obs.bet + state.to_call:
        kind = ActionKind.CALL  # a raise to the current bet or less is a call
    for name in FALLBACKS[kind]:
        if name not in state.valid_actions:
            continue
        if name == "raise":
            return Action.raise_to(max(min(state.min_raise, state.max_raise), min(state.max_raise, action.amount)))
        return Action(action=name)
    return Action.fold()


class MicroholdemBot(Bot):
    """microholdem's ``HeadsUpAgent`` behind the Chipzen protocol."""

    def __init__(self) -> None:
        self.agent = HeadsUpAgent(
            seed=random.getrandbits(32),
            equity_budget_ns=EQUITY_BUDGET_NS,
            max_samples=MAX_SAMPLES,
            min_samples=MIN_SAMPLES,
        )
        self.small_blind = 0
        self.big_blind = 0
        self.ante = 0
        self.seat = -1
        self.hand_stacks: list[int] = []  # stacks before the blinds, from round_start

    def on_match_start(self, match_info: dict[str, Any]) -> None:
        # A new match is a new opponent: what was learned about the last one does not carry over.
        self.agent.new_opponent()
        self.hand_stacks = []
        self._configure(match_info)

    def on_reconnected(self, message: dict[str, Any]) -> None:
        # A re-attached session gets ``reconnected`` instead of ``match_start``, same game_config.
        # It is the same match and opponent, so the model is kept.
        self._configure(message)

    def _configure(self, message: dict[str, Any]) -> None:
        config = message.get("game_config")
        config = config if isinstance(config, dict) else {}
        self.small_blind = _int(config.get("small_blind"))
        self.big_blind = _int(config.get("big_blind"))
        self.ante = _int(config.get("ante"))
        self.seat = _int(message.get("your_seat"), -1)
        seats = _int(config.get("num_players"))
        if self.seat >= 0 and seats >= 2:
            self.agent.reset(self.seat, seats, random.getrandbits(32))

    def on_round_start(self, message: dict[str, Any]) -> None:
        state = message.get("state")
        stacks = state.get("stacks") if isinstance(state, dict) else None
        self.hand_stacks = [_int(s) for s in stacks] if isinstance(stacks, list) else []

    def on_round_result(self, message: dict[str, Any]) -> None:
        """Feed the finished hand to the opponent model (two-seat tables only)."""
        try:
            result = message.get("result")
            entries = result.get("action_history") if isinstance(result, dict) else None
            if not isinstance(entries, list) or len(self.hand_stacks) != 2 or self.seat not in (0, 1):
                return
            history: list[tuple[int, int, int, int]] = []
            button = -1
            big_blind = self.big_blind
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("action", ""))
                seat, amount = _int(entry.get("seat"), -1), _int(entry.get("amount"))
                if name == "post_small_blind":
                    button = seat  # heads-up, the button posts the small blind
                    big_blind = max(big_blind, 2 * amount)
                elif name == "post_big_blind":
                    big_blind = max(big_blind, amount)
                elif name in PLAYER_ACTIONS and seat in (0, 1):
                    history.append((STREETS.get(str(entry.get("phase")), 0), seat, int(PLAYER_ACTIONS[name]), amount))
            if button in (0, 1):
                self.agent.observe_hand(history, self.seat, button, big_blind, min(self.hand_stacks))
        except Exception as exc:  # the model is an aid; never let it break a match
            print(f"microholdem: {type(exc).__name__}: {exc}; hand not counted", file=sys.stderr)

    def decide(self, state: GameState) -> Action:
        try:
            obs = to_observation(state, self.small_blind, self.big_blind, self.ante)
            # Blinds only go up: remember the level for hands where a short stack posts less.
            self.small_blind, self.big_blind = obs.small_blind, obs.big_blind
            self.seat = obs.seat
            return to_chipzen(self.agent.act(callable_view(obs)), obs, state)
        except Exception as exc:  # one malformed state must not cost more than this decision
            print(f"microholdem: {type(exc).__name__}: {exc}; checking or folding", file=sys.stderr)
            return Action.check() if "check" in state.valid_actions else Action.fold()


def main() -> None:
    """Entry point for the Docker image. The platform sets CHIPZEN_WS_URL and CHIPZEN_TOKEN."""
    url = os.environ.get("CHIPZEN_WS_URL") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not url:
        print("error: CHIPZEN_WS_URL not set and no URL passed on the command line", file=sys.stderr)
        sys.exit(1)
    asyncio.run(
        run_bot(
            url,
            MicroholdemBot(),
            token=os.environ.get("CHIPZEN_TOKEN"),
            ticket=os.environ.get("CHIPZEN_TICKET"),
        )
    )


if __name__ == "__main__":
    main()
