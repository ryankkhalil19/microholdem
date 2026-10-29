import random
import time

import pytest

from microholdem.archetypes import make_archetype
from microholdem.engine import (
    CALL,
    CHECK,
    FOLD,
    Action,
    ActionKind,
    BotStats,
    HandConfig,
    Observation,
    play_hand,
)
from microholdem.evaluator import evaluate


class Scripted:
    """Replays a fixed list of actions; checks or calls once the script runs out."""

    name = "scripted"

    def __init__(self, actions: list[Action]) -> None:
        self.actions = list(actions)
        self.seen: list[Observation] = []

    def reset(self, seat: int, n_seats: int, seed: int) -> None:
        pass

    def act(self, obs: Observation) -> Action:
        self.seen.append(obs)
        if self.actions:
            return self.actions.pop(0)
        return CHECK if obs.to_call == 0 else CALL


class Sleeper:
    name = "sleeper"

    def reset(self, seat: int, n_seats: int, seed: int) -> None:
        pass

    def act(self, obs: Observation) -> Action:
        time.sleep(0.003)
        return Action.raise_to(obs.max_raise_to)


class Crasher:
    name = "crasher"

    def reset(self, seat: int, n_seats: int, seed: int) -> None:
        pass

    def act(self, obs: Observation) -> Action:
        raise RuntimeError("boom")


CFG = HandConfig(small_blind=50, big_blind=100)


def test_chip_conservation_random_play() -> None:
    bots = [make_archetype("scatter", i) for i in range(6)]
    rng = random.Random(1)
    stacks = [rng.randint(300, 5000) for _ in range(6)]
    total = sum(stacks)
    stats = [BotStats() for _ in range(6)]
    button = 0
    for h in range(300):
        if sum(1 for s in stacks if s > 0) < 2:
            break
        while stacks[button] == 0:
            button = (button + 1) % 6
        res = play_hand(bots, stacks, button, CFG, rng, h, stats)
        assert sum(stacks) == total
        assert all(s >= 0 for s in stacks)
        assert res.end_stacks == stacks
        button = (button + 1) % 6
    assert sum(s.actions for s in stats) > 0
    assert all(s.timeouts == 0 and s.errors == 0 for s in stats)


def test_side_pots_three_way_all_in() -> None:
    # Seat 0 (button) 300 chips, seat 1 (SB) 1000, seat 2 (BB) 2000. Everyone goes all in preflop.
    bots = [Scripted([Action.raise_to(300)]), Scripted([Action.raise_to(1000)]), Scripted([CALL])]
    stacks = [300, 1000, 2000]
    rng = random.Random(5)
    res = play_hand(bots, stacks, 0, CFG, rng, 0)  # type: ignore[arg-type]
    assert sum(stacks) == 3300
    assert res.showdown
    # Main pot 900 (300 x 3), side pot 1400 (700 x 2); seat 2 keeps its uncovered 1000.
    values = [evaluate([*res.hole[i], *res.board]) for i in range(3)]  # type: ignore[misc]
    best_main = max(values)
    main_winners = [i for i in range(3) if values[i] == best_main]
    best_side = max(values[1], values[2])
    side_winners = [i for i in (1, 2) if values[i] == best_side]
    expected = [0, 0, 1000]
    for i in main_winners:
        expected[i] += 900 // len(main_winners)
    for i in side_winners:
        expected[i] += 1400 // len(side_winners)
    assert stacks == expected


def test_uncalled_bet_returns_to_bettor() -> None:
    bots = [Scripted([Action.raise_to(900)]), Scripted([FOLD]), Scripted([FOLD])]
    stacks = [1000, 1000, 1000]
    res = play_hand(bots, stacks, 0, CFG, random.Random(1), 0)  # type: ignore[arg-type]
    assert not res.showdown
    assert stacks == [1150, 950, 900]
    assert res.winners == [0]


def test_heads_up_button_posts_small_blind_and_acts_first() -> None:
    a, b = Scripted([CALL]), Scripted([CHECK])
    stacks = [1000, 1000]
    play_hand([a, b], stacks, 0, CFG, random.Random(2), 0)  # type: ignore[list-item]
    first = a.seen[0]
    assert first.street == 0
    assert first.bets[0] == 50 and first.bets[1] == 100
    assert first.to_call == 50
    # Postflop the big blind (non-button) acts first.
    assert b.seen[1].street == 1 and b.seen[1].to_call == 0


def test_timeout_and_exception_fold() -> None:
    sleeper, crasher, tight = Sleeper(), Crasher(), Scripted([CALL])
    stacks = [1000, 1000, 1000]
    stats = [BotStats() for _ in range(3)]
    play_hand([sleeper, crasher, tight], stacks, 0, CFG, random.Random(3), 0, stats)  # type: ignore[list-item]
    assert stats[0].timeouts == 1 and stats[1].errors == 1
    # Sleeper (button) and crasher (SB) were folded; the big blind wins the blinds.
    assert stacks == [1000, 950, 1050]


def test_illegal_actions_are_legalized() -> None:
    # Checking when facing a bet is treated as an error and folds; a raise below the minimum is lifted.
    bots = [Scripted([Action.raise_to(150)]), Scripted([CHECK]), Scripted([CALL])]
    stacks = [1000, 1000, 1000]
    stats = [BotStats() for _ in range(3)]
    res = play_hand(bots, stacks, 0, CFG, random.Random(4), 0, stats)  # type: ignore[arg-type]
    assert stats[0].illegal == 1  # raise to 150 lifted to the 200 minimum
    assert stats[1].illegal == 1  # check facing a raise -> fold
    assert res.actions[0].amount == 200
    assert res.actions[1].kind == ActionKind.FOLD


def test_short_all_in_does_not_reopen_betting() -> None:
    # Seat 0 opens to 200, seat 1 (SB) jams 250 total (less than a full raise), seat 2 folds.
    # Seat 0 may call the 50 but not raise.
    opener = Scripted([Action.raise_to(200), Action.raise_to(1000)])
    bots = [opener, Scripted([Action.raise_to(250)]), Scripted([FOLD])]
    stacks = [1000, 250, 1000]
    res = play_hand(bots, stacks, 0, CFG, random.Random(6), 0)  # type: ignore[arg-type]
    last_obs = opener.seen[-1]
    assert last_obs.to_call == 50 and not last_obs.can_raise
    kinds = [a.kind for a in res.actions if a.seat == 0]
    assert kinds[-1] == ActionKind.CALL


def test_determinism() -> None:
    def run(seed: int) -> list[int]:
        bots = [make_archetype(n, 1) for n in ("blitz", "granite", "sponge", "oracle")]
        stacks = [2000] * 4
        rng = random.Random(seed)
        for h in range(40):
            if sum(1 for s in stacks if s > 0) < 2:
                break
            button = h % 4
            while stacks[button] == 0:
                button = (button + 1) % 4
            play_hand(bots, stacks, button, CFG, rng, h)
        return stacks

    assert run(42) == run(42)
    assert run(42) != run(43)


def test_rejects_bad_setup() -> None:
    with pytest.raises(ValueError):
        play_hand([Scripted([]), Scripted([])], [100, 0], 0, CFG, random.Random(1), 0)  # type: ignore[list-item]
