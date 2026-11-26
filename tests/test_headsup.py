from dataclasses import replace

import pytest

from microholdem.agent import MicroAgent
from microholdem.archetypes import make_archetype
from microholdem.cards import parse_cards
from microholdem.engine import ActionKind, Observation
from microholdem.headsup import HeadsUpAgent, OpponentModel, callable_view
from microholdem.simulate import run_duel

FOLD, CHECK, CALL, RAISE = (int(k) for k in (ActionKind.FOLD, ActionKind.CHECK, ActionKind.CALL, ActionKind.RAISE))


def obs(hole: str, **kw: object) -> Observation:
    """A heads-up observation for seat 0 at 50/100; override any field."""
    cards = parse_cards(hole)
    base: dict[str, object] = dict(
        seat=0,
        hole=(cards[0], cards[1]),
        board=(),
        street=0,
        pot=150,
        to_call=50,
        min_raise_to=200,
        max_raise_to=10_000,
        can_raise=True,
        stacks=(9_950, 9_900),
        bets=(50, 100),
        in_hand=(True, True),
        button=0,
        small_blind=50,
        big_blind=100,
        ante=0,
        n_in_hand=2,
        hand_number=1,
        history=(),
        payouts=(),
    )
    base.update(kw)
    return Observation(**base)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------
# callable_view
# ----------------------------------------------------------------------------
def test_callable_view_caps_a_covering_shove() -> None:
    # Big blind with 3,656 behind faces a 15,894 shove: only 3,956 of it can be called.
    o = obs("5h 4h", seat=1, button=0, pot=16_194, to_call=3_656, stacks=(0, 3_656), bets=(15_894, 300), big_blind=300)
    view = callable_view(o)
    assert view.bets == (3_956, 300)
    assert view.pot == 16_194 - (15_894 - 3_956)
    assert view.to_call / (view.pot + view.to_call) == pytest.approx(3_656 / 7_912)


def test_callable_view_leaves_a_callable_bet_alone() -> None:
    o = obs("As Ks", pot=400, to_call=250, bets=(50, 300), stacks=(9_950, 9_700))
    assert callable_view(o) is o


# ----------------------------------------------------------------------------
# OpponentModel
# ----------------------------------------------------------------------------
def test_model_starts_at_its_priors() -> None:
    m = OpponentModel()
    for stat, (mean, _weight) in OpponentModel.PRIORS.items():
        assert m.rate(stat) == pytest.approx(mean)


def test_model_counts_small_blind_opens_and_limps() -> None:
    m = OpponentModel()
    for _ in range(30):  # villain (seat 1) is the button and opens every time
        m.observe_hand([(0, 1, RAISE, 250), (0, 0, FOLD, 0)], hero=0, button=1, big_blind=100, effective_stack=10_000)
    assert m.rate("sb_open") > 0.9
    assert m.rate("sb_limp") < 0.05
    assert m.hands == 30


def test_model_counts_replies_to_our_open() -> None:
    m = OpponentModel()
    for _ in range(20):  # we open, villain re-raises to 9 big blinds
        m.observe_hand([(0, 0, RAISE, 250), (0, 1, RAISE, 900), (0, 0, FOLD, 0)], 0, 0, 100, 10_000)
    assert m.rate("bb_3bet") > 0.6
    assert m.rate("bb_fold") < 0.15
    assert m.rate("bb_jam_deep") < 0.03  # a normal re-raise is not a shove
    for _ in range(20):  # ... and now shoves over our open
        m.observe_hand([(0, 0, RAISE, 250), (0, 1, RAISE, 10_000), (0, 0, FOLD, 0)], 0, 0, 100, 10_000)
    assert m.rate("bb_jam_deep") > 0.3  # 20 shoves in 40 replies, pulled towards a 3% prior


def test_model_counts_postflop_bets_and_folds() -> None:
    m = OpponentModel()
    pre = [(0, 0, RAISE, 250), (0, 1, CALL, 150)]
    for _ in range(20):  # villain leads the flop; we fold
        m.observe_hand([*pre, (1, 1, RAISE, 250), (1, 0, FOLD, 0)], 0, 0, 100, 10_000)
    assert m.rate("bet") > 0.85
    before = m.rate("fold_to_bet")
    for _ in range(20):  # villain checks, we bet, villain folds
        m.observe_hand([*pre, (1, 1, CHECK, 0), (1, 0, RAISE, 250), (1, 1, FOLD, 0)], 0, 0, 100, 10_000)
    assert m.rate("fold_to_bet") > max(0.8, before)
    assert m.rate("bet") == pytest.approx((0.5 * 5 + 20) / (5 + 40))


# ----------------------------------------------------------------------------
# HeadsUpAgent
# ----------------------------------------------------------------------------
def test_defers_to_the_parent_with_more_than_two_seats() -> None:
    kw: dict[str, object] = dict(
        stacks=(9_950, 9_900, 10_000), bets=(50, 100, 0), in_hand=(True, True, True), n_in_hand=3, button=2
    )
    for hole in ("As Ks", "7d 2c", "9h 9s"):
        a, b = HeadsUpAgent(seed=5, samples=64), MicroAgent(seed=5, samples=64)
        assert a.act(obs(hole, **kw)) == b.act(obs(hole, **kw))


def test_prices_a_covering_shove_on_the_chips_it_can_lose() -> None:
    # Season 16, the last hand against 6BP9: the deployed bot called this with five-four suited.
    shove: dict[str, object] = dict(
        seat=1, button=0, pot=16_194, to_call=3_656, stacks=(0, 3_656), bets=(15_894, 300), small_blind=150,
        big_blind=300, can_raise=False, min_raise_to=0, max_raise_to=0, history=((0, 0, RAISE, 15_894),),
    )  # fmt: skip
    agent = HeadsUpAgent(seed=1, samples=64)
    assert agent.act(obs("5h 4h", **shove)).kind == ActionKind.FOLD
    assert agent.act(obs("As Ad", **shove)).kind == ActionKind.CALL


def test_continues_against_a_reraise_when_the_price_is_there() -> None:
    reraised: dict[str, object] = dict(
        pot=1_150, to_call=650, stacks=(9_750, 9_100), bets=(250, 900), min_raise_to=1_550,
        history=((0, 0, RAISE, 250), (0, 1, RAISE, 900)),
    )  # fmt: skip
    agent = HeadsUpAgent(seed=1, samples=64)
    # 650 to call into 1,150: 36% pot odds, 39% equity needed in position against a top-10% range.
    assert agent.act(obs("As Ks", **reraised)).kind in (ActionKind.CALL, ActionKind.RAISE)
    assert agent.act(obs("Ad Qc", **reraised)).kind == ActionKind.CALL
    assert agent.act(obs("Ah Ts", **reraised)).kind == ActionKind.FOLD
    assert agent.act(obs("7d 2c", **reraised)).kind == ActionKind.FOLD


def test_calls_a_frequent_reraiser_wider() -> None:
    reraised: dict[str, object] = dict(
        pot=1_150, to_call=650, stacks=(9_750, 9_100), bets=(250, 900), min_raise_to=1_550,
        history=((0, 0, RAISE, 250), (0, 1, RAISE, 900)),
    )  # fmt: skip
    agent = HeadsUpAgent(seed=1, samples=64)
    assert agent.act(obs("Kd 9s", **reraised)).kind == ActionKind.FOLD
    for _ in range(40):
        agent.observe_hand([(0, 0, RAISE, 250), (0, 1, RAISE, 900), (0, 0, FOLD, 0)], 0, 0, 100, 10_000)
    assert agent.act(obs("Kd 9s", **reraised)).kind == ActionKind.CALL


def short_open(hole: str, stack_bb: float, open_to: int = 645, big_blind: int = 300) -> Observation:
    """Big blind with ``stack_bb`` big blinds against a button open to ``open_to``; the button covers."""
    stack = int(stack_bb * big_blind)
    return obs(
        hole,
        seat=1,
        button=0,
        pot=open_to + big_blind,
        to_call=open_to - big_blind,
        min_raise_to=2 * open_to - big_blind,
        max_raise_to=stack,
        stacks=(20_000 - stack - open_to, stack - big_blind),
        bets=(open_to, big_blind),
        small_blind=big_blind // 2,
        big_blind=big_blind,
        history=((0, 0, RAISE, open_to),),
    )


def reply_mix(agent: HeadsUpAgent, stack_bb: float) -> dict[ActionKind, float]:
    """Share of the 1,326 starting hands the agent folds, calls and shoves in :func:`short_open`."""
    ranks = "23456789TJQKA"
    mix = dict.fromkeys((ActionKind.FOLD, ActionKind.CALL, ActionKind.RAISE), 0.0)
    for i, high in enumerate(ranks):
        for low in ranks[: i + 1]:
            hands = [(f"{high}h {low}d", 6)] if high == low else [(f"{high}s {low}s", 4), (f"{high}h {low}d", 12)]
            for hole, combos in hands:
                mix[agent.act(short_open(hole, stack_bb)).kind] += combos / 1326
    return mix


def test_a_short_stack_does_not_shove_every_hand_over_an_open() -> None:
    # Chatterbox v megaholdem, hands 71 and 79: nine-seven and eight-seven offsuit were shoved for
    # 19 and 17 big blinds over a 2.15x open, as the parent does with every hand at this depth.
    assert MicroAgent(seed=1, samples=64).act(short_open("9d 7c", 5_788 / 300)).kind == ActionKind.RAISE
    agent = HeadsUpAgent(seed=1, samples=64)
    assert agent.act(short_open("9d 7c", 5_788 / 300)).kind != ActionKind.RAISE
    assert agent.act(short_open("7h 8c", 5_215 / 300)).kind != ActionKind.RAISE
    assert agent.act(short_open("7d 2c", 19)).kind == ActionKind.FOLD
    for hole in ("As Ad", "Ah Kd", "8s 8d"):
        shove = agent.act(short_open(hole, 19))
        assert shove.kind == ActionKind.RAISE and shove.amount == 19 * 300


@pytest.mark.parametrize("stack_bb", [8, 12, 16, 20])
def test_short_reply_shoves_a_minority_of_hands_and_more_of_them_when_shorter(stack_bb: int) -> None:
    agent = HeadsUpAgent(seed=1, samples=64)
    mix = reply_mix(agent, stack_bb)
    assert 0.25 <= mix[ActionKind.RAISE] <= 0.55
    assert mix[ActionKind.FOLD] >= 0.10
    assert mix[ActionKind.RAISE] >= reply_mix(agent, 20)[ActionKind.RAISE] - 1e-9


def test_fold_to_shove_follows_the_price_and_then_the_opponent() -> None:
    agent = HeadsUpAgent(seed=1, samples=64)
    lo, hi = agent._villain_slice(short_open("9d 7c", 19))
    by_depth = [agent._fold_to_shove(short_open("9d 7c", n), lo, hi) for n in (6, 8, 12, 16, 20)]
    assert by_depth == sorted(by_depth)  # the better the raiser's price, the less it folds
    assert by_depth[0] == 0.0  # an open that is a third of the stack is never folded
    assert 0.45 <= by_depth[-1] <= 0.70

    def answers(kind: int) -> HeadsUpAgent:
        seen = HeadsUpAgent(seed=1, samples=64)
        reply = (0, 0, kind, 5_055 if kind == CALL else 0)
        for _ in range(12):
            seen.observe_hand([(0, 0, RAISE, 645), (0, 1, RAISE, 5_700), reply], 1, 0, 300, 5_700)
        return seen

    caller, folder = answers(CALL), answers(FOLD)
    assert caller.model.rate("fold_to_jam") < 0.2 < 0.8 < folder.model.rate("fold_to_jam")
    # Against a seat that folds its opens, any two cards are a shove; against one that calls, they are not.
    assert folder.act(short_open("7d 2c", 19)).kind == ActionKind.RAISE
    assert caller.act(short_open("9d 7c", 19)).kind != ActionKind.RAISE
    assert reply_mix(caller, 19)[ActionKind.RAISE] < 0.5


def test_a_small_reraise_is_not_counted_as_a_shove() -> None:
    m = OpponentModel()
    m.observe_hand([(0, 1, RAISE, 250), (0, 0, RAISE, 900), (0, 1, FOLD, 0)], 0, 1, 100, 10_000)
    assert m.seen["fold_to_3bet"] == 1 and m.seen["fold_to_jam"] == 0


def test_short_reply_leaves_other_spots_to_the_parent() -> None:
    assert HeadsUpAgent._is_short_reply(short_open("9d 7c", 20))
    assert not HeadsUpAgent._is_short_reply(short_open("9d 7c", 21))  # above the push-or-fold depth
    assert not HeadsUpAgent._is_short_reply(replace(short_open("9d 7c", 19), stacks=(0, 5_400)))  # the button is all-in
    assert not HeadsUpAgent._is_short_reply(obs("9d 7c", stacks=(1_950, 1_900)))  # small blind, nobody has raised


def flop(history: tuple[tuple[int, int, int, int], ...], street: int = 1) -> Observation:
    """We opened the button and were called; ``history`` is the postflop action so far."""
    pre = ((0, 0, RAISE, 250), (0, 1, CALL, 150))
    board = tuple(parse_cards("Kh 8d 3c" if street == 1 else "Kh 8d 3c 2s"))
    return obs("Ah 8s", street=street, board=board, pot=500, to_call=0, bets=(0, 0), stacks=(9_750, 9_750),
               min_raise_to=100, history=(*pre, *history))  # fmt: skip


def test_standing_rises_with_equity_and_falls_with_each_bet() -> None:
    agent = HeadsUpAgent(seed=1, samples=64)
    unbet = flop(())
    assert agent._standing(unbet, 0.3) < agent._standing(unbet, 0.6) < agent._standing(unbet, 0.9)
    assert agent._standing(unbet, 0.6) == pytest.approx(0.6)
    one = flop(((1, 1, RAISE, 250),))
    two = flop(((1, 1, RAISE, 250), (1, 0, CALL, 250), (2, 1, RAISE, 600)), street=2)
    assert agent._standing(two, 0.6) < agent._standing(one, 0.6) < 0.6
    assert agent._standing(one, 0.97) > 0.9  # a near-nut hand stays ahead of a betting range


def test_standing_reads_a_frequent_bettor_as_weaker() -> None:
    bet = flop(((1, 1, RAISE, 250),))
    pre = [(0, 0, RAISE, 250), (0, 1, CALL, 150)]
    maniac, rock = HeadsUpAgent(seed=1, samples=64), HeadsUpAgent(seed=1, samples=64)
    for _ in range(40):
        maniac.observe_hand([*pre, (1, 1, RAISE, 250), (1, 0, FOLD, 0)], 0, 0, 100, 10_000)
        rock.observe_hand([*pre, (1, 1, CHECK, 0), (1, 0, CHECK, 0)], 0, 0, 100, 10_000)
    assert maniac._standing(bet, 0.55) > rock._standing(bet, 0.55) + 0.2


def test_a_check_and_a_call_move_the_standing_the_right_way() -> None:
    agent = HeadsUpAgent(seed=1, samples=64)
    checked = flop(((1, 1, CHECK, 0),))
    assert agent._standing(checked, 0.6) > 0.6  # a check takes strong hands out of the range
    called = flop(((1, 1, CHECK, 0), (1, 0, RAISE, 250), (1, 1, CALL, 250)), street=2)
    assert agent._standing(called, 0.6) < agent._standing(checked, 0.6)  # the weakest hands folded


@pytest.mark.parametrize("opponent", ["oracle", "blitz", "sponge", "jammer"])
def test_plays_legal_actions_through_the_engine(opponent: str) -> None:
    agent = HeadsUpAgent(seed=4, samples=64)
    for stack in (10_000, 1_500):
        res = run_duel(agent, make_archetype(opponent, 9), 300, seed=11, stack=stack, time_limit_ns=10**12)
        assert res.stats[0].errors == 0 and res.stats[0].illegal == 0


def test_new_opponent_resets_the_model() -> None:
    agent = HeadsUpAgent(seed=1, samples=64)
    agent.observe_hand([(0, 1, RAISE, 250), (0, 0, FOLD, 0)], 0, 1, 100, 10_000)
    assert agent.model.hands == 1
    agent.new_opponent()
    assert agent.model.hands == 0
