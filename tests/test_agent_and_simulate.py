import random

import pytest

from microholdem.agent import MicroAgent
from microholdem.archetypes import all_archetypes, make_archetype
from microholdem.engine import ActionKind, Bot, Observation
from microholdem.simulate import SngConfig, latency_summary, run_duel, run_ring, run_sng

LATENCY_LIMIT_US = 1000.0


def lineup(seed: int) -> list[Bot]:
    return [make_archetype(n, seed + i) for i, n in enumerate(all_archetypes())]


def test_agent_plays_legal_actions_fast_in_sng() -> None:
    agent = MicroAgent(seed=1)
    stats = []
    points_total = 0.0
    for t in range(12):
        bots: list[Bot] = [agent, *lineup(t)]
        random.Random(t).shuffle(bots)
        res = run_sng(bots, seed=100 + t)
        idx = bots.index(agent)
        st = res.stats[idx]
        assert st.errors == 0 and st.illegal == 0
        assert st.timeouts <= 1 + st.actions // 1000  # allow 0.1% for GC and scheduler stalls
        assert sorted(res.finish) == list(range(1, 9))
        assert sum(res.points) == pytest.approx(100.0)
        assert res.hands <= SngConfig().hands_cap
        points_total += res.points[idx]
        stats.append(st)
    lat = latency_summary(stats)
    assert lat["actions"] > 500
    assert lat["p99_us"] < LATENCY_LIMIT_US
    assert points_total / 12 > 100.0 / 8  # better than an average seat


def test_agent_heads_up_duel() -> None:
    agent = MicroAgent(seed=2)
    res = run_duel(agent, make_archetype("oracle", 3), 600, seed=7)
    st = res.stats[0]
    assert st.errors == 0 and st.illegal == 0
    assert st.timeouts <= 1 + st.actions // 1000
    lat = latency_summary(res.stats, [0])
    assert lat["p99_us"] < LATENCY_LIMIT_US
    assert res.hands == 600


def test_agent_decides_on_hand_crafted_observations() -> None:
    agent = MicroAgent(seed=3)
    agent.reset(0, 2, 1)

    def obs(**kw: object) -> Observation:
        base = dict(
            seat=0,
            hole=(51, 47),  # As Ks
            board=(),
            street=0,
            pot=150,
            to_call=50,
            min_raise_to=200,
            max_raise_to=1000,
            can_raise=True,
            stacks=(950, 900),
            bets=(50, 100),
            in_hand=(True, True),
            button=0,
            small_blind=50,
            big_blind=100,
            ante=0,
            n_in_hand=2,
            hand_number=0,
            history=(),
            payouts=(),
        )
        base.update(kw)
        return Observation(**base)  # type: ignore[arg-type]

    # 10 BB effective heads-up with AKs: the Nash chart jams.
    a = agent.act(obs())
    assert a.kind == ActionKind.RAISE and a.amount == 1000
    # 72o at 15 BB heads-up folds.
    a = agent.act(obs(hole=(20, 1), stacks=(1450, 1400), max_raise_to=1500))
    assert a.kind == ActionKind.FOLD
    # Deep stack AKs opens for a raise.
    a = agent.act(obs(stacks=(9950, 9900), max_raise_to=10000))
    assert a.kind == ActionKind.RAISE and 200 <= a.amount <= 300
    # Facing an all-in with AA under ICM: call.
    a = agent.act(
        obs(
            hole=(51, 50),
            to_call=900,
            stacks=(950, 0),
            bets=(50, 1000),
            pot=1050,
            can_raise=False,
            min_raise_to=0,
            max_raise_to=0,
            history=((0, 1, 3, 1000),),
            payouts=(50.0, 30.0, 20.0),
        )
    )
    assert a.kind == ActionKind.CALL
    # River heads-up with the nuts facing a pot bet: never folds.
    a = agent.act(
        obs(
            hole=(51, 47),
            board=(50, 46, 2, 7, 13),  # Ad Kd 2h 3s 5d: two pair aces and kings
            street=3,
            pot=600,
            to_call=300,
            stacks=(2000, 1700),
            bets=(0, 300),
            min_raise_to=600,
            max_raise_to=2000,
            history=(
                (0, 0, 3, 200),
                (0, 1, 2, 100),
                (1, 1, 1, 0),
                (1, 0, 1, 0),
                (2, 1, 1, 0),
                (2, 0, 1, 0),
                (3, 1, 3, 300),
            ),
        )
    )
    assert a.kind in (ActionKind.CALL, ActionKind.RAISE)


def test_archetypes_run_in_ring_and_duel() -> None:
    bots = lineup(1)
    net, stats = run_ring(bots, 150, seed=3)
    assert sum(net) == 0
    assert all(s.timeouts == 0 and s.errors == 0 for s in stats)
    res = run_duel(make_archetype("blitz", 1), make_archetype("granite", 2), 200, seed=4)
    assert res.hands == 200
    assert res.wins[0] + res.wins[1] + res.wins[2] == 200


def test_sng_hand_cap_allocates_points_by_icm() -> None:
    bots: list[Bot] = [make_archetype("passenger", i) for i in range(8)]
    res = run_sng(bots, seed=1, cfg=SngConfig(hands_cap=20))
    assert res.capped
    assert sum(res.points) == pytest.approx(100.0)
    assert res.hands == 20
    assert sorted(res.finish) == list(range(1, 9))
