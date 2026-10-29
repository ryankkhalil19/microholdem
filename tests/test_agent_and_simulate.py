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
    assert all(s.errors == 0 for s in stats)
    assert sum(s.timeouts for s in stats) <= 1 + sum(s.actions for s in stats) // 1000  # scheduler stalls
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


def _allin_obs(stacks: tuple[int, ...], bets: tuple[int, ...], payouts: tuple[float, ...]) -> Observation:
    pot = sum(bets)
    return Observation(
        seat=0,
        hole=(51, 50),
        board=(),
        street=0,
        pot=pot,
        to_call=min(max(bets) - bets[0], stacks[0]),
        min_raise_to=0,
        max_raise_to=0,
        can_raise=False,
        stacks=stacks,
        bets=bets,
        in_hand=tuple(True for _ in stacks),
        button=1,
        small_blind=50,
        big_blind=100,
        ante=0,
        n_in_hand=len(stacks),
        hand_number=0,
        history=((0, 1, 3, max(bets)),),
        payouts=payouts,
    )


def test_icm_call_values_the_place_hero_finishes_in_when_busting() -> None:
    agent = MicroAgent(seed=3)
    agent.reset(0, 2, 1)
    obs = _allin_obs((2900, 0), (100, 5000), (50.0, 30.0))  # heads-up, hero covered
    assert agent._icm_call_allin(obs, 0.60, 1)
    assert not agent._icm_call_allin(obs, 0.40, 1)
    # With two players ICM is linear in chips, so the break-even equals the chip break-even 2900/6000.
    lo, hi = 0.0, 1.0
    for _ in range(20):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if agent._icm_call_allin(obs, mid, 1) else (mid, hi)
    assert abs(hi - 2900 / 6000) < 0.01
    obs3 = _allin_obs((2900, 0, 7000), (100, 5000, 0), (50.0, 30.0, 20.0))  # three-handed, hero covered
    assert agent._icm_call_allin(obs3, 0.60, 1)


def test_icm_stack_vectors_conserve_the_chips_in_play() -> None:
    agent = MicroAgent(seed=4)
    agent.reset(0, 3, 1)
    seen: list[float] = []
    original = agent._icm

    def recording(stacks: list[float], payouts: tuple[float, ...], hero: int) -> float:
        seen.append(sum(stacks))
        return original(stacks, payouts, hero)

    agent._icm = recording  # type: ignore[method-assign]
    obs = _allin_obs((2900, 0, 7000), (100, 5000, 50), (50.0, 30.0, 20.0))
    agent._icm_call_allin(obs, 0.5, 1)
    jam_obs = Observation(
        seat=0,
        hole=(51, 47),
        board=(),
        street=0,
        pot=150,
        to_call=100,
        min_raise_to=200,
        max_raise_to=1500,
        can_raise=True,
        stacks=(1500, 1950, 1900),
        bets=(0, 50, 100),
        in_hand=(True, True, True),
        button=0,
        small_blind=50,
        big_blind=100,
        ante=0,
        n_in_hand=3,
        hand_number=0,
        history=(),
        payouts=(50.0, 30.0, 20.0),
    )
    agent._icm_jam_vs_fold(jam_obs, 150, 15.0, 2)
    assert len(seen) == 3 + 2 * 2 + 2  # fold, win, lose; then fold, two callers (win, lose each), no call
    assert all(total == pytest.approx(sum(obs.stacks) + obs.pot) for total in seen[:3])
    assert all(total == pytest.approx(sum(jam_obs.stacks) + jam_obs.pot) for total in seen[3:])


def test_fixed_sample_agent_plays_deterministically() -> None:
    def run() -> tuple[list[float], list[int]]:
        agent = MicroAgent(seed=5, samples=120)
        bots: list[Bot] = [agent, *lineup(9)]
        res = run_sng(bots, seed=77)
        return res.points, res.finish

    assert run() == run()


def test_sng_follows_the_dead_button_rule() -> None:
    from microholdem.engine import next_live

    checked = dead_small_blinds = 0
    for seed in range(4):
        bots: list[Bot] = [*lineup(seed), make_archetype("jammer", 50 + seed)]
        res = run_sng(bots, seed=seed, cfg=SngConfig(record_hands=True))
        for prev, cur in zip(res.history, res.history[1:], strict=False):
            alive = [s > 0 for s in prev.end_stacks]
            if sum(alive) <= 2:
                break
            assert cur.bb_seat == next_live(prev.bb_seat, alive)
            assert cur.sb_seat == (prev.bb_seat if alive[prev.bb_seat] else None)
            if prev.sb_seat is not None:
                assert cur.button == prev.sb_seat
            checked += 1
            dead_small_blinds += cur.sb_seat is None
    assert checked > 50 and dead_small_blinds > 0
