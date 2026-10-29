import random

import pytest

from microholdem.icm import allocate_points, icm_equity, icm_reference, risk_premium


def test_textbook_example() -> None:
    ev = icm_equity([5000, 3000, 2000], [50, 30, 20])
    assert ev == pytest.approx([38.39, 32.75, 28.86], abs=0.01)
    assert sum(ev) == pytest.approx(100.0)


def test_matches_reference_recursion() -> None:
    rng = random.Random(9)
    for n in (2, 3, 5, 8):
        for _ in range(5):
            stacks = [rng.randint(100, 5000) for _ in range(n)]
            pays = sorted((rng.random() * 100 for _ in range(min(n, 4))), reverse=True)
            fast = icm_equity(stacks, pays)
            slow = icm_reference(stacks, pays)
            assert fast == pytest.approx(slow, rel=1e-9, abs=1e-9)
            assert sum(fast) == pytest.approx(sum(pays), rel=1e-9)


def test_properties() -> None:
    assert icm_equity([1000, 1000, 1000, 1000], [50, 30, 20]) == pytest.approx([25.0] * 4)
    ev = icm_equity([6000, 1000, 0], [60, 40])
    assert ev[2] == 0.0
    assert ev[0] > ev[1]
    assert icm_equity([2000, 1000], [100]) == pytest.approx([200 / 3, 100 / 3])
    # A single payout equals chip-proportional equity.
    ev = icm_equity([5, 3, 2], [10])
    assert ev == pytest.approx([5, 3, 2])


def test_risk_premium_exceeds_chip_ev_on_bubble() -> None:
    # Four left, three paid, hero and villain are mid stacks: calling off needs more than pot odds.
    stacks = [4000, 3000, 3000, 2000]
    req = risk_premium(stacks, [50, 30, 20], hero=1, villain=2, pot=3000)
    assert req > 0.5  # chip-EV break-even for an even all-in is 50%
    # Winner-take-all payouts collapse to chip EV.
    req_wta = risk_premium(stacks, [100], hero=1, villain=2, pot=3000)
    assert req_wta == pytest.approx(0.5, abs=1e-9)


def test_allocate_points_at_hand_cap() -> None:
    pts = allocate_points([6000, 4000, 2000], [50, 30, 20])
    assert sum(pts) == pytest.approx(100.0)
    assert pts[0] > pts[1] > pts[2]
