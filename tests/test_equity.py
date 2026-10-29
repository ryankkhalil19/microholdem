import math
import random

import numpy as np
import pytest

from microholdem import _core
from microholdem.cards import class_combo_count, parse_cards, parse_class
from microholdem.equity import Range, equity_known, equity_vs_random, equity_vs_ranges, exact_equity_hu
from microholdem.evaluator import evaluate


def test_exact_matches_known_matchups() -> None:
    assert exact_equity_hu(parse_cards("Ah Ad"), parse_cards("Kh Kd"), []) == pytest.approx(0.8264, abs=0.002)
    assert exact_equity_hu(parse_cards("Ah Kh"), parse_cards("Qd Qc"), []) == pytest.approx(0.4615, abs=0.003)
    # Symmetry: hero equity + villain equity == 1.
    a, b = parse_cards("Js Ts"), parse_cards("7c 7d")
    assert exact_equity_hu(a, b, []) + exact_equity_hu(b, a, []) == pytest.approx(1.0, abs=1e-12)


def test_monte_carlo_agrees_with_exact() -> None:
    rng = random.Random(3)
    for _ in range(6):
        cards = rng.sample(range(52), 7)
        hero, villain, board = cards[:2], cards[2:4], cards[4:7]
        exact = exact_equity_hu(hero, villain, board)
        n = 40_000
        mc = equity_vs_ranges(
            hero,
            board,
            [(np.asarray([sorted(villain)], dtype=np.int32), np.asarray([1.0]))],
            n,
            seed=rng.getrandbits(32),
        )
        se = math.sqrt(0.25 / n)
        assert abs(mc - exact) < 5 * se + 1e-9


def test_river_equity_is_deterministic_evaluation() -> None:
    hero, villain, board = parse_cards("Ah Kd"), parse_cards("Qs Qc"), parse_cards("As 7d 2c 9h 3s")
    assert exact_equity_hu(hero, villain, board) == 1.0
    assert evaluate([*hero, *board]) > evaluate([*villain, *board])
    eqs = equity_known([hero, villain], board)
    assert eqs == [1.0, 0.0]


def test_equity_vs_random_bounds_and_seed() -> None:
    a = equity_vs_random(parse_cards("Ah Ad"), [], 1, 20_000, seed=5)
    b = equity_vs_random(parse_cards("Ah Ad"), [], 1, 20_000, seed=5)
    assert a == b
    assert 0.83 < a < 0.88  # AA vs one random hand is about 85%
    multi = equity_vs_random(parse_cards("Ah Ad"), [], 7, 20_000, seed=5)
    assert 0.30 < multi < 0.40  # AA 8-way is about 35%
    with pytest.raises(ValueError):
        equity_vs_random(parse_cards("Ah Ad"), parse_cards("Ah 2c 3c"), 1, 100)


def test_equity_known_sums_to_one() -> None:
    hands = [parse_cards("Ah Kh"), parse_cards("Qd Qc"), parse_cards("7s 6s")]
    for board in ([], parse_cards("2c 3d 9h"), parse_cards("2c 3d 9h Ts")):
        eqs = equity_known(hands, board, n_samples=20_000 if len(board) == 0 else 0)
        assert sum(eqs) == pytest.approx(1.0, abs=1e-9)
        assert all(0.0 <= e <= 1.0 for e in eqs)


def test_range_parsing() -> None:
    assert Range.parse("22+").weights.sum() == 13
    assert Range.parse("A2s+").weights.sum() == 12
    assert Range.parse("KTo+").weights.sum() == 3
    assert Range.parse("T9s-65s").weights.sum() == 5
    r = Range.parse("22+, A2s+, KTo+")
    assert r.fraction() == pytest.approx((13 * 6 + 12 * 4 + 3 * 12) / 1326)
    combos, weights = r.combos()
    assert combos.shape == (13 * 6 + 12 * 4 + 3 * 12, 2)
    assert weights.shape[0] == combos.shape[0]
    assert r.contains(*parse_cards("Ah 2h")) == 1.0
    assert r.contains(*parse_cards("Ah 2d")) == 0.0
    with pytest.raises(ValueError):
        Range.parse("A2s-K2s")


def test_top_fraction_mass() -> None:
    order = list(range(169))
    r = Range.top_fraction(order, 0.10)
    assert r.fraction() == pytest.approx(0.10, abs=1e-9)
    total = sum(r.weights[i] * class_combo_count(i) for i in range(169))
    assert total == pytest.approx(132.6)


def test_class_tables_consistency() -> None:
    w = _core.class_pair_weights()
    assert w.shape == (169, 169)
    assert w.sum() == 1326 * 1225
    assert np.allclose(w, w.T)
    eq = _core.class_equity_matrix(1500, 2)
    assert eq.shape == (169, 169)
    assert np.allclose(eq + eq.T, 1.0, atol=1e-6)
    aa, kk, s72 = parse_class("AA"), parse_class("KK"), parse_class("72o")
    assert eq[aa, kk] > 0.78
    assert eq[s72, aa] < 0.16
    vs_random = _core.class_equity_vs_random(1, 3000, 4)
    assert vs_random[aa] > vs_random[kk] > vs_random[s72]
