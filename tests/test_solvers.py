import numpy as np
import numpy.typing as npt
import pytest

from microholdem import _core
from microholdem.cards import parse_class
from microholdem.pushfold import (
    class_prior,
    conditional_deal_matrix,
    headsup_exploitability,
    posts_for_first_in,
    solve_headsup,
    solve_multiway,
)
from microholdem.rivercfr import RiverCFR, build_tree, decision_nodes, solve_river
from microholdem.tables import tables

EqW = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]


@pytest.fixture(scope="module")
def eq_and_weights() -> EqW:
    tb = tables()
    return tb.class_equity.astype(np.float64), _core.class_pair_weights()


def test_headsup_converges_and_matches_known_structure(eq_and_weights: EqW) -> None:
    eq, w = eq_and_weights
    aa, s72, k2o, a2o = parse_class("AA"), parse_class("72o"), parse_class("K2o"), parse_class("A2o")
    prev_jam = 2.0
    for stack in (3.0, 8.0, 15.0, 25.0):
        sol = solve_headsup(eq, w, stack, iterations=1500, tol=1e-4)
        assert sol.exploitability < 1e-3
        assert sol.jam[aa] > 0.99 and sol.call[aa] > 0.99
        jam_frac = float(
            sum(sol.jam[i] * (6 if i // 13 == i % 13 else 4 if i // 13 > i % 13 else 12) for i in range(169)) / 1326
        )
        assert jam_frac < prev_jam  # shorter stacks jam wider
        prev_jam = jam_frac
    deep = solve_headsup(eq, w, 20.0, iterations=1500, tol=1e-4)
    assert deep.jam[s72] < 0.01
    assert deep.jam[k2o] < 0.5
    short = solve_headsup(eq, w, 5.0, iterations=1500, tol=1e-4)
    assert short.jam[k2o] > 0.99 and short.call[a2o] > 0.99


def test_exploitability_is_zero_only_at_equilibrium(eq_and_weights: EqW) -> None:
    eq, w = eq_and_weights
    P, prior = conditional_deal_matrix(w), class_prior(w)
    sol = solve_headsup(eq, w, 10.0, iterations=1500, tol=1e-4)
    always_jam = np.ones(169)
    always_call = np.ones(169)
    assert headsup_exploitability(eq, P, prior, 10.0, always_jam, always_call) > 0.05
    assert headsup_exploitability(eq, P, prior, 10.0, sol.jam, sol.call) < 1e-3


def test_multiway_first_in_tightens_with_players_behind(eq_and_weights: EqW) -> None:
    eq, w = eq_and_weights
    fracs = []
    for k in (1, 3, 7):
        posts, jp = posts_for_first_in(k)
        sol = solve_multiway(eq, w, 10.0, posts, jp, iterations=300)
        assert sol.jam[parse_class("AA")] > 0.99
        assert sol.calls.shape == (k, 169)
        fracs.append(
            float(
                sum(sol.jam[i] * (6 if i // 13 == i % 13 else 4 if i // 13 > i % 13 else 12) for i in range(169)) / 1326
            )
        )
    assert fracs[0] > fracs[1] > fracs[2]


def test_river_tree_and_convergence() -> None:
    root = build_tree(2.0)
    keys = {n.key for n in decision_nodes(root)}
    assert {"", "x", "b0.5", "b1", "x b0.5", "x b1"} <= keys
    sol = solve_river(2.0, n_buckets=8, iterations=600)
    assert sol.exploitability < 2e-3
    for key, strat in sol.strategies.items():
        assert np.allclose(strat.sum(axis=1), 1.0)
        assert strat.shape[1] == len(sol.actions[key])
    # The nut bucket never folds facing a bet; the worst bucket never calls a pot-sized bet.
    acts = sol.actions["b1"]
    strat = sol.strategies["b1"]
    assert strat[-1][acts.index("f")] < 0.01
    assert strat[0][acts.index("c")] < 0.05


def test_river_uniform_strategy_is_exploitable() -> None:
    solver = RiverCFR(stack=2.0, n_buckets=6)
    assert solver.exploitability() > 0.05  # uniform strategies before any iteration
    solver.run(300)
    assert solver.exploitability() < 5e-3


def test_multiway_best_response_gain_decreases_and_k1_matches_headsup(eq_and_weights: EqW) -> None:
    eq, w = eq_and_weights
    posts, jp = posts_for_first_in(3)
    gains = [solve_multiway(eq, w, 10.0, posts, jp, iterations=it).br_gain for it in (50, 200, 800)]
    assert gains[0] > gains[1] > gains[2]
    assert gains[2] < 1e-4
    one = solve_multiway(eq, w, 10.0, [1.0], 0.5, iterations=1500)
    hu = solve_headsup(eq, w, 10.0, iterations=1500, tol=0.0)
    assert np.abs(one.jam - hu.jam).max() < 0.02
    assert np.abs(one.calls[0] - hu.call).max() < 0.02
