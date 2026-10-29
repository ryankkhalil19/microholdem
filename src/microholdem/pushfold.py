"""Preflop push/fold solvers (CFR+ with regret matching+).

Heads-up
    SB (button) jams or folds; BB calls or folds. Two-player zero-sum, solved
    to a measured exploitability.

Multiway first-in
    A first-in player with ``k`` opponents behind jams or folds; each player
    behind calls or folds. Opponents are assumed to act independently and at
    most one call is resolved ("no overcall" model, the standard push/fold
    chart assumption). This is a general-sum approximation, so convergence is
    reported as the residual strategy change rather than exploitability.

Units: all chips are in big blinds. ``stack`` is the effective stack at the
start of the hand and includes anything a player posts. Payoffs are the
change in a player's stack over the hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


def conditional_deal_matrix(weights: F64) -> F64:
    """``P[a, b] = P(opponent holds class b | hero holds class a)`` from pair weights."""
    w = np.asarray(weights, dtype=np.float64)
    out: F64 = w / w.sum(axis=1, keepdims=True)
    return out


def class_prior(weights: F64) -> F64:
    """``P(a)``: marginal probability of each class (proportional to combo count)."""
    w = np.asarray(weights, dtype=np.float64)
    m = w.sum(axis=1)
    out: F64 = m / m.sum()
    return out


def _regret_match(regret: F64) -> F64:
    pos = np.maximum(regret, 0.0)
    total = pos.sum(axis=1, keepdims=True)
    uniform = np.full_like(regret, 1.0 / regret.shape[1])
    return np.where(total > 0, pos / np.where(total > 0, total, 1.0), uniform)


@dataclass(frozen=True)
class HeadsUpSolution:
    stack: float
    jam: F64  # P(jam | class) for the small blind
    call: F64  # P(call | class) for the big blind
    value_sb: float  # game value for SB in BB per hand
    exploitability: float  # average best-response gain, BB per hand
    iterations: int


def headsup_values(eq: F64, P: F64, stack: float, jam: F64, call: F64) -> tuple[F64, F64, F64, F64]:
    """Counterfactual action values for both players under (jam, call)."""
    S = float(stack)
    showdown_sb = (2.0 * eq - 1.0) * S  # SB net when called, per (a, b)
    u_jam = (P * (call[None, :] * showdown_sb + (1.0 - call[None, :]) * 1.0)).sum(axis=1)
    u_fold_sb = np.full_like(u_jam, -0.5)
    # BB facing a jam: reach-weighted by SB's jam frequency.
    reach = P * jam[None, :]  # P[b, a] * jam[a] with P symmetric in meaning (P[b, a] = P(a | b))
    u_call = (reach * (-showdown_sb.T)).sum(axis=1)
    u_fold_bb = (reach * (-1.0)).sum(axis=1)
    return u_jam, u_fold_sb, u_call, u_fold_bb


def solve_headsup(eq: F64, weights: F64, stack: float, iterations: int = 2000, tol: float = 1e-4) -> HeadsUpSolution:
    """CFR+ (alternating updates, linear averaging) for the heads-up push/fold game."""
    eq = np.asarray(eq, dtype=np.float64)
    P = conditional_deal_matrix(weights)
    prior = class_prior(weights)
    n = eq.shape[0]
    r_sb = np.zeros((n, 2))
    r_bb = np.zeros((n, 2))
    jam_sum = np.zeros(n)
    call_sum = np.zeros(n)
    wsum = 0.0
    jam: F64 = np.full(n, 0.5)
    call: F64 = np.full(n, 0.5)
    done = 0
    for t in range(1, iterations + 1):
        # SB update against BB's current strategy.
        u_jam, u_fold_sb, _, _ = headsup_values(eq, P, stack, jam, call)
        v_sb = jam * u_jam + (1.0 - jam) * u_fold_sb
        r_sb[:, 0] = np.maximum(r_sb[:, 0] + (u_jam - v_sb), 0.0)
        r_sb[:, 1] = np.maximum(r_sb[:, 1] + (u_fold_sb - v_sb), 0.0)
        jam = _regret_match(r_sb)[:, 0]
        # BB update against SB's updated strategy.
        _, _, u_call, u_fold_bb = headsup_values(eq, P, stack, jam, call)
        v_bb = call * u_call + (1.0 - call) * u_fold_bb
        r_bb[:, 0] = np.maximum(r_bb[:, 0] + (u_call - v_bb), 0.0)
        r_bb[:, 1] = np.maximum(r_bb[:, 1] + (u_fold_bb - v_bb), 0.0)
        call = _regret_match(r_bb)[:, 0]
        jam_sum += t * jam
        call_sum += t * call
        wsum += t
        done = t
        if t % 50 == 0 or t == iterations:
            jam_avg, call_avg = jam_sum / wsum, call_sum / wsum
            if headsup_exploitability(eq, P, prior, stack, jam_avg, call_avg) < tol:
                break
    jam_avg, call_avg = jam_sum / wsum, call_sum / wsum
    u_jam, u_fold_sb, _, _ = headsup_values(eq, P, stack, jam_avg, call_avg)
    value = float((prior * (jam_avg * u_jam + (1.0 - jam_avg) * u_fold_sb)).sum())
    expl = headsup_exploitability(eq, P, prior, stack, jam_avg, call_avg)
    return HeadsUpSolution(
        stack=stack, jam=jam_avg, call=call_avg, value_sb=value, exploitability=expl, iterations=done
    )


def headsup_exploitability(eq: F64, P: F64, prior: F64, stack: float, jam: F64, call: F64) -> float:
    """Average best-response gain over both players, in BB per hand (0 at equilibrium)."""
    u_jam, u_fold_sb, u_call, u_fold_bb = headsup_values(eq, P, stack, jam, call)
    br_sb = float((prior * np.maximum(u_jam, u_fold_sb)).sum())
    # BB's value: SB folds -> BB gains 0.5; SB jams -> BB best-responds per class.
    br_bb = float((prior * (1.0 - jam) * 0.5).sum() + (prior * np.maximum(u_call, u_fold_bb)).sum())
    return 0.5 * (br_sb + br_bb)


@dataclass(frozen=True)
class MultiwaySolution:
    stack: float
    n_behind: int
    jam: F64  # P(jam | class) for the first-in player
    calls: F64  # shape (n_behind, 169): P(call | class) for each player behind, in order
    ev_jam: F64  # chip EV of jamming per class (BB)
    residual: float  # max absolute strategy change over the last iteration
    iterations: int


def multiway_values(
    eq: F64, P: F64, stack: float, posts_behind: list[float], jam: F64, calls: F64
) -> tuple[F64, F64, F64]:
    """Jammer's jam value per class and each caller's (call, fold) values per class."""
    S = float(stack)
    k = len(posts_behind)
    W_all = float(sum(posts_behind))
    n = eq.shape[0]
    u_jam = np.zeros(n)
    survive = np.ones(n)  # probability nobody before i has called, per jammer class
    u_call = np.zeros((k, n))
    u_fold = np.zeros((k, n))
    for i in range(k):
        D_i = W_all - posts_behind[i]
        pot = 2.0 * S + D_i
        showdown_jammer = eq * pot - S  # (a, b): jammer net when i calls with b
        C_i = (P * calls[i][None, :]).sum(axis=1)  # P(i calls | a)
        V_i = (P * calls[i][None, :] * showdown_jammer).sum(axis=1)
        u_jam += survive * V_i
        survive = survive * (1.0 - C_i)
        reach = P * jam[None, :]  # (b, a): P(a | b) * jam[a]
        showdown_caller = eq * pot - S  # (b, a): caller net with b vs jammer a
        u_call[i] = (reach * showdown_caller).sum(axis=1)
        u_fold[i] = (reach * (-posts_behind[i])).sum(axis=1)
    u_jam += survive * W_all
    return u_jam, u_call, u_fold


def solve_multiway(
    eq: F64,
    weights: F64,
    stack: float,
    posts_behind: list[float],
    jammer_post: float,
    iterations: int = 1500,
) -> MultiwaySolution:
    """CFR+-style regret matching for the first-in jam game with ``len(posts_behind)`` players behind."""
    eq = np.asarray(eq, dtype=np.float64)
    P = conditional_deal_matrix(weights)
    n = eq.shape[0]
    k = len(posts_behind)
    r_j = np.zeros((n, 2))
    r_c = np.zeros((k, n, 2))
    jam: F64 = np.full(n, 0.5)
    calls: F64 = np.full((k, n), 0.5)
    jam_sum = np.zeros(n)
    call_sum = np.zeros((k, n))
    wsum = 0.0
    residual = 1.0
    for t in range(1, iterations + 1):
        u_jam, u_call, u_fold = multiway_values(eq, P, stack, posts_behind, jam, calls)
        u_fold_j = np.full(n, -jammer_post)
        v_j = jam * u_jam + (1.0 - jam) * u_fold_j
        r_j[:, 0] = np.maximum(r_j[:, 0] + (u_jam - v_j), 0.0)
        r_j[:, 1] = np.maximum(r_j[:, 1] + (u_fold_j - v_j), 0.0)
        new_jam = _regret_match(r_j)[:, 0]
        new_calls = np.empty_like(calls)
        for i in range(k):
            v_c = calls[i] * u_call[i] + (1.0 - calls[i]) * u_fold[i]
            r_c[i, :, 0] = np.maximum(r_c[i, :, 0] + (u_call[i] - v_c), 0.0)
            r_c[i, :, 1] = np.maximum(r_c[i, :, 1] + (u_fold[i] - v_c), 0.0)
            new_calls[i] = _regret_match(r_c[i])[:, 0]
        residual = float(max(np.abs(new_jam - jam).max(), np.abs(new_calls - calls).max()))
        jam, calls = new_jam, new_calls
        jam_sum += t * jam
        call_sum += t * calls
        wsum += t
    jam_avg = jam_sum / wsum
    calls_avg = call_sum / wsum
    u_jam, _, _ = multiway_values(eq, P, stack, posts_behind, jam_avg, calls_avg)
    return MultiwaySolution(
        stack=stack, n_behind=k, jam=jam_avg, calls=calls_avg, ev_jam=u_jam, residual=residual, iterations=iterations
    )


def posts_for_first_in(n_behind: int, ante: float = 0.0) -> tuple[list[float], float]:
    """Blinds/antes posted by the players behind a first-in jammer, and the jammer's own post.

    The last two players behind are the small and big blind. With one player
    behind the jammer is the small blind.
    """
    if n_behind < 1 or n_behind > 9:
        raise ValueError("n_behind must be in [1, 9]")
    if n_behind == 1:
        return [1.0 + ante], 0.5 + ante
    posts = [ante] * (n_behind - 2) + [0.5 + ante, 1.0 + ante]
    return posts, ante
