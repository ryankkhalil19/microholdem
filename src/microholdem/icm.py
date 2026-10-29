"""Independent Chip Model (Malmuth-Harville).

The C++ core implements the subset-DP version (``O(2^n * n)``); this module
adds a direct recursive reference for tests and tournament-level helpers
(risk premium, points allocation at a hand cap).
"""

from __future__ import annotations

from collections.abc import Sequence

from . import _core


def icm_equity(stacks: Sequence[float], payouts: Sequence[float]) -> list[float]:
    """Expected payout per player under ICM. Zero-stack players get zero."""
    return [float(x) for x in _core.icm_equity([float(s) for s in stacks], [float(p) for p in payouts])]


def icm_reference(stacks: Sequence[float], payouts: Sequence[float]) -> list[float]:
    """Slow recursive Malmuth-Harville (used to cross-check the C++ DP)."""
    n = len(stacks)
    out = [0.0] * n

    def rec(remaining: list[int], place: int, prob: float) -> None:
        if place >= len(payouts) or not remaining:
            return
        total = sum(stacks[i] for i in remaining)
        if total <= 0:
            return
        for i in remaining:
            if stacks[i] <= 0:
                continue
            p = prob * stacks[i] / total
            out[i] += p * payouts[place]
            rec([j for j in remaining if j != i], place + 1, p)

    rec(list(range(n)), 0, 1.0)
    return out


def icm_delta(
    stacks: Sequence[float], payouts: Sequence[float], hero: int, new_hero_stack: float, loser: int | None = None
) -> float:
    """Hero's ICM equity after a hypothetical chip transfer.

    If ``loser`` is given, the chips hero gains are taken from that player (and
    vice versa for a loss), and a transfer larger than the loser's stack raises.
    Otherwise only hero's stack changes and the total grows by the delta, which
    under ICM is the same as hero taking ``delta * (T - h) / (T + delta)`` chips
    from the others in proportion to their stacks (``T`` total, ``h`` hero's
    stack); a rough approximation used when several opponents are involved.
    """
    s = [float(x) for x in stacks]
    if new_hero_stack < 0:
        raise ValueError("hero's stack cannot be negative")
    delta = new_hero_stack - s[hero]
    s[hero] = new_hero_stack
    if loser is not None and loser != hero:
        if delta > s[loser]:
            raise ValueError("transfer exceeds the loser's stack")
        s[loser] -= delta
    return icm_equity(s, payouts)[hero]


def risk_premium(stacks: Sequence[float], payouts: Sequence[float], hero: int, villain: int, pot: float) -> float:
    """Equity hero needs to break even on an all-in for ``pot`` chips vs ``villain`` (ICM).

    Returns the required win probability; the chip-EV break-even would be the pot odds.
    A value above the pot odds is the ICM risk premium ("bubble factor").
    """
    if pot <= 0 or hero == villain:
        raise ValueError("pot must be positive and hero must differ from villain")
    s = [float(x) for x in stacks]
    base = icm_equity(s, payouts)[hero]
    win = list(s)
    win[hero] += pot
    win[villain] -= pot
    lose = list(s)
    lose[hero] -= pot
    lose[villain] += pot
    if lose[hero] < 0 or win[villain] < 0:
        raise ValueError("pot exceeds a stack")
    e_win = icm_equity(win, payouts)[hero]
    e_lose = icm_equity(lose, payouts)[hero]
    if e_win <= e_lose:
        return 1.0
    return max(0.0, min(1.0, (base - e_lose) / (e_win - e_lose)))


def allocate_points(stacks: Sequence[float], points_remaining: Sequence[float]) -> list[float]:
    """Split the points of the unfilled places among survivors by ICM (hand-cap rule)."""
    return icm_equity(stacks, points_remaining)
