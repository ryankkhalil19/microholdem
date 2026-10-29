"""Hand evaluation: C++ fast path plus a slow pure-Python reference.

The reference evaluator is deliberately naive (enumerate every five-card
subset, classify by counting) so it can serve as an independent oracle in
the tests. The C++ evaluator must agree with it on every hand.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

from . import _core

CAT_SHIFT = int(_core.CAT_SHIFT)
CATEGORY_NAMES = (
    "high card",
    "one pair",
    "two pair",
    "three of a kind",
    "straight",
    "flush",
    "full house",
    "four of a kind",
    "straight flush",
)

evaluate = _core.evaluate
evaluate7 = _core.evaluate7
evaluate_mask = _core.evaluate_mask
evaluate_batch = _core.evaluate_batch
card_mask = _core.card_mask


def category(value: int) -> int:
    """Category index (0 = high card .. 8 = straight flush) of an evaluated value."""
    return value >> CAT_SHIFT


def category_name(value: int) -> str:
    return CATEGORY_NAMES[category(value)]


def best_of(cards: Sequence[int]) -> int:
    """Evaluate the best hand from 5 to 7 cards."""
    return int(_core.evaluate(list(cards)))


# --------------------------------------------------------------------------
# Pure-Python reference (slow, used only for verification).
# --------------------------------------------------------------------------
def _classify5(cards: Sequence[int]) -> tuple[int, tuple[int, ...]]:
    ranks = sorted((c >> 2 for c in cards), reverse=True)
    suits = {c & 3 for c in cards}
    counts: dict[int, int] = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    flush = len(suits) == 1
    distinct = sorted(counts, reverse=True)
    straight_top = -1
    if len(distinct) == 5:
        if distinct[0] - distinct[4] == 4:
            straight_top = distinct[0]
        elif distinct == [12, 3, 2, 1, 0]:
            straight_top = 3
    if flush and straight_top >= 0:
        return 8, (straight_top,)
    if groups[0][1] == 4:
        return 7, (groups[0][0], groups[1][0])
    if groups[0][1] == 3 and groups[1][1] == 2:
        return 6, (groups[0][0], groups[1][0])
    if flush:
        return 5, tuple(ranks)
    if straight_top >= 0:
        return 4, (straight_top,)
    if groups[0][1] == 3:
        return 3, (groups[0][0], groups[1][0], groups[2][0])
    if groups[0][1] == 2 and groups[1][1] == 2:
        return 2, (groups[0][0], groups[1][0], groups[2][0])
    if groups[0][1] == 2:
        return 1, (groups[0][0], groups[1][0], groups[2][0], groups[3][0])
    return 0, tuple(ranks)


def reference_rank(cards: Sequence[int]) -> tuple[int, tuple[int, ...]]:
    """Best (category, tiebreak tuple) over all five-card subsets. Slow."""
    if not 5 <= len(cards) <= 7:
        raise ValueError("need 5 to 7 cards")
    return max(_classify5(c) for c in combinations(cards, 5))
