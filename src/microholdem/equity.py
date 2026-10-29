"""Equity estimation: Monte Carlo and exact, plus weighted ranges.

Everything heavy runs in C++; this module parses ranges, compiles them to
contiguous arrays and keeps the call surface small.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from . import _core
from .cards import N_CLASSES, RANKS, class_combos, class_index, parse_class

RangeArrays = tuple[npt.NDArray[np.int32], npt.NDArray[np.float64]]


@dataclass(frozen=True)
class Range:
    """A weighted set of preflop classes (weights in ``[0, 1]`` per class)."""

    weights: npt.NDArray[np.float64]  # shape (169,)

    @staticmethod
    def from_classes(classes: Iterable[int], weight: float = 1.0) -> Range:
        w = np.zeros(N_CLASSES, dtype=np.float64)
        for c in classes:
            w[c] = weight
        return Range(w)

    @staticmethod
    def parse(text: str) -> Range:
        """Parse a compact range string such as ``"22+, A2s+, KTo+, QJs"``.

        Supported tokens: ``AA``, ``77+`` (pairs and up), ``AKs``, ``A5s+``
        (suited, kicker and up), ``KTo+``, ``T9s-65s`` (connected run) and
        a bare class. Whitespace and commas separate tokens.
        """
        w = np.zeros(N_CLASSES, dtype=np.float64)
        for tok in text.replace(",", " ").split():
            for idx in _expand_token(tok):
                w[idx] = 1.0
        return Range(w)

    @staticmethod
    def random() -> Range:
        return Range(np.ones(N_CLASSES, dtype=np.float64))

    @staticmethod
    def top_fraction(order: Sequence[int], fraction: float) -> Range:
        """Top ``fraction`` of hands by combo count, following ``order`` (best first)."""
        w = np.zeros(N_CLASSES, dtype=np.float64)
        budget = fraction * 1326.0
        for idx in order:
            n = len(class_combos(idx))
            if budget <= 0:
                break
            if n <= budget:
                w[idx] = 1.0
                budget -= n
            else:
                w[idx] = budget / n
                budget = 0
        return Range(w)

    def combos(self) -> RangeArrays:
        """Expand to ``(M, 2)`` combos with per-combo weights (one row per combo)."""
        rows: list[tuple[int, int]] = []
        ws: list[float] = []
        for idx in np.flatnonzero(self.weights > 0):
            wt = float(self.weights[idx])
            for a, b in class_combos(int(idx)):
                rows.append((a, b))
                ws.append(wt)
        if not rows:
            raise ValueError("empty range")
        return np.asarray(rows, dtype=np.int32), np.asarray(ws, dtype=np.float64)

    def compile(self) -> _core.CompiledRange:
        combos, ws = self.combos()
        return _core.CompiledRange(combos, ws)

    def fraction(self) -> float:
        total = sum(float(self.weights[i]) * len(class_combos(i)) for i in range(N_CLASSES))
        return total / 1326.0

    def contains(self, c1: int, c2: int) -> float:
        return float(self.weights[class_index(c1, c2)])


def _expand_token(tok: str) -> list[int]:
    t = tok.strip()
    plus = t.endswith("+")
    if plus:
        t = t[:-1]
    if "-" in t:
        a, b = t.split("-", 1)
        ia, ib = parse_class(a), parse_class(b)
        ra, ca = divmod(ia, 13)
        rb, cb = divmod(ib, 13)
        if (ra - ca) != (rb - cb):
            raise ValueError(f"range run must keep the same gap: {tok!r}")
        lo, hi = min(ra, rb), max(ra, rb)
        gap = ra - ca
        return [(r * 13 + (r - gap)) for r in range(lo, hi + 1)]
    idx = parse_class(t)
    if not plus:
        return [idx]
    row, col = divmod(idx, 13)
    if row == col:  # pair and up
        return [r * 13 + r for r in range(row, 13)]
    if row > col:  # suited: hi fixed, kicker up to hi-1
        return [row * 13 + k for k in range(col, row)]
    return [k * 13 + col for k in range(row, col)]  # offsuit


def equity_vs_random(hero: Sequence[int], board: Sequence[int], n_opp: int, n_samples: int, seed: int = 1) -> float:
    """Monte Carlo equity of ``hero`` against ``n_opp`` uniformly random hands."""
    return float(_core.equity_vs_random(list(hero), list(board), n_opp, n_samples, seed))


def equity_vs_ranges(
    hero: Sequence[int],
    board: Sequence[int],
    ranges: Sequence[Range | RangeArrays],
    n_samples: int,
    seed: int = 1,
) -> float:
    """Monte Carlo equity of ``hero`` against opponents holding weighted ranges."""
    arrs: list[RangeArrays] = [r.combos() if isinstance(r, Range) else r for r in ranges]
    return float(_core.equity_vs_ranges(list(hero), list(board), arrs, n_samples, seed))


def equity_vs_compiled(
    hero: Sequence[int],
    board: Sequence[int],
    ranges: Sequence[_core.CompiledRange],
    n_samples: int,
    seed: int = 1,
) -> float:
    """Same as :func:`equity_vs_ranges` but with pre-compiled ranges (agent hot path)."""
    return float(_core.equity_vs_compiled(list(hero), list(board), list(ranges), n_samples, seed))


def exact_equity_hu(hero: Sequence[int], villain: Sequence[int], board: Sequence[int]) -> float:
    """Exact heads-up equity by full enumeration of the remaining board."""
    win, tie, lose = _core.enumerate_equity_hu(list(hero), list(villain), list(board))
    total = win + tie + lose
    return (win + 0.5 * tie) / total


def equity_known(
    hands: Sequence[Sequence[int]], board: Sequence[int], n_samples: int = 0, seed: int = 1
) -> list[float]:
    """Equities of several known hands. Exact unless more than two cards are to come and ``n_samples > 0``."""
    return [float(x) for x in _core.equity_known_hands([list(h) for h in hands], list(board), n_samples, seed)]


def class_grid(values: Sequence[float]) -> str:
    """Render a 169-vector as a 13x13 text grid (A at the top-left)."""
    lines = []
    for row in range(12, -1, -1):
        cells = []
        for col in range(12, -1, -1):
            cells.append(f"{values[row * 13 + col]:5.2f}")
        lines.append(RANKS[row] + " " + " ".join(cells))
    return "\n".join(lines)
