"""Precomputed tables shipped with the package, and the script that builds them.

Build (takes a few minutes, deterministic seeds)::

    python -m microholdem.tables build

Tables
    class_equity.npy      169x169 class-vs-class preflop equity (Monte Carlo)
    preflop_equity.npy    9x169 class equity vs 1..9 random hands
    pushfold_hu.npz       heads-up jam/call frequencies on a stack grid
    pushfold_multi.npz    first-in jam and per-position call frequencies, 2..8 handed
    river.npz             bucketed river CFR strategies for several stack depths
    meta.json             parameters, seeds, timings and convergence metrics
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from bisect import bisect_left
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from . import _core
from .cards import N_CLASSES, class_combo_count
from .pushfold import posts_for_first_in, solve_headsup, solve_multiway
from .rivercfr import solve_river

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]

HU_STACKS: tuple[float, ...] = tuple(float(x) / 2.0 for x in range(2, 51))  # 1.0 .. 25.0 step 0.5
MULTI_STACKS: tuple[float, ...] = tuple(float(x) for x in range(2, 26))  # 2 .. 25 step 1
MAX_BEHIND = 7
# With bet sizes of 0.5 and 1 pot and one raise to 3x, the river tree commits at most 3 pots, so
# every stack of 3 pots or more is the same game; 4 is the deepest grid point for that reason.
RIVER_STACKS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
RIVER_BUCKETS = 10


def data_dir() -> Path:
    return Path(str(resources.files("microholdem").joinpath("data")))


def build_tables(
    out: Path,
    class_samples: int = 40_000,
    preflop_samples: int = 100_000,
    seed: int = 20261028,
    hu_iterations: int = 3000,
    multi_iterations: int = 2000,
    river_iterations: int = 4000,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {"seed": seed, "timings_s": {}, "convergence": {}}

    t0 = time.perf_counter()
    eq = _core.class_equity_matrix(class_samples, seed)
    np.save(out / "class_equity.npy", eq.astype(np.float32))
    meta["timings_s"]["class_equity"] = round(time.perf_counter() - t0, 2)
    meta["class_equity_samples_per_pair"] = class_samples
    print(f"class equity matrix: {meta['timings_s']['class_equity']} s", file=sys.stderr)

    t0 = time.perf_counter()
    pre = np.stack([_core.class_equity_vs_random(k, preflop_samples, seed + k) for k in range(1, 10)])
    np.save(out / "preflop_equity.npy", pre.astype(np.float32))
    meta["timings_s"]["preflop_equity"] = round(time.perf_counter() - t0, 2)
    meta["preflop_equity_samples_per_class"] = preflop_samples
    print(f"preflop equity vs 1..9: {meta['timings_s']['preflop_equity']} s", file=sys.stderr)

    weights = _core.class_pair_weights()
    eq64 = eq.astype(np.float64)

    t0 = time.perf_counter()
    jam = np.zeros((len(HU_STACKS), N_CLASSES), dtype=np.float32)
    call = np.zeros_like(jam)
    expl = np.zeros(len(HU_STACKS))
    value = np.zeros(len(HU_STACKS))
    iters = np.zeros(len(HU_STACKS), dtype=np.int64)
    for i, s in enumerate(HU_STACKS):
        hu_sol = solve_headsup(eq64, weights, s, iterations=hu_iterations, tol=5e-5)
        jam[i], call[i] = hu_sol.jam, hu_sol.call
        expl[i], value[i], iters[i] = hu_sol.exploitability, hu_sol.value_sb, hu_sol.iterations
    np.savez(
        out / "pushfold_hu.npz",
        stacks=np.asarray(HU_STACKS),
        jam=jam,
        call=call,
        exploitability=expl,
        value_sb=value,
        iterations=iters,
    )
    meta["timings_s"]["pushfold_hu"] = round(time.perf_counter() - t0, 2)
    meta["convergence"]["pushfold_hu_max_exploitability_bb"] = float(np.max(expl))
    meta["convergence"]["pushfold_hu_iterations_max"] = int(np.max(iters))
    print(
        f"heads-up push/fold: {meta['timings_s']['pushfold_hu']} s, max exploitability {np.max(expl):.2e} BB/hand",
        file=sys.stderr,
    )

    t0 = time.perf_counter()
    mjam = np.zeros((MAX_BEHIND + 1, len(MULTI_STACKS), N_CLASSES), dtype=np.float32)
    mcall = np.zeros((MAX_BEHIND + 1, len(MULTI_STACKS), MAX_BEHIND, N_CLASSES), dtype=np.float32)
    mresid = np.zeros((MAX_BEHIND + 1, len(MULTI_STACKS)))
    mgain = np.zeros((MAX_BEHIND + 1, len(MULTI_STACKS)))
    for k in range(1, MAX_BEHIND + 1):
        posts, jpost = posts_for_first_in(k)
        for i, s in enumerate(MULTI_STACKS):
            m_sol = solve_multiway(eq64, weights, s, posts, jpost, iterations=multi_iterations)
            mjam[k, i] = m_sol.jam
            mcall[k, i, :k] = m_sol.calls
            mresid[k, i] = m_sol.residual
            mgain[k, i] = m_sol.br_gain
    np.savez(
        out / "pushfold_multi.npz",
        stacks=np.asarray(MULTI_STACKS),
        jam=mjam,
        call=mcall,
        residual=mresid,
        br_gain=mgain,
    )
    meta["timings_s"]["pushfold_multi"] = round(time.perf_counter() - t0, 2)
    meta["convergence"]["pushfold_multi_max_br_gain_bb"] = float(np.max(mgain[1:]))
    meta["convergence"]["pushfold_multi_max_residual"] = float(np.max(mresid[1:]))
    print(
        f"multiway push/fold: {meta['timings_s']['pushfold_multi']} s, max best-response gain "
        f"{np.max(mgain[1:]):.2e} BB/hand, max residual {np.max(mresid[1:]):.2e}",
        file=sys.stderr,
    )

    t0 = time.perf_counter()
    river: dict[str, Any] = {"stacks": np.asarray(RIVER_STACKS), "n_buckets": np.asarray(RIVER_BUCKETS)}
    river_expl: dict[str, float] = {}
    for s in RIVER_STACKS:
        r_sol = solve_river(s, RIVER_BUCKETS, river_iterations)
        for key, strat in r_sol.strategies.items():
            river[f"S{s:g}|{key}"] = strat.astype(np.float32)
            river[f"A{s:g}|{key}"] = np.asarray(r_sol.actions[key])
        river_expl[f"{s:g}"] = r_sol.exploitability
    np.savez(out / "river.npz", **river)
    meta["timings_s"]["river"] = round(time.perf_counter() - t0, 2)
    meta["convergence"]["river_exploitability_pots"] = river_expl
    meta["river_iterations"] = river_iterations
    print(f"river CFR: {meta['timings_s']['river']} s, exploitability {river_expl}", file=sys.stderr)

    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


class Tables:
    """Lazy, read-only access to the shipped tables."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or data_dir()
        self.class_equity: F32 = np.load(self.root / "class_equity.npy")
        self.preflop_equity: F32 = np.load(self.root / "preflop_equity.npy")
        hu = np.load(self.root / "pushfold_hu.npz")
        self.hu_stacks: F64 = hu["stacks"]
        self._hu_grid: list[float] = [float(x) for x in self.hu_stacks]
        self.hu_jam: F32 = hu["jam"]
        self.hu_call: F32 = hu["call"]
        self.hu_exploitability: F64 = hu["exploitability"]
        multi = np.load(self.root / "pushfold_multi.npz")
        self.multi_stacks: F64 = multi["stacks"]
        self._multi_grid: list[float] = [float(x) for x in self.multi_stacks]
        self.multi_jam: F32 = multi["jam"]
        self.multi_call: F32 = multi["call"]
        if "br_gain" in multi.files:
            self.multi_br_gain: F64 = multi["br_gain"]
        else:
            self.multi_br_gain = np.full(multi["jam"].shape[:2], np.nan)
        river = np.load(self.root / "river.npz", allow_pickle=False)
        self.river_stacks: F64 = river["stacks"]
        self._river_grid: list[float] = [float(x) for x in self.river_stacks]
        self.river_buckets = int(river["n_buckets"])
        self.river_strategy: dict[tuple[float, str], F32] = {}
        self.river_actions: dict[tuple[float, str], list[str]] = {}
        for name in river.files:
            if name.startswith("S"):
                s, key = name[1:].split("|", 1)
                self.river_strategy[(float(s), key)] = river[name]
            elif name.startswith("A"):
                s, key = name[1:].split("|", 1)
                self.river_actions[(float(s), key)] = [str(a) for a in river[name]]
        self.meta: dict[str, Any] = json.loads((self.root / "meta.json").read_text())
        self.weights: F64 = _core.class_pair_weights()
        self.cond: F64 = self.weights / self.weights.sum(axis=1, keepdims=True)  # P(b | a)
        # Class ordering by equity vs 1 and vs 2 random hands (best first), and the rank fraction
        # of each class (0 = best) by combo mass.
        self.order_hu: npt.NDArray[np.int64] = np.argsort(-self.preflop_equity[0]).astype(np.int64)
        self.order_multi: npt.NDArray[np.int64] = np.argsort(-self.preflop_equity[2]).astype(np.int64)
        self.rank_hu: F64 = self._rank_fraction(self.order_hu)
        self.rank_multi: F64 = self._rank_fraction(self.order_multi)

    @staticmethod
    def _rank_fraction(order: npt.NDArray[np.int64]) -> F64:
        frac = np.zeros(N_CLASSES)
        acc = 0.0
        for idx in order:
            n = class_combo_count(int(idx))
            frac[idx] = (acc + n / 2.0) / 1326.0
            acc += n
        return frac

    # ---- push/fold lookups -------------------------------------------
    @staticmethod
    def _grid_index(stacks: F64 | list[float], stack: float) -> int:
        """Index of the nearest grid point (ascending grid; pure Python, no numpy ops)."""
        grid: list[float] = stacks if isinstance(stacks, list) else [float(x) for x in stacks]
        i = bisect_left(grid, stack)
        if i <= 0:
            return 0
        if i >= len(grid):
            return len(grid) - 1
        return i if grid[i] - stack < stack - grid[i - 1] else i - 1

    def pushfold_jam(self, stack_bb: float, cls: int, n_players: int) -> float:
        """P(jam) for a first-in player with ``n_players - 1`` opponents still in the hand."""
        k = max(1, min(MAX_BEHIND, n_players - 1))
        if k == 1:
            return float(self.hu_jam[self._grid_index(self._hu_grid, stack_bb), cls])
        return float(self.multi_jam[k, self._grid_index(self._multi_grid, stack_bb), cls])

    def pushfold_call(self, stack_bb: float, cls: int, n_players: int, caller_index: int | None = None) -> float:
        """P(call) facing a first-in jam. ``caller_index`` is the caller's position among
        the players behind the jammer (0 = first to act, last = big blind). Defaults to the big blind."""
        k = max(1, min(MAX_BEHIND, n_players - 1))
        if k == 1:
            return float(self.hu_call[self._grid_index(self._hu_grid, stack_bb), cls])
        ci = k - 1 if caller_index is None else max(0, min(k - 1, caller_index))
        return float(self.multi_call[k, self._grid_index(self._multi_grid, stack_bb), ci, cls])

    def jam_range(self, stack_bb: float, n_players: int) -> F32:
        k = max(1, min(MAX_BEHIND, n_players - 1))
        if k == 1:
            out: F32 = self.hu_jam[self._grid_index(self._hu_grid, stack_bb)]
            return out
        out = self.multi_jam[k, self._grid_index(self._multi_grid, stack_bb)]
        return out

    def call_ranges(self, stack_bb: float, n_players: int) -> F32:
        """Calling frequencies (k, 169) for every player behind a first-in jammer."""
        k = max(1, min(MAX_BEHIND, n_players - 1))
        if k == 1:
            out: F32 = self.hu_call[self._grid_index(self._hu_grid, stack_bb)][None, :]
            return out
        out = self.multi_call[k, self._grid_index(self._multi_grid, stack_bb), :k]
        return out

    def equity_vs_class_range(self, cls: int, rng: npt.NDArray[np.floating[Any]]) -> float:
        """Equity of ``cls`` against a weighted class range, with card-removal weights."""
        w = self.cond[cls] * rng
        tot = float(w.sum())
        if tot <= 0.0:
            return 0.5
        return float((w * self.class_equity[cls]).sum() / tot)

    def range_mass(self, cls: int, rng: npt.NDArray[np.floating[Any]]) -> float:
        """P(an opponent's hand is in ``rng`` | hero holds ``cls``)."""
        return float((self.cond[cls] * rng).sum())

    # ---- river lookups -------------------------------------------------
    def river_lookup(self, stack_pots: float, key: str) -> tuple[list[str], F32] | None:
        s = float(self.river_stacks[self._grid_index(self._river_grid, stack_pots)])
        acts = self.river_actions.get((s, key))
        if acts is None:
            return None
        return acts, self.river_strategy[(s, key)]


@lru_cache(maxsize=1)
def tables() -> Tables:
    return Tables()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="microholdem-tables", description="Build the precomputed tables.")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="generate all tables (deterministic)")
    b.add_argument("--out", type=Path, default=data_dir())
    b.add_argument("--quick", action="store_true", help="low sample counts for smoke testing")
    b.add_argument("--seed", type=int, default=20261028)
    s = sub.add_parser("show", help="print table metadata")
    s.add_argument("--root", type=Path, default=None)
    args = p.parse_args(argv)
    if args.cmd == "build":
        if args.quick:
            meta = build_tables(
                args.out,
                class_samples=2000,
                preflop_samples=5000,
                seed=args.seed,
                hu_iterations=400,
                multi_iterations=150,
                river_iterations=300,
            )
        else:
            meta = build_tables(args.out, seed=args.seed)
        print(json.dumps(meta, indent=2))
        return 0
    tb = Tables(args.root)
    print(json.dumps(tb.meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
