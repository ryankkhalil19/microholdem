"""Benchmark harness.

Reports evaluator and equity throughput, ICM cost, solver convergence, the
agent's per-action latency distribution (p50 / p90 / p99 / p99.9 / max) in
heads-up duel, 8-max sit-and-go and 8-max ring play, its first actions after
construction, fresh-process startup costs, simulator hands per second, and a
deterministic strength table (the agent plays with a fixed sample count there,
so the table reproduces exactly from the seeds). Writes ``bench/results.json``
and ``bench/results.md``.

Run::

    python -m microholdem.bench --out bench
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__, _core
from .agent import MicroAgent, describe
from .archetypes import make_archetype
from .cards import parse_cards
from .engine import Bot, BotStats
from .equity import Range
from .pushfold import solve_headsup
from .rivercfr import solve_river
from .simulate import SngConfig, latency_summary, run_duel, run_ring, run_sng
from .tables import tables

LINEUP = ("granite", "blitz", "sponge", "oracle", "jammer", "scatter", "passenger")


def _cpu_name() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def bench_evaluator(n: int = 20_000_000) -> dict[str, float]:
    t0 = time.perf_counter()
    _core.evaluate_random7(n, 1)
    dt = time.perf_counter() - t0
    rng = random.Random(3)
    hands = np.asarray([rng.sample(range(52), 7) for _ in range(50_000)], dtype=np.int32)
    t0 = time.perf_counter()
    for _ in range(4):
        _core.evaluate_batch(hands)
    dt_batch = (time.perf_counter() - t0) / 4
    cards = [51, 47, 43, 39, 35, 2, 7]
    reps = 200_000
    t0 = time.perf_counter_ns()
    for _ in range(reps):
        _core.evaluate(cards)
    per_call = (time.perf_counter_ns() - t0) / reps
    return {
        "evals_per_sec_cpp_loop": n / dt,
        "ns_per_eval_cpp_loop": dt * 1e9 / n,
        "evals_per_sec_numpy_batch": len(hands) / dt_batch,
        "ns_per_python_call": per_call,
    }


def bench_equity() -> dict[str, float]:
    out: dict[str, float] = {}
    hero = parse_cards("Ah Kh")
    n = 2_000_000
    t0 = time.perf_counter()
    _core.equity_vs_random(hero, [], 1, n, 1)
    out["hu_preflop_samples_per_sec"] = n / (time.perf_counter() - t0)
    board = parse_cards("Qs Jd 2c")
    t0 = time.perf_counter()
    _core.equity_vs_random(hero, board, 1, n, 1)
    out["hu_flop_samples_per_sec"] = n / (time.perf_counter() - t0)
    rng = Range.parse("22+, A2s+, K9s+, QTs+, JTs, ATo+, KQo").compile()
    n = 500_000
    t0 = time.perf_counter()
    _core.equity_vs_compiled(hero, board, [rng], n, 1)
    out["hu_flop_vs_range_samples_per_sec"] = n / (time.perf_counter() - t0)
    t0 = time.perf_counter()
    _core.equity_vs_random(hero, [], 7, n, 1)
    out["eight_way_preflop_samples_per_sec"] = n / (time.perf_counter() - t0)
    t0 = time.perf_counter()
    _core.enumerate_equity_hu(parse_cards("Ah Ad"), parse_cards("Kh Kd"), [])
    out["exact_hu_preflop_enumeration_ms"] = (time.perf_counter() - t0) * 1e3
    return out


def bench_icm() -> dict[str, float]:
    stacks = [3200.0, 2500.0, 2100.0, 1800.0, 1600.0, 1500.0, 1700.0, 1600.0]
    pay = [50.0, 30.0, 20.0]
    reps = 20_000
    t0 = time.perf_counter_ns()
    for _ in range(reps):
        _core.icm_equity(stacks, pay)
    ns8 = (time.perf_counter_ns() - t0) / reps
    t0 = time.perf_counter_ns()
    for _ in range(reps):
        _core.icm_equity(stacks[:4], pay)
    ns4 = (time.perf_counter_ns() - t0) / reps
    return {"icm_8_players_us": ns8 / 1e3, "icm_4_players_us": ns4 / 1e3}


def bench_solvers() -> dict[str, Any]:
    tb = tables()
    weights = _core.class_pair_weights()
    eq = tb.class_equity.astype(np.float64)
    t0 = time.perf_counter()
    sol = solve_headsup(eq, weights, 10.0, iterations=3000, tol=5e-5)
    dt = time.perf_counter() - t0
    t1 = time.perf_counter()
    river = solve_river(2.0, 10, 2000)
    dt_r = time.perf_counter() - t1
    return {
        "hu_pushfold_10bb_seconds": dt,
        "hu_pushfold_10bb_iterations": sol.iterations,
        "hu_pushfold_10bb_exploitability_bb_per_hand": sol.exploitability,
        "hu_pushfold_10bb_sb_jam_fraction": float(
            sum(sol.jam[i] * (6 if i // 13 == i % 13 else 4 if i // 13 > i % 13 else 12) for i in range(169)) / 1326
        ),
        "shipped_hu_max_exploitability_bb_per_hand": float(tb.meta["convergence"]["pushfold_hu_max_exploitability_bb"]),
        "river_2pot_seconds": dt_r,
        "river_2pot_iterations": river.iterations,
        "river_2pot_exploitability_pots": river.exploitability,
        "shipped_river_exploitability_pots": tb.meta["convergence"]["river_exploitability_pots"],
    }


def _lineup(seed: int) -> list[Bot]:
    return [make_archetype(n, seed + i) for i, n in enumerate(LINEUP)]


def bench_startup() -> dict[str, float]:
    """Fresh-process costs in ms: package import, table load, agent construction (ranges, calibration)."""
    code = (
        "import time; t0 = time.perf_counter(); import microholdem.agent as A; t1 = time.perf_counter(); "
        "A.tables(); t2 = time.perf_counter(); A.MicroAgent(seed=1); t3 = time.perf_counter(); "
        "print(t1 - t0, t2 - t1, t3 - t2)"
    )
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout.split()
    imp, load, build = (float(x) * 1e3 for x in out)
    return {"import_ms": imp, "tables_load_ms": load, "agent_construction_ms": build}


def bench_agent_latency(duel_hands: int, sng_tables: int, ring_hands: int, gc_mode: str) -> dict[str, Any]:
    """Per-action latency of the agent in heads-up duel, 8-max SNG and 8-max ring play."""
    if gc_mode == "frozen":
        agent = MicroAgent(seed=11, gc_freeze=True)
        gc.disable()
    else:
        agent = MicroAgent(seed=11)
    try:
        # First decisions of a freshly constructed agent. Construction already loaded the tables,
        # compiled the ranges and calibrated the equity call; bench_startup measures those costs.
        cold = MicroAgent(seed=12)
        res_cold = run_duel(cold, make_archetype("oracle", 5), 3, 99)
        first = list(res_cold.stats[0].latencies_ns)[:3]
        t0 = time.perf_counter()
        duel = run_duel(agent, make_archetype("oracle", 21), duel_hands, 31)
        duel_dt = time.perf_counter() - t0
        duel_lat = latency_summary(duel.stats, [0])
        agent_stats: list[BotStats] = []
        hands = 0
        t0 = time.perf_counter()
        for t in range(sng_tables):
            rng = random.Random(1000 + t)
            bots: list[Bot] = [agent, *_lineup(t * 10)]
            rng.shuffle(bots)
            res = run_sng(bots, seed=rng.getrandbits(32))
            hands += res.hands
            agent_stats.append(res.stats[bots.index(agent)])
        sng_dt = time.perf_counter() - t0
        sng_lat = latency_summary(agent_stats)
        # 8-max ring with stacks reset every hand against seven calling stations: multiway pots with up to
        # seven opponent ranges postflop, the most expensive equity path (the SNG lineup rarely gets there).
        t0 = time.perf_counter()
        ring_bots: list[Bot] = [agent, *[make_archetype("sponge", 70 + i) for i in range(7)]]
        _net, ring_stats = run_ring(ring_bots, ring_hands, 41)
        ring_dt = time.perf_counter() - t0
        ring_lat = latency_summary(ring_stats, [0])
        all_stats = [*agent_stats, duel.stats[0], ring_stats[0]]
        return {
            "gc_mode": gc_mode,
            "first_actions_after_construction_us": [x / 1e3 for x in first],
            "duel": {
                "opponent": "oracle",
                "hands": duel_hands,
                "bb_per_100_vs_oracle": duel.bb_per_100,
                "hands_per_sec": duel_hands / duel_dt,
                **duel_lat,
            },
            "sng": {
                "opponents": list(LINEUP),
                "tables": sng_tables,
                "hands": hands,
                "hands_per_sec": hands / sng_dt,
                **sng_lat,
            },
            "ring": {"opponents": ["sponge"] * 7, "hands": ring_hands, "hands_per_sec": ring_hands / ring_dt, **ring_lat},
            "timeouts": sum(s.timeouts for s in all_stats),
            "errors": sum(s.errors for s in all_stats),
            "illegal_actions": sum(s.illegal for s in all_stats),
            "calibration": describe(agent),
        }
    finally:
        gc.enable()


def bench_simulator(sng_tables: int = 60, duel_hands: int = 3000, ring_hands: int = 2000) -> dict[str, Any]:
    out: dict[str, Any] = {}
    hands = 0
    t0 = time.perf_counter()
    for t in range(sng_tables):
        bots: list[Bot] = [*_lineup(t), make_archetype("granite", 999 + t)]
        res = run_sng(bots, seed=t, cfg=SngConfig(record_latency=False))
        hands += res.hands
    dt = time.perf_counter() - t0
    out["sng_archetypes_only_hands_per_sec"] = hands / dt
    out["sng_archetypes_only_hands"] = hands
    t0 = time.perf_counter()
    res_d = run_duel(make_archetype("granite", 1), make_archetype("oracle", 2), duel_hands, 3, record_latency=False)
    out["duel_archetypes_hands_per_sec"] = res_d.hands / (time.perf_counter() - t0)
    t0 = time.perf_counter()
    bots8: list[Bot] = [*_lineup(7), make_archetype("oracle", 77)]
    run_ring(bots8, ring_hands, 5, record_latency=False)
    out["ring_8max_archetypes_hands_per_sec"] = ring_hands / (time.perf_counter() - t0)
    return out


SNG_RESULT_SAMPLES = 400


def bench_sng_results(tables_n: int = 100) -> dict[str, Any]:
    """Average points per table for the agent against the archetype lineup.

    The agent uses a fixed Monte Carlo sample count and the engine's 1 ms fold rule is switched off
    (latency is measured separately), so every decision depends only on the seeds and the table
    reproduces exactly; with the time budget the sample count follows the wall clock, and a scheduler
    stall can trigger a timeout fold, so outcomes would move by a few wins from run to run.
    """
    names = ["micro", *LINEUP]
    totals = {n: 0.0 for n in names}
    wins = {n: 0 for n in names}
    agent = MicroAgent(seed=5, samples=SNG_RESULT_SAMPLES)
    for t in range(tables_n):
        rng = random.Random(5000 + t)
        order = list(names)
        rng.shuffle(order)
        bots: list[Bot] = [agent if n == "micro" else make_archetype(n, rng.getrandbits(32)) for n in order]
        res = run_sng(bots, seed=rng.getrandbits(32), cfg=SngConfig(record_latency=False, time_limit_ns=10**12))
        for i, n in enumerate(order):
            totals[n] += res.points[i]
            wins[n] += res.finish[i] == 1
    rows: list[dict[str, Any]] = [{"bot": n, "avg_points": totals[n] / tables_n, "wins": wins[n]} for n in names]
    rows.sort(key=lambda r: -float(r["avg_points"]))
    return {"tables": tables_n, "samples": SNG_RESULT_SAMPLES, "points": list(SngConfig().points), "rows": rows}


def format_markdown(r: dict[str, Any]) -> str:
    L: list[str] = []
    env = r["environment"]
    L.append(f"Measured on {env['cpu']}, {env['platform']}, Python {env['python']}, microholdem {env['version']}.")
    L.append("")
    L.append("| Agent per-action latency | actions | p50 | p90 | p99 | p99.9 | max |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    labels = {
        "duel": "heads-up duel vs oracle",
        "sng": "8-max SNG vs the seven archetypes",
        "ring": "8-max ring (stacks reset) vs seven sponges",
    }
    for key in ("default", "frozen"):
        a = r["agent_latency"][key]
        for mode in ("duel", "sng", "ring"):
            d = a[mode]
            label = f"{labels[mode]}, GC {key}"
            L.append(
                f"| {label} | {int(d['actions'])} | {d['p50_us']:.1f} us | {d['p90_us']:.1f} us | {d['p99_us']:.1f} us | {d['p999_us']:.1f} us | {d['max_us']:.1f} us |"
            )
    a = r["agent_latency"]["default"]
    f = r["agent_latency"]["frozen"]
    st = r["startup"]
    total_actions = sum(int(r["agent_latency"][k][m]["actions"]) for k in ("default", "frozen") for m in ("duel", "sng", "ring"))
    L.append("")
    L.append(
        f"First three actions after construction: {', '.join(f'{x:.1f} us' for x in a['first_actions_after_construction_us'])}. "
        f"Fresh process: import {st['import_ms']:.0f} ms, table load {st['tables_load_ms']:.1f} ms, agent construction {st['agent_construction_ms']:.1f} ms. "
        f"Over all {total_actions} timed actions in the six rows: timeouts {a['timeouts'] + f['timeouts']}, errors {a['errors'] + f['errors']}, illegal actions {a['illegal_actions'] + f['illegal_actions']}."
    )
    L.append("")
    L.append("| Throughput | value |")
    L.append("|---|---:|")
    ev = r["evaluator"]
    L.append(f"| 7-card evaluations/s (C++ loop, includes dealing) | {ev['evals_per_sec_cpp_loop'] / 1e6:.1f} M |")
    L.append(f"| 7-card evaluations/s (numpy batch from Python) | {ev['evals_per_sec_numpy_batch'] / 1e6:.1f} M |")
    L.append(f"| single evaluate() call from Python | {ev['ns_per_python_call']:.0f} ns |")
    eq = r["equity"]
    L.append(f"| Monte Carlo samples/s, heads-up preflop | {eq['hu_preflop_samples_per_sec'] / 1e6:.1f} M |")
    L.append(
        f"| Monte Carlo samples/s, heads-up flop vs weighted range | {eq['hu_flop_vs_range_samples_per_sec'] / 1e6:.1f} M |"
    )
    L.append(f"| Monte Carlo samples/s, 8-way preflop | {eq['eight_way_preflop_samples_per_sec'] / 1e6:.2f} M |")
    L.append(
        f"| exact heads-up preflop enumeration (1,712,304 boards) | {eq['exact_hu_preflop_enumeration_ms']:.0f} ms |"
    )
    ic = r["icm"]
    L.append(f"| ICM, 8 players, 3 payouts | {ic['icm_8_players_us']:.1f} us |")
    sm = r["simulator"]
    L.append(f"| simulator, 8-max SNG, archetypes only | {sm['sng_archetypes_only_hands_per_sec']:.0f} hands/s |")
    L.append(f"| simulator, heads-up duel, archetypes only | {sm['duel_archetypes_hands_per_sec']:.0f} hands/s |")
    L.append(f"| simulator, 8-max ring, archetypes only | {sm['ring_8max_archetypes_hands_per_sec']:.0f} hands/s |")
    L.append(f"| simulator, 8-max SNG with the agent seated | {a['sng']['hands_per_sec']:.0f} hands/s |")
    L.append(f"| simulator, 8-max ring with the agent seated | {a['ring']['hands_per_sec']:.0f} hands/s |")
    so = r["solvers"]
    L.append("")
    L.append("| Solver | value |")
    L.append("|---|---:|")
    L.append(
        f"| heads-up push/fold CFR+, 10 BB: time to exploitability < 5e-5 BB/hand | {so['hu_pushfold_10bb_seconds']:.2f} s ({so['hu_pushfold_10bb_iterations']} iterations) |"
    )
    L.append(
        f"| heads-up push/fold, 10 BB: SB jam frequency | {100 * so['hu_pushfold_10bb_sb_jam_fraction']:.1f}% of hands |"
    )
    L.append(
        f"| shipped heads-up tables (1 to 25 BB): max exploitability | {so['shipped_hu_max_exploitability_bb_per_hand']:.1e} BB/hand |"
    )
    L.append(
        f"| river CFR+, 10 buckets, 2 pots deep: exploitability after {so['river_2pot_iterations']} iterations | {so['river_2pot_exploitability_pots']:.1e} pots ({so['river_2pot_seconds']:.1f} s) |"
    )
    sr = r["sng_results"]
    L.append("")
    L.append(
        f"| 8-max SNG, {sr['tables']} tables, points {sr['points'][:3]}, agent at a fixed {sr['samples']} samples | avg points | wins |"
    )
    L.append("|---|---:|---:|")
    for row in sr["rows"]:
        L.append(f"| {row['bot']} | {row['avg_points']:.2f} | {row['wins']} |")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="microholdem-bench")
    p.add_argument("--out", type=Path, default=Path("bench"))
    p.add_argument("--quick", action="store_true", help="smaller workloads (CI smoke run)")
    args = p.parse_args(argv)
    quick = args.quick
    results: dict[str, Any] = {
        "environment": {
            "cpu": _cpu_name(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "version": __version__,
            "numpy": np.__version__,
        }
    }
    print("evaluator...", file=sys.stderr)
    results["evaluator"] = bench_evaluator(2_000_000 if quick else 20_000_000)
    print("equity...", file=sys.stderr)
    results["equity"] = bench_equity()
    print("icm...", file=sys.stderr)
    results["icm"] = bench_icm()
    print("solvers...", file=sys.stderr)
    results["solvers"] = bench_solvers()
    print("startup...", file=sys.stderr)
    results["startup"] = bench_startup()
    print("agent latency...", file=sys.stderr)
    results["agent_latency"] = {
        "default": bench_agent_latency(500 if quick else 4000, 5 if quick else 60, 300 if quick else 1500, "default"),
        "frozen": bench_agent_latency(500 if quick else 4000, 5 if quick else 60, 300 if quick else 1500, "frozen"),
    }
    print("simulator...", file=sys.stderr)
    results["simulator"] = bench_simulator(10 if quick else 60, 500 if quick else 3000, 300 if quick else 2000)
    print("sng results...", file=sys.stderr)
    results["sng_results"] = bench_sng_results(10 if quick else 200)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    md = format_markdown(results)
    (args.out / "results.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
