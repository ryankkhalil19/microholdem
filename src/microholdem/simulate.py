"""Tournament simulator: 8-max sit-and-go, heads-up duel and ring modes.

The sit-and-go follows the published competition format: eight bots, 2,000
starting chips, blinds from 10/20 rising every 25 hands, a 200-hand cap, and
unfilled places paid out by ICM over the survivors' stacks at the cap. The
duel mode mirrors the heads-up exhibit: stacks reset every hand, so the
result is a clean chips-per-hand measurement.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from .engine import Bot, BotStats, HandConfig, HandResult, next_live, play_hand
from .icm import icm_equity

DEFAULT_LEVELS: tuple[tuple[int, int, int], ...] = (
    (10, 20, 0),
    (15, 30, 0),
    (20, 40, 0),
    (30, 60, 0),
    (50, 100, 0),
    (75, 150, 0),
    (100, 200, 0),
    (150, 300, 0),
    (200, 400, 0),
    (300, 600, 0),
    (400, 800, 0),
    (500, 1000, 0),
)
DEFAULT_POINTS: tuple[float, ...] = (50.0, 30.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(slots=True)
class SngConfig:
    starting_stack: int = 2000
    hands_cap: int = 200
    level_hands: int = 25
    levels: tuple[tuple[int, int, int], ...] = DEFAULT_LEVELS
    points: tuple[float, ...] = DEFAULT_POINTS
    time_limit_ns: int = 1_000_000
    record_latency: bool = True
    record_hands: bool = False


@dataclass(slots=True)
class SngResult:
    finish: list[int]  # finishing place per seat (1 = winner)
    points: list[float]
    hands: int
    stats: list[BotStats]
    capped: bool
    history: list[HandResult] = field(default_factory=list)


def blinds_for_hand(cfg: SngConfig, hand_index: int) -> tuple[int, int, int]:
    level = min(hand_index // cfg.level_hands, len(cfg.levels) - 1)
    return cfg.levels[level]


def run_sng(bots: list[Bot], seed: int, cfg: SngConfig | None = None) -> SngResult:
    cfg = cfg or SngConfig()
    n = len(bots)
    if n < 2:
        raise ValueError("need at least two bots")
    rng = random.Random(seed)
    stacks = [cfg.starting_stack] * n
    stats = [BotStats() for _ in range(n)]
    for i, b in enumerate(bots):
        b.reset(i, n, rng.getrandbits(32))
    finish = [0] * n
    eliminated: list[int] = []
    button = rng.randrange(n)
    hands = 0
    history: list[HandResult] = []
    payouts = tuple(cfg.points)
    while hands < cfg.hands_cap and sum(1 for s in stacks if s > 0) > 1:
        sb, bb, ante = blinds_for_hand(cfg, hands)
        alive = [s > 0 for s in stacks]
        if not alive[button]:
            button = next_live(button, alive)
        live_payouts = payouts[: sum(alive)]
        hc = HandConfig(
            small_blind=sb,
            big_blind=bb,
            ante=ante,
            time_limit_ns=cfg.time_limit_ns,
            payouts=live_payouts,
            record_latency=cfg.record_latency,
        )
        result = play_hand(bots, stacks, button, hc, rng, hands, stats)
        hands += 1
        if cfg.record_hands:
            history.append(result)
        busted = [i for i in range(n) if result.start_stacks[i] > 0 and stacks[i] == 0]
        # Larger starting stack finishes higher when several bust on the same hand.
        busted.sort(key=lambda i: result.start_stacks[i])
        eliminated.extend(busted)
        button = next_live(button, [s > 0 for s in stacks])
    survivors = [i for i in range(n) if stacks[i] > 0]
    places_left = len(survivors)
    points = [0.0] * n
    # Eliminated players take places from the bottom up.
    for rank_from_bottom, seat in enumerate(eliminated):
        place = n - rank_from_bottom
        finish[seat] = place
        points[seat] = cfg.points[place - 1] if place - 1 < len(cfg.points) else 0.0
    capped = places_left > 1
    if capped:
        remaining = list(cfg.points[:places_left])
        share = icm_equity([stacks[i] for i in survivors], remaining)
        order = sorted(survivors, key=lambda i: -stacks[i])
        for place_idx, seat in enumerate(order):
            finish[seat] = place_idx + 1
        for seat, pts in zip(survivors, share, strict=True):
            points[seat] = pts
    else:
        finish[survivors[0]] = 1
        points[survivors[0]] = cfg.points[0]
    return SngResult(finish=finish, points=points, hands=hands, stats=stats, capped=capped, history=history)


@dataclass(slots=True)
class DuelResult:
    net_chips: int  # for bot A
    hands: int
    bb_per_100: float
    stats: list[BotStats]
    wins: tuple[int, int, int]  # A wins, B wins, splits


def run_duel(
    bot_a: Bot,
    bot_b: Bot,
    hands: int,
    seed: int,
    stack: int = 10_000,
    blinds: tuple[int, int] = (50, 100),
    time_limit_ns: int = 1_000_000,
    record_latency: bool = True,
) -> DuelResult:
    """Heads-up with stacks reset to ``stack`` every hand and alternating button."""
    rng = random.Random(seed)
    bots: list[Bot] = [bot_a, bot_b]
    stats = [BotStats(), BotStats()]
    for i, b in enumerate(bots):
        b.reset(i, 2, rng.getrandbits(32))
    hc = HandConfig(
        small_blind=blinds[0], big_blind=blinds[1], time_limit_ns=time_limit_ns, record_latency=record_latency
    )
    net = 0
    wa = wb = sp = 0
    for h in range(hands):
        stacks = [stack, stack]
        result = play_hand(bots, stacks, h % 2, hc, rng, h, stats)
        d = result.net[0]
        net += d
        if d > 0:
            wa += 1
        elif d < 0:
            wb += 1
        else:
            sp += 1
    bb100 = 100.0 * net / blinds[1] / hands if hands else 0.0
    return DuelResult(net_chips=net, hands=hands, bb_per_100=bb100, stats=stats, wins=(wa, wb, sp))


def run_ring(
    bots: list[Bot],
    hands: int,
    seed: int,
    stack: int = 10_000,
    blinds: tuple[int, int] = (50, 100),
    time_limit_ns: int = 1_000_000,
    record_latency: bool = True,
) -> tuple[list[int], list[BotStats]]:
    """N-max with stacks reset every hand. Returns net chips per seat and stats."""
    rng = random.Random(seed)
    n = len(bots)
    stats = [BotStats() for _ in range(n)]
    for i, b in enumerate(bots):
        b.reset(i, n, rng.getrandbits(32))
    hc = HandConfig(
        small_blind=blinds[0], big_blind=blinds[1], time_limit_ns=time_limit_ns, record_latency=record_latency
    )
    net = [0] * n
    for h in range(hands):
        stacks = [stack] * n
        result = play_hand(bots, stacks, h % n, hc, rng, h, stats)
        for i, d in enumerate(result.net):
            net[i] += d
    return net, stats


# --------------------------------------------------------------------------
# Lineups and CLI
# --------------------------------------------------------------------------
def make_bot(name: str, seed: int) -> Bot:
    if name in ("micro", "agent"):
        from .agent import MicroAgent

        return MicroAgent(seed=seed)
    from .archetypes import make_archetype

    return make_archetype(name, seed)


def latency_summary(stats: list[BotStats], seats: list[int] | None = None) -> dict[str, float]:
    lat: list[int] = []
    for i, st in enumerate(stats):
        if seats is None or i in seats:
            lat.extend(st.latencies_ns)
    if not lat:
        return {}
    lat.sort()
    n = len(lat)

    def pct(p: float) -> float:
        return lat[min(n - 1, int(p * n))] / 1000.0

    return {
        "actions": n,
        "p50_us": pct(0.50),
        "p90_us": pct(0.90),
        "p99_us": pct(0.99),
        "p999_us": pct(0.999),
        "max_us": lat[-1] / 1000.0,
        "mean_us": statistics.fmean(lat) / 1000.0,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="microholdem-sim", description="Run simulated tournaments.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sng", help="8-max sit-and-go tournaments")
    s.add_argument("--tables", type=int, default=100)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--lineup", default="micro,granite,blitz,sponge,oracle,jammer,scatter,passenger")
    s.add_argument("--json", action="store_true")
    d = sub.add_parser("duel", help="heads-up duel with stack resets")
    d.add_argument("--hands", type=int, default=5000)
    d.add_argument("--seed", type=int, default=1)
    d.add_argument("--a", default="micro")
    d.add_argument("--b", default="oracle")
    d.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    if args.cmd == "sng":
        names = args.lineup.split(",")
        totals = {nm: 0.0 for nm in names}
        wins = {nm: 0 for nm in names}
        t0 = time.perf_counter()
        hands = 0
        all_stats: dict[str, list[BotStats]] = {nm: [] for nm in names}
        for t in range(args.tables):
            rng = random.Random(args.seed * 1_000_003 + t)
            order = list(names)
            rng.shuffle(order)
            bots = [make_bot(nm, rng.getrandbits(32)) for nm in order]
            res = run_sng(bots, seed=rng.getrandbits(32))
            hands += res.hands
            for i, nm in enumerate(order):
                totals[nm] += res.points[i]
                wins[nm] += res.finish[i] == 1
                all_stats[nm].append(res.stats[i])
        dt = time.perf_counter() - t0
        rows: list[dict[str, Any]] = []
        for nm in names:
            lat = latency_summary(all_stats[nm])
            rows.append({"bot": nm, "avg_points": totals[nm] / args.tables, "wins": wins[nm], **lat})
        rows.sort(key=lambda r: -r["avg_points"])
        out = {"tables": args.tables, "hands": hands, "seconds": dt, "hands_per_sec": hands / dt, "rows": rows}
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            print(f"{args.tables} tables, {hands} hands in {dt:.1f}s ({hands / dt:.0f} hands/s)")
            print(
                f"{'bot':<10}{'avg pts':>9}{'wins':>6}{'actions':>9}{'p50us':>8}{'p99us':>8}{'p999us':>9}{'maxus':>9}"
            )
            for r in rows:
                print(
                    f"{r['bot']:<10}{r['avg_points']:>9.2f}{r['wins']:>6}{int(r.get('actions', 0)):>9}"
                    f"{r.get('p50_us', 0):>8.1f}{r.get('p99_us', 0):>8.1f}"
                    f"{r.get('p999_us', 0):>9.1f}{r.get('max_us', 0):>9.1f}"
                )
        return 0

    a = make_bot(args.a, args.seed)
    b = make_bot(args.b, args.seed + 1)
    t0 = time.perf_counter()
    duel = run_duel(a, b, args.hands, args.seed)
    dt = time.perf_counter() - t0
    out_d: dict[str, Any] = {
        "a": args.a,
        "b": args.b,
        "hands": duel.hands,
        "net_chips_a": duel.net_chips,
        "bb_per_100_a": duel.bb_per_100,
        "wins": duel.wins,
        "hands_per_sec": duel.hands / dt,
        "latency_a": latency_summary(duel.stats, [0]),
        "latency_b": latency_summary(duel.stats, [1]),
    }
    if args.json:
        print(json.dumps(out_d, indent=2))
    else:
        print(
            f"{args.a} vs {args.b}: {duel.hands} hands, {duel.bb_per_100:+.1f} BB/100 for {args.a}, "
            f"{duel.hands / dt:.0f} hands/s"
        )
        print("latency", args.a, out_d["latency_a"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
