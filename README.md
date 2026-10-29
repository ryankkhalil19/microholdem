# microholdem

A sub-millisecond No-Limit Hold'em decision engine built for the format of the
"No Limit" pokerbot tournament (8-bot sit-and-gos, 1 ms per action, ICM
payouts, offline bots): a C++ hand evaluator and Monte Carlo equity sampler
behind a Python agent, CFR+ solvers for push/fold and a bucketed river game,
an ICM calculator, a rules-accurate tournament simulator with opponent
archetypes, and a benchmark harness that reports per-action latency
percentiles and hands per second.

Python 3.10+, numpy, pybind11 (build only). MIT license. CI runs ruff, mypy
(strict), pytest and a benchmark smoke run on Linux and macOS.

## Measured results

Apple M1, macOS, Python 3.10.13, single core. Produced by `python -m microholdem.bench --out bench`;
the raw numbers are in `bench/results.json` and `bench/results.md`.

Per-action latency of the agent, measured by the engine with `perf_counter_ns` around every `act()` call
(the engine builds the observation before the timer starts; everything the agent does, including its own
range inference and the C++ equity call, is inside the timed region):

| Agent per-action latency | actions | p50 | p90 | p99 | p99.9 | max |
|---|---:|---:|---:|---:|---:|---:|
| heads-up duel, GC default | 13,725 | 84.2 us | 93.0 us | 98.2 us | 109.9 us | 128.6 us |
| 8-max SNG, GC default | 8,654 | 7.1 us | 92.7 us | 101.3 us | 112.8 us | 121.5 us |
| heads-up duel, GC frozen | 13,711 | 84.0 us | 92.8 us | 97.8 us | 111.3 us | 121.9 us |
| 8-max SNG, GC frozen | 8,653 | 7.0 us | 92.5 us | 100.9 us | 111.6 us | 122.2 us |

Cold start (first three actions of a freshly constructed agent): 20.5 us, 87.0 us, 5.5 us.
Timeouts over the 22,379 timed actions: 0. Errors: 0. Illegal actions: 0.
The postflop budget is set by the agent (100 us for the Monte Carlo sample, at most 800 samples), so the
distribution is deliberately flat: the agent spends its budget, not more. Earlier runs with a 220 us budget
and the host's garbage collector active showed rare 0.5 to 0.9 ms stalls; `MicroAgent(gc_freeze=True)` plus
`gc.disable()` in the hosting process removes them.

| Throughput | value |
|---|---:|
| 7-card evaluations/s, C++ loop including dealing | 41.0 M |
| 7-card evaluations/s, numpy batch called from Python | 153.5 M |
| single `evaluate()` call from Python | 336 ns |
| Monte Carlo equity samples/s, heads-up preflop | 19.2 M |
| Monte Carlo equity samples/s, heads-up flop vs a weighted range | 5.7 M |
| Monte Carlo equity samples/s, 8-way preflop | 12.4 M |
| exact heads-up preflop enumeration (1,712,304 boards) | 21 ms |
| ICM, 8 players, 3 payouts (subset DP) | 1.7 us |
| simulator, 8-max SNG, archetypes only | 14,829 hands/s |
| simulator, heads-up duel, archetypes only | 14,341 hands/s |
| simulator, 8-max ring, archetypes only | 8,597 hands/s |
| simulator, 8-max SNG with the agent seated | 9,628 hands/s |

| Solver | value |
|---|---:|
| heads-up push/fold CFR+, 10 BB: time to exploitability below 5e-5 BB/hand | 0.05 s (150 iterations) |
| heads-up push/fold, 10 BB: small-blind jam frequency | 58.4% of hands |
| shipped heads-up tables, 1 to 25 BB in 0.5 BB steps: max exploitability | 4.9e-5 BB/hand |
| river CFR+, 10 buckets, 2 pots deep: exploitability after 2,000 iterations | 6.7e-6 pots (0.7 s) |
| multiway first-in jam model, 2 to 8 handed, 2 to 25 BB: max residual strategy change | 5.6e-3 |

Correctness: the C++ evaluator reproduces the known census of all 133,784,560 seven-card hands
(23,294,460 high card ... 41,584 straight flushes) in 0.5 s and agrees with a naive pure-Python
reference on every tested hand; the C++ ICM matches a recursive Malmuth-Harville reference to 1e-9.

8-max SNG results, 200 tables, points 50/30/20 for places 1 to 3, agent against the seven archetypes:

| bot | avg points | wins |
|---|---:|---:|
| micro (the agent) | 26.45 | 75 |
| passenger (checks or folds) | 24.51 | 0 |
| granite (tight passive) | 18.46 | 31 |
| jammer (Nash push/fold, fit-or-fold) | 13.70 | 41 |
| oracle (pure equity vs pot odds) | 7.35 | 25 |
| sponge (calling station) | 4.24 | 12 |
| blitz (loose aggressive) | 3.45 | 11 |
| scatter (uniform random) | 1.85 | 5 |

The archetypes are simple, legible policies for measurement, not strong opponents. A bot that never plays a hand
finishes second on average because the loose archetypes eliminate each other; that is a property of the field, and
it is why the agent's result is reported as points and table wins rather than as "beats everyone".

## Reproduce

```bash
git clone https://github.com/ryankkhalil19/microholdem && cd microholdem
python -m pip install -e ".[dev]"          # builds the C++ extension (needs a C++17 compiler)
pytest -q                                   # 46 tests, under 10 s, includes the exhaustive 7-card census
ruff check src tests && mypy               # lint and strict type check
python -m microholdem.bench --out bench    # full benchmark, about 6 minutes
python -m microholdem.simulate sng --tables 100 --lineup micro,granite,blitz,sponge,oracle,jammer,scatter,passenger
python -m microholdem.simulate duel --hands 5000 --a micro --b oracle
python -m microholdem.tables build         # regenerate the shipped tables (about 6 minutes, deterministic seed)
```

## How it works

### Evaluator (C++)

Cards are `rank * 4 + suit`; a hand is a 64-bit mask with a 16-bit lane per suit. Rank multiplicities are
computed bit-parallel from the four lanes (`pairs = OR of pairwise ANDs`, `trips`, `quads`), flushes from the
lane popcounts, and straights from an 8,192-entry table over the 13-bit rank mask. The result is a 32-bit
value, comparable across 5, 6 and 7 card hands, with the category in the top four bits. No large lookup
tables, so it is cache-resident and its latency does not depend on memory layout.

### Equity (C++)

xoshiro256** RNG, a reusable deck with partial Fisher-Yates draws, and weighted ranges compiled into
cumulative arrays with rejection of combos that collide with known cards. Exact enumeration is available for
heads-up (all board completions) and for any number of known hands with at most two cards to come.

### ICM (C++)

Malmuth-Harville by subset dynamic programming: `f[mask]` is the probability that exactly the players in
`mask` occupy the top places, extended one player at a time. `O(2^n * n)`, about 1.7 us for 8 players.
Used for the hand-cap payout in the simulator and for the agent's jam and call decisions.

### Solvers (Python, numpy)

- `pushfold.solve_headsup`: SB jam/fold vs BB call/fold over the 169 hand classes with exact joint deal
  weights, CFR+ (regret matching+, alternating updates, linear averaging), best-response exploitability.
- `pushfold.solve_multiway`: first-in jam vs up to seven independent callers with the standard no-overcall
  assumption; chip EV; ICM is applied at runtime by the agent.
- `rivercfr`: a discretized continuous-poker river (uniform strengths, K buckets, pot 1, stack S) with
  check / bet 0.5 pot / bet pot / raise / all-in, solved by vanilla CFR+ with all bucket pairs vectorized.
  Exploitability is computed by exact best response.

### Agent

Preflop at 20 BB or less: the push/fold tables; with a payout structure present the first-in jam is
re-evaluated in ICM equity using the solved calling ranges (no-overcall model, up to 16 ICM calls). Facing an
all-in: equity versus the solved jam range from the class matrix, chip EV or ICM EV. Deeper: position-indexed
opening, three-bet, call and four-bet thresholds over the class ranking. Postflop: Monte Carlo equity versus
per-opponent weighted ranges inferred from their preflop line, with the sample count chosen from the remaining
time budget using a calibration made at construction; heads-up river decisions come from the CFR strategy
(check-raise and bluff mixes by bucket), everything else from pot-odds thresholds with street margins and an
ICM risk premium on stack-committing calls.

### Simulator

`engine.play_hand` plays one hand for 2 to 10 seats: blinds and antes, full-raise and short all-in semantics
(a short all-in does not reopen the action for players who already acted), uncalled bet return, side pots,
showdown. Every `act()` call is timed; a bot that exceeds the limit (1 ms by default) is folded, or checked
when checking is free, and exceptions are handled the same way, mirroring the tournament rule. `simulate.run_sng`
follows the published format: eight bots, 2,000 chips, blinds from 10/20 rising every 25 hands, 200-hand cap,
unfilled places paid by ICM over the survivors' stacks. `run_duel` resets stacks every hand (chips per hand
measurement) and `run_ring` is 8-max with resets.

## Competition notes

From the official rules (pokerbot.sequoiacap.com/rules.html): the poker variant is revealed at the start of
the build phase (3.2); bots are Python, implement a provided Agent interface, run offline in a sandbox, and
must answer within one millisecond with a 100 MB file-size limit (3.3, 3.6, 3.7); bot state resets at the start
of each table and opponents are anonymized (3.8); each scored tournament is 10,000 eight-bot sit-and-gos with
2,000 starting chips, blinds from 10/20 rising every 25 hands, a 200-hand cap and ICM allocation at the cap
(4.3 to 4.5). The rules do not address code written before the event; they permit "publicly available
documentation" during the build (3.4) and require compliance with third-party licenses (8.4). This repository
is a portfolio project under the MIT license; whether any of it may be reused at the event is a question for
the organizers.

## Limitations

- The 169x169 class equity matrix is Monte Carlo (40,000 deals per pair, standard error about 0.25%), not exact.
- The abstractions ignore card removal between callers and blockers; the river model assumes independent
  uniform strengths.
- Opponent ranges are inferred from preflop lines with fixed fractions; there is no Bayesian updating from
  postflop actions beyond a tightening factor.
- The simulator is pure Python; the C++ core only covers evaluation, sampling and ICM.
- Benchmarks are single-machine numbers; the tournament hardware is unknown, so the agent keeps a wide margin
  below the limit rather than filling it.
