# microholdem

A sub-millisecond No-Limit Hold'em decision engine built for the format of the
"No Limit" pokerbot tournament (8-bot sit-and-gos, 1 ms per action, ICM
payouts, offline bots): a C++ hand evaluator and Monte Carlo equity sampler
behind a Python agent, CFR+ solvers for push/fold and a bucketed river game,
an ICM calculator, a tournament simulator with opponent archetypes, and a
benchmark harness that reports per-action latency percentiles and hands per
second.

Python 3.10+, numpy, pybind11 (build only). MIT license. CI runs ruff, mypy
(strict, package and tests), pytest and a benchmark smoke run on Linux and macOS.

## Measured results

Apple M1, macOS, Python 3.10.13, numpy 2.2.6, single core. Produced by `python -m microholdem.bench --out bench`
(about 15 seconds); the raw numbers are in `bench/results.json` and `bench/results.md`.

Per-action latency of the agent, measured by the engine with `perf_counter_ns` around every `act()` call
(the engine builds the observation before the timer starts; everything the agent does, including its own
range inference and the C++ equity call, is inside the timed region). Each row names its opponents, because
the median is set by the share of decisions that reach the postflop Monte Carlo step: about 70% against the
equity-driven oracle in the duel, under half in the sit-and-go. The tail (p90 and above) is set by the agent's
own budget.

| Agent per-action latency | actions | p50 | p90 | p99 | p99.9 | max |
|---|---:|---:|---:|---:|---:|---:|
| heads-up duel vs oracle (4,000 hands, 100 BB), GC default | 13,709 | 100.1 us | 114.8 us | 127.6 us | 144.8 us | 350.2 us |
| 8-max SNG vs the seven archetypes (60 tables), GC default | 9,648 | 7.5 us | 111.2 us | 124.9 us | 137.2 us | 152.3 us |
| 8-max ring vs seven calling stations (1,500 hands), GC default | 2,983 | 9.7 us | 108.4 us | 118.9 us | 132.4 us | 139.5 us |
| heads-up duel vs oracle, GC frozen | 13,716 | 99.8 us | 114.9 us | 126.7 us | 156.4 us | 392.1 us |
| 8-max SNG vs the seven archetypes, GC frozen | 9,624 | 7.7 us | 111.4 us | 124.8 us | 139.1 us | 209.9 us |
| 8-max ring vs seven calling stations, GC frozen | 2,985 | 14.6 us | 108.5 us | 122.1 us | 178.3 us | 246.5 us |

Timeouts over all 52,665 timed actions in the six rows: 0. Errors: 0. Illegal actions: 0.
The ring row exists to exercise the most expensive path: postflop pots with up to seven opponent ranges.

The agent targets 100 us for the Monte Carlo sample (counted from the start of `act()`, at most 800 samples).
It sizes the sample from a per-opponent-count cost table measured at construction (flop board, one and seven
ranges) and corrected after every call from the measured time per sample, so heads-up spots use the budget
and seven-opponent spots no longer overrun it. First three actions after construction: 17.7 us, 102.5 us,
6.1 us. Fresh-process costs, outside any decision: importing the package 64 ms, loading the tables 9.3 ms,
constructing the agent 13.6 ms (range compilation and calibration).

| Throughput | value |
|---|---:|
| 7-card evaluations/s, C++ loop including dealing | 41.7 M |
| 7-card evaluations/s, numpy batch of 50,000 distinct random hands, validated | 80.2 M |
| single `evaluate()` call from Python | 230 ns |
| Monte Carlo equity samples/s, heads-up preflop vs a random hand | 25.9 M |
| Monte Carlo equity samples/s, heads-up flop vs a weighted range | 6.2 M |
| Monte Carlo equity samples/s, 8-way preflop | 12.4 M |
| exact heads-up preflop enumeration (1,712,304 boards) | 20 ms |
| ICM, 8 players, 3 payouts (subset DP) | 1.8 us |
| simulator, 8-max SNG, archetypes only | 13,907 hands/s |
| simulator, heads-up duel, archetypes only | 13,501 hands/s |
| simulator, 8-max ring, archetypes only | 7,766 hands/s |
| simulator, 8-max SNG with the agent seated | 8,592 hands/s |
| simulator, 8-max ring with the agent seated (seven calling stations) | 1,793 hands/s |

| Solver | value |
|---|---:|
| heads-up push/fold CFR+, 10 BB: time to exploitability below 5e-5 BB/hand | 0.05 s (150 iterations) |
| heads-up push/fold, 10 BB: small-blind jam frequency | 58.4% of hands |
| shipped heads-up tables, 1 to 25 BB in 0.5 BB steps: max exploitability | 4.9e-5 BB/hand |
| river CFR+, 10 buckets, 2 pots deep: exploitability after 2,000 iterations | 6.7e-6 pots (0.7 s) |
| shipped river tables, 0.5 to 4 pots deep: max exploitability | 1.1e-5 pots |
| multiway first-in jam model, 2 to 8 handed, 2 to 25 BB: max best-response gain of any player under the model | 1.9e-5 BB/hand |

The last two rows come from `src/microholdem/data/meta.json`, written by the table build.

Correctness: the C++ evaluator reproduces the known census of all 133,784,560 seven-card hands
(23,294,460 high card ... 41,584 straight flushes) in 0.5 s and agrees with a naive pure-Python reference on
every tested hand; the C++ ICM matches a recursive Malmuth-Harville reference to 1e-9; multi-opponent equity
matches exact weighted enumeration in either opponent order.

8-max SNG results, 200 tables, points 50/30/20 for places 1 to 3, agent against the seven archetypes. For this
table only, the agent uses a fixed 400 Monte Carlo samples and the engine's 1 ms rule is switched off, so the
result depends only on the seeds and the README command reproduces it exactly (with the time budget the sample
count follows the wall clock and the outcome moves by a few wins between runs):

| bot | avg points | wins |
|---|---:|---:|
| micro (the agent) | 28.31 | 86 |
| passenger (checks or folds) | 24.21 | 0 |
| granite (tight passive) | 18.02 | 35 |
| jammer (Nash push/fold, fit-or-fold) | 13.67 | 37 |
| oracle (pure equity vs pot odds) | 7.60 | 23 |
| blitz (loose aggressive) | 3.45 | 10 |
| sponge (calling station) | 3.19 | 8 |
| scatter (uniform random) | 1.55 | 1 |

The archetypes are simple, legible policies for measurement, not strong opponents. A bot that never plays a hand
finishes second on average because the loose archetypes eliminate each other; that is a property of the field, and
it is why the agent's result is reported as points and table wins rather than as "beats everyone".

## Reproduce

```bash
git clone https://github.com/ryankkhalil19/microholdem && cd microholdem
python -m pip install -e ".[dev]"          # builds the C++ extension (needs a C++17 compiler)
pytest -q                                   # 66 tests, about 8 s, includes the exhaustive 7-card census
ruff check src tests && mypy               # lint and strict type check (package and tests)
python -m microholdem.bench --out bench    # full benchmark, about 15 s
python -m microholdem.simulate sng --tables 100 --lineup micro,granite,blitz,sponge,oracle,jammer,scatter,passenger
python -m microholdem.simulate duel --hands 5000 --a micro --b oracle
python -m microholdem.tables build         # regenerate the shipped tables (about 5 minutes, deterministic seed)
```

## How it works

### Evaluator (C++)

Cards are `rank * 4 + suit`; a hand is a 64-bit mask with a 16-bit lane per suit. Rank multiplicities are
computed bit-parallel from the four lanes (`pairs = OR of pairwise ANDs`, `trips`, `quads`), flushes from the
lane popcounts, and straights and kickers from five 8,192-entry tables indexed by the 13-bit rank mask
(72 KiB in total, small enough to stay in L1/L2 cache; no multi-megabyte perfect-hash table). The result is a
32-bit value, comparable across 5, 6 and 7 card hands, with the category in the top four bits. The list,
positional, mask and numpy batch entry points all reject duplicate and out-of-range cards.

### Equity (C++)

xoshiro256** RNG, a reusable deck with partial Fisher-Yates draws, and weighted ranges compiled into
cumulative arrays. With several opponents, each opponent's combo is drawn by rejection against the cards
already out, and every deal is weighted by the product of the range mass still available to each opponent when
it was drawn (sequential importance sampling, using per-card and per-pair range masses). The estimate therefore
converges to equity over deals weighted by the product of the range weights, independent of the order the
opponents are listed in, and narrow overlapping ranges never stall a draw. Exact enumeration is available for
heads-up (all board completions) and for any number of known hands with at most two cards to come.

### ICM (C++)

Malmuth-Harville by subset dynamic programming: `f[mask]` is the probability that exactly the players in
`mask` occupy the top places, extended one player at a time; the chips left after `mask` are read from the
complement's subset sum, so lopsided or fractional stacks keep full precision. `O(2^n * n)`, about 1.8 us for
8 players. A zero stack gets nothing and places that cannot be filled are not paid; callers that model a bust
add the finishing-place payout themselves (the agent does). Used for the hand-cap payout in the simulator and
for the agent's jam and call decisions.

### Solvers (Python, numpy)

- `pushfold.solve_headsup`: SB jam/fold vs BB call/fold over the 169 hand classes with exact joint deal
  weights, CFR+ (regret matching+, alternating updates, linear averaging), best-response exploitability.
- `pushfold.solve_multiway`: first-in jam vs up to seven independent callers with the standard no-overcall
  assumption; each caller conditions on the hand reaching it (the jam and every earlier fold); chip EV; ICM is
  applied at runtime by the agent. Convergence is reported as the largest best-response gain of any player
  under the model, since the multiway game is general-sum.
- `rivercfr`: a discretized continuous-poker river (uniform strengths, K buckets, pot 1, stack S) with
  check / bet 0.5 pot / bet pot / one raise to 3x / all-in when a bet or raise is capped by the stack, solved by
  vanilla CFR+ with all bucket pairs vectorized. The deepest line commits 3 pots, so every stack above 3 pots
  is the same game; the shipped tables cover 0.5, 1, 2 and 4 pots. Exploitability is computed by exact best
  response.

### Agent

Preflop at 20 BB or less: the push/fold tables; with a payout structure present the first-in jam is
re-evaluated in ICM equity using the solved calling ranges (no-overcall model, up to 16 ICM calls). Facing an
all-in: equity versus the solved jam range from the class matrix, chip EV or ICM EV. Every stack vector handed
to the ICM conserves the chips in play, and a covered call or jam that loses is valued at the payout of the
place the agent finishes in. Deeper: position-indexed opening, three-bet, call and four-bet thresholds over the
class ranking. Postflop: Monte Carlo equity versus per-opponent weighted ranges inferred from their preflop
line, with the sample count chosen from the remaining time budget; heads-up river decisions come from the CFR
strategy (check-raise and bluff mixes by bucket), everything else from pot-odds thresholds with street margins
and an ICM risk premium on stack-committing calls.

### Simulator

`engine.play_hand` plays one hand for 2 to 10 seats: blinds and antes, full-raise and short all-in semantics
(an all-in below a full raise does not reopen the action for players who already acted, unless several short
all-ins add up to a full raise, TDA Rule 49-A), a short big blind that does not lower the amount to call, uncalled
bet return, side pots, showdown. No player is asked to act once everyone else is all-in and the bet is covered.
Every `act()` call is timed; a bot that exceeds the limit (1 ms by default) is folded, or checked when checking
is free, and exceptions or malformed return values are handled the same way, mirroring the tournament rule.
`simulate.run_sng` follows the published format: eight bots, 2,000 chips, blinds from 10/20 rising every 25
hands, 200-hand cap, unfilled places paid by ICM over the survivors' stacks, and a dead button (the big blind
moves one live seat per hand). `run_duel` resets stacks every hand (chips per hand measurement) and `run_ring`
is 8-max with resets.

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
- The latency numbers need a quiet host. The published run was taken at a load average of about 2.3; a run
  taken while other processes held the load average near 18 recorded 7 timeouts in 52,866 actions, all from
  scheduler stalls of up to 5.9 ms rather than from the agent's own work.
