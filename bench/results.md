Measured on Apple M1, macOS-27.0.1-arm64-arm-64bit, Python 3.10.13, microholdem 0.1.0.

| Agent per-action latency | actions | p50 | p90 | p99 | p99.9 | max |
|---|---:|---:|---:|---:|---:|---:|
| heads-up duel vs oracle, GC default | 13709 | 100.1 us | 114.8 us | 127.6 us | 144.8 us | 350.2 us |
| 8-max SNG vs the seven archetypes, GC default | 9648 | 7.5 us | 111.2 us | 124.9 us | 137.2 us | 152.3 us |
| 8-max ring (stacks reset) vs seven sponges, GC default | 2983 | 9.7 us | 108.4 us | 118.9 us | 132.4 us | 139.5 us |
| heads-up duel vs oracle, GC frozen | 13716 | 99.8 us | 114.9 us | 126.7 us | 156.4 us | 392.1 us |
| 8-max SNG vs the seven archetypes, GC frozen | 9624 | 7.7 us | 111.4 us | 124.8 us | 139.1 us | 209.9 us |
| 8-max ring (stacks reset) vs seven sponges, GC frozen | 2985 | 14.6 us | 108.5 us | 122.1 us | 178.3 us | 246.5 us |

First three actions after construction: 17.7 us, 102.5 us, 6.1 us. Fresh process: import 64 ms, table load 9.3 ms, agent construction 13.6 ms. Over all 52665 timed actions in the six rows: timeouts 0, errors 0, illegal actions 0.

| Throughput | value |
|---|---:|
| 7-card evaluations/s (C++ loop, includes dealing) | 41.7 M |
| 7-card evaluations/s (numpy batch from Python) | 80.2 M |
| single evaluate() call from Python | 230 ns |
| Monte Carlo samples/s, heads-up preflop | 25.9 M |
| Monte Carlo samples/s, heads-up flop vs weighted range | 6.2 M |
| Monte Carlo samples/s, 8-way preflop | 12.39 M |
| exact heads-up preflop enumeration (1,712,304 boards) | 20 ms |
| ICM, 8 players, 3 payouts | 1.8 us |
| simulator, 8-max SNG, archetypes only | 13907 hands/s |
| simulator, heads-up duel, archetypes only | 13501 hands/s |
| simulator, 8-max ring, archetypes only | 7766 hands/s |
| simulator, 8-max SNG with the agent seated | 8592 hands/s |
| simulator, 8-max ring with the agent seated | 1793 hands/s |

| Solver | value |
|---|---:|
| heads-up push/fold CFR+, 10 BB: time to exploitability < 5e-5 BB/hand | 0.05 s (150 iterations) |
| heads-up push/fold, 10 BB: SB jam frequency | 58.4% of hands |
| shipped heads-up tables (1 to 25 BB): max exploitability | 4.9e-05 BB/hand |
| river CFR+, 10 buckets, 2 pots deep: exploitability after 2000 iterations | 6.7e-06 pots (0.7 s) |

| 8-max SNG, 200 tables, points [50.0, 30.0, 20.0], agent at a fixed 400 samples | avg points | wins |
|---|---:|---:|
| micro | 28.31 | 86 |
| passenger | 24.21 | 0 |
| granite | 18.02 | 35 |
| jammer | 13.67 | 37 |
| oracle | 7.60 | 23 |
| blitz | 3.45 | 10 |
| sponge | 3.19 | 8 |
| scatter | 1.55 | 1 |
