Measured on Apple M1, macOS-27.0.1-arm64-arm-64bit, Python 3.10.13, microholdem 0.1.0.

| Agent per-action latency | actions | p50 | p90 | p99 | p99.9 | max |
|---|---:|---:|---:|---:|---:|---:|
| heads-up duel, GC default | 13725 | 84.2 us | 93.0 us | 98.2 us | 109.9 us | 128.6 us |
| 8-max SNG, GC default | 8654 | 7.1 us | 92.7 us | 101.3 us | 112.8 us | 121.5 us |
| heads-up duel, GC frozen | 13711 | 84.0 us | 92.8 us | 97.8 us | 111.3 us | 121.9 us |
| 8-max SNG, GC frozen | 8653 | 7.0 us | 92.5 us | 100.9 us | 111.6 us | 122.2 us |

Cold start (first three actions of a fresh agent): 20.5 us, 87.0 us, 5.5 us. Timeouts: 0, errors: 0, illegal actions: 0.

| Throughput | value |
|---|---:|
| 7-card evaluations/s (C++ loop, includes dealing) | 41.0 M |
| 7-card evaluations/s (numpy batch from Python) | 153.5 M |
| single evaluate() call from Python | 336 ns |
| Monte Carlo samples/s, heads-up preflop | 19.2 M |
| Monte Carlo samples/s, heads-up flop vs weighted range | 5.7 M |
| Monte Carlo samples/s, 8-way preflop | 12.38 M |
| exact heads-up preflop enumeration (1,712,304 boards) | 21 ms |
| ICM, 8 players, 3 payouts | 1.7 us |
| simulator, 8-max SNG, archetypes only | 14829 hands/s |
| simulator, heads-up duel, archetypes only | 14341 hands/s |
| simulator, 8-max ring, archetypes only | 8597 hands/s |
| simulator, 8-max SNG with the agent seated | 9628 hands/s |

| Solver | value |
|---|---:|
| heads-up push/fold CFR+, 10 BB: time to exploitability < 5e-5 BB/hand | 0.05 s (150 iterations) |
| heads-up push/fold, 10 BB: SB jam frequency | 58.4% of hands |
| shipped heads-up tables (1 to 25 BB): max exploitability | 4.9e-05 BB/hand |
| river CFR+, 10 buckets, 2 pots deep: exploitability after 2000 iterations | 6.7e-06 pots (0.7 s) |

| 8-max SNG, 200 tables, points [50.0, 30.0, 20.0] | avg points | wins |
|---|---:|---:|
| micro | 26.45 | 75 |
| passenger | 24.51 | 0 |
| granite | 18.46 | 31 |
| jammer | 13.70 | 41 |
| oracle | 7.35 | 25 |
| sponge | 4.24 | 12 |
| blitz | 3.45 | 11 |
| scatter | 1.85 | 5 |
