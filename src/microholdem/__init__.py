"""microholdem: sub-millisecond No-Limit Hold'em decision engine.

Components
----------
- :mod:`microholdem.evaluator`  C++ seven-card evaluator (bit-parallel rank counting)
- :mod:`microholdem.equity`     Monte Carlo and exact equity, weighted ranges
- :mod:`microholdem.icm`        Independent Chip Model (subset DP in C++)
- :mod:`microholdem.pushfold`   CFR+ solver for heads-up and multiway push/fold
- :mod:`microholdem.rivercfr`   vanilla CFR for a bucketed river betting game
- :mod:`microholdem.tables`     precomputed tables shipped with the package
- :mod:`microholdem.agent`      the sub-millisecond decision agent
- :mod:`microholdem.engine`     NLHE hand engine (side pots, blind levels, SNG rules)
- :mod:`microholdem.archetypes` opponent bots used by the simulator
- :mod:`microholdem.simulate`   tournament simulator (8-max SNG, heads-up duel)
- :mod:`microholdem.bench`      latency and throughput harness
"""

from __future__ import annotations

from .cards import card_str, cards_str, class_index, class_name, parse_cards
from .equity import Range, equity_vs_random, equity_vs_ranges, exact_equity_hu
from .evaluator import best_of, category, category_name, evaluate
from .icm import icm_equity

__all__ = [
    "Range",
    "best_of",
    "card_str",
    "cards_str",
    "category",
    "category_name",
    "class_index",
    "class_name",
    "equity_vs_random",
    "equity_vs_ranges",
    "evaluate",
    "exact_equity_hu",
    "icm_equity",
    "parse_cards",
]

__version__ = "0.1.0"
