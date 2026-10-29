"""Card encoding, parsing and the 169 preflop hand classes.

A card is an ``int`` in ``0..51``: ``card = rank * 4 + suit`` with ranks
``2..A`` mapped to ``0..12`` and suits ``c, d, h, s`` mapped to ``0..3``.
The C++ core uses exactly the same encoding.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

RANKS = "23456789TJQKA"
SUITS = "cdhs"
N_CARDS = 52
N_CLASSES = 169


def make_card(rank: int, suit: int) -> int:
    """Build a card from a rank index (0..12) and a suit index (0..3)."""
    if not 0 <= rank < 13 or not 0 <= suit < 4:
        raise ValueError(f"bad rank/suit: {rank}/{suit}")
    return rank * 4 + suit


def rank_of(card: int) -> int:
    return card >> 2


def suit_of(card: int) -> int:
    return card & 3


def card_str(card: int) -> str:
    """``0`` -> ``"2c"``, ``51`` -> ``"As"``."""
    if not 0 <= card < N_CARDS:
        raise ValueError(f"card out of range: {card}")
    return RANKS[card >> 2] + SUITS[card & 3]


def cards_str(cards: Iterable[int]) -> str:
    return " ".join(card_str(c) for c in cards)


def parse_card(text: str) -> int:
    """``"As"`` -> ``51``. Accepts upper or lower case rank letters."""
    t = text.strip()
    if len(t) != 2:
        raise ValueError(f"bad card text: {text!r}")
    r = RANKS.find(t[0].upper())
    s = SUITS.find(t[1].lower())
    if r < 0 or s < 0:
        raise ValueError(f"bad card text: {text!r}")
    return r * 4 + s


def parse_cards(text: str) -> list[int]:
    """Parse ``"Ah Kd"`` or ``"AhKd"`` into a list of card ints."""
    t = text.replace(",", " ").strip()
    out: list[int] = []
    for tok in t.split():
        if len(tok) % 2:
            raise ValueError(f"bad card group: {tok!r}")
        for i in range(0, len(tok), 2):
            out.append(parse_card(tok[i : i + 2]))
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate card in {text!r}")
    return out


def full_deck() -> list[int]:
    return list(range(N_CARDS))


def shuffled_deck(rng: random.Random) -> list[int]:
    deck = full_deck()
    rng.shuffle(deck)
    return deck


# --------------------------------------------------------------------------
# 169 preflop classes. Index layout (13 x 13 grid, row-major):
#   pair     -> r * 13 + r
#   suited   -> hi * 13 + lo   (below the diagonal)
#   offsuit  -> lo * 13 + hi   (above the diagonal)
# --------------------------------------------------------------------------
def class_index(c1: int, c2: int) -> int:
    r1, r2 = c1 >> 2, c2 >> 2
    hi, lo = (r1, r2) if r1 >= r2 else (r2, r1)
    if hi == lo:
        return hi * 13 + hi
    if (c1 & 3) == (c2 & 3):
        return hi * 13 + lo
    return lo * 13 + hi


def class_name(idx: int) -> str:
    """``class_name(168)`` -> ``"AA"``, ``class_name(12*13+11)`` -> ``"AKs"``."""
    if not 0 <= idx < N_CLASSES:
        raise ValueError(f"class index out of range: {idx}")
    row, col = divmod(idx, 13)
    if row == col:
        return RANKS[row] * 2
    if row > col:
        return RANKS[row] + RANKS[col] + "s"
    return RANKS[col] + RANKS[row] + "o"


def parse_class(name: str) -> int:
    """``"AKs"`` -> index. Pairs are two letters, others need an ``s``/``o`` suffix."""
    n = name.strip()
    if len(n) == 2 and n[0].upper() == n[1].upper():
        r = RANKS.find(n[0].upper())
        if r < 0:
            raise ValueError(f"bad class: {name!r}")
        return r * 13 + r
    if len(n) != 3 or n[2].lower() not in "so":
        raise ValueError(f"bad class: {name!r}")
    a, b = RANKS.find(n[0].upper()), RANKS.find(n[1].upper())
    if a < 0 or b < 0 or a == b:
        raise ValueError(f"bad class: {name!r}")
    hi, lo = max(a, b), min(a, b)
    return hi * 13 + lo if n[2].lower() == "s" else lo * 13 + hi


def class_combos(idx: int) -> list[tuple[int, int]]:
    """All specific two-card combos (c1 < c2) belonging to a class."""
    row, col = divmod(idx, 13)
    out: list[tuple[int, int]] = []
    if row == col:
        for s1 in range(4):
            for s2 in range(s1 + 1, 4):
                out.append((make_card(row, s1), make_card(row, s2)))
    elif row > col:  # suited
        hi, lo = row, col
        for s in range(4):
            out.append((make_card(lo, s), make_card(hi, s)))
    else:  # offsuit
        lo, hi = row, col
        for s1 in range(4):
            for s2 in range(4):
                if s1 != s2:
                    a, b = make_card(lo, s1), make_card(hi, s2)
                    out.append((min(a, b), max(a, b)))
    return out


def class_combo_count(idx: int) -> int:
    row, col = divmod(idx, 13)
    if row == col:
        return 6
    return 4 if row > col else 12


def all_classes() -> list[int]:
    return list(range(N_CLASSES))


def hand_class(cards: Sequence[int]) -> int:
    if len(cards) != 2:
        raise ValueError("a hole hand has exactly two cards")
    return class_index(cards[0], cards[1])
