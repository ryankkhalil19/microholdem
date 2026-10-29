import random
from itertools import combinations

import numpy as np
import pytest

from microholdem import _core
from microholdem.cards import parse_cards
from microholdem.evaluator import category, category_name, evaluate, evaluate_batch, reference_rank

# Known census of all C(52, 7) = 133,784,560 seven-card hands by best category.
CENSUS7 = [23_294_460, 58_627_800, 31_433_400, 6_461_620, 6_180_020, 4_047_644, 3_473_184, 224_848, 41_584]
# Known census of all C(52, 5) = 2,598,960 five-card hands.
CENSUS5 = [1_302_540, 1_098_240, 123_552, 54_912, 10_200, 5_108, 3_744, 624, 40]


def test_named_hands() -> None:
    cases = {
        "As Ks Qs Js Ts 2c 3d": "straight flush",
        "Ah Ad Ac As Kd 2c 3d": "four of a kind",
        "Ah Ad Ac Ks Kd 2c 3d": "full house",
        "Ah 9h 5h 3h 2h Kd Qc": "flush",
        "Ah 2d 3c 4s 5d 9c Kd": "straight",
        "Ah Ad Ac Ks Qd 2c 3d": "three of a kind",
        "Ah Ad Kc Ks Qd 2c 3d": "two pair",
        "Ah Ad Kc Qs Jd 2c 3d": "one pair",
        "Ah Kd Qc Js 9d 2c 3d": "high card",
    }
    for text, name in cases.items():
        assert category_name(evaluate(parse_cards(text))) == name, text


def test_wheel_is_lowest_straight() -> None:
    wheel = evaluate(parse_cards("Ah 2d 3c 4s 5d 9c Kd"))
    six_high = evaluate(parse_cards("2h 3d 4c 5s 6d 9c Kd"))
    broadway = evaluate(parse_cards("Th Jd Qc Ks Ad 2c 3d"))
    assert category(wheel) == category(six_high) == category(broadway) == 4
    assert wheel < six_high < broadway


def test_kickers_matter() -> None:
    assert evaluate(parse_cards("Ah Ad Kc Qs Jd 2c 3d")) > evaluate(parse_cards("Ah Ad Kc Qs Td 2c 3d"))
    assert evaluate(parse_cards("Ah Ad Kc Ks 9d 2c 3d")) > evaluate(parse_cards("Ah Ad Kc Ks 8d 2c 3d"))
    # Three pair: the best two pairs plus the best remaining card (which may be the third pair's rank).
    three_pair = evaluate(parse_cards("Ah Ad Kc Ks Qd Qc 2d"))
    two_pair_a_kicker = evaluate(parse_cards("Ah Ad Kc Ks Qd 3c 2d"))
    assert three_pair == two_pair_a_kicker


def test_five_and_six_card_inputs() -> None:
    five = parse_cards("Ah Kh Qh Jh Th")
    assert category_name(evaluate(five)) == "straight flush"
    assert evaluate(five) == evaluate([*five, *parse_cards("2c")])
    assert evaluate(five) == evaluate([*five, *parse_cards("2c 3d")])
    with pytest.raises(ValueError):
        evaluate(five[:4])
    with pytest.raises(ValueError):
        evaluate([*five, five[0], five[1]])


def test_matches_reference_ordering() -> None:
    rng = random.Random(7)
    hands = [rng.sample(range(52), rng.choice((5, 6, 7))) for _ in range(1500)]
    fast = [evaluate(h) for h in hands]
    slow = [reference_rank(h) for h in hands]
    for i in range(len(hands)):
        assert (fast[i] >> 26) == slow[i][0]
        for j in range(i + 1, min(i + 25, len(hands))):
            a = (fast[i] > fast[j]) - (fast[i] < fast[j])
            b = (slow[i] > slow[j]) - (slow[i] < slow[j])
            assert a == b, (hands[i], hands[j])


def test_batch_matches_scalar() -> None:
    rng = random.Random(11)
    hands = np.asarray([rng.sample(range(52), 7) for _ in range(2000)], dtype=np.int32)
    out = evaluate_batch(hands)
    assert out.dtype == np.uint32
    for row, v in zip(hands.tolist(), out.tolist(), strict=True):
        assert evaluate(row) == v


def test_five_card_census() -> None:
    counts = [0] * 9
    for combo in combinations(range(52), 5):
        counts[category(_core.evaluate(list(combo)))] += 1
    assert counts == CENSUS5


@pytest.mark.slow
def test_seven_card_census_exhaustive() -> None:
    assert list(_core.category_census7()) == CENSUS7


def test_positional_mask_and_batch_entry_points_validate_input() -> None:
    cards = parse_cards("Ah Kd Qc Js Td 2c 3d")
    assert _core.evaluate7(*cards) == evaluate(cards)
    assert _core.evaluate7(a=cards[0], b=cards[1], c=cards[2], d=cards[3], e=cards[4], f=cards[5], g=cards[6]) == (
        evaluate(cards)
    )
    assert _core.evaluate_mask(_core.card_mask(cards)) == evaluate(cards)
    assert _core.evaluate_mask(_core.card_mask(cards[:5])) == evaluate(cards[:5])
    with pytest.raises(ValueError):
        _core.evaluate7(cards[0], cards[0], *cards[2:])
    with pytest.raises(ValueError):
        _core.evaluate7(52, *cards[1:])
    with pytest.raises(ValueError):
        _core.evaluate_mask(_core.card_mask(cards[:4]))
    with pytest.raises(ValueError):
        _core.evaluate_mask(1 << 13)  # a bit outside the four 13-bit rank lanes
    for row in ([0, 0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 52], [0, 1, 2, 3, 4, 5, -1]):
        with pytest.raises(ValueError):
            evaluate_batch(np.asarray([row], dtype=np.int32))
