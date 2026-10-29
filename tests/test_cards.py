import pytest

from microholdem.cards import (
    N_CLASSES,
    card_str,
    class_combo_count,
    class_combos,
    class_index,
    class_name,
    parse_card,
    parse_cards,
    parse_class,
)


def test_card_roundtrip() -> None:
    for c in range(52):
        assert parse_card(card_str(c)) == c
    assert parse_cards("AhKd 2c") == [parse_card("Ah"), parse_card("Kd"), parse_card("2c")]
    with pytest.raises(ValueError):
        parse_cards("Ah Ah")


def test_classes_cover_all_combos() -> None:
    counts = [0] * N_CLASSES
    for a in range(52):
        for b in range(a + 1, 52):
            counts[class_index(a, b)] += 1
    assert sum(counts) == 1326
    for idx in range(N_CLASSES):
        assert counts[idx] == class_combo_count(idx) == len(class_combos(idx))
        for a, b in class_combos(idx):
            assert class_index(a, b) == idx
            assert a < b


def test_class_names_roundtrip() -> None:
    names = {class_name(i) for i in range(N_CLASSES)}
    assert len(names) == N_CLASSES
    for i in range(N_CLASSES):
        assert parse_class(class_name(i)) == i
    assert class_name(parse_class("AKs")) == "AKs"
    assert class_name(parse_class("72o")) == "72o"
    assert class_name(parse_class("TT")) == "TT"
    assert class_index(parse_card("As"), parse_card("Ks")) == parse_class("AKs")
    assert class_index(parse_card("As"), parse_card("Kd")) == parse_class("AKo")
