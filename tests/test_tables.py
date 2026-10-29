import numpy as np

from microholdem.cards import parse_class
from microholdem.tables import HU_STACKS, MAX_BEHIND, MULTI_STACKS, RIVER_STACKS, tables


def test_shipped_tables_load_and_have_expected_shapes() -> None:
    tb = tables()
    assert tb.class_equity.shape == (169, 169)
    assert tb.preflop_equity.shape == (9, 169)
    assert tb.hu_jam.shape == (len(HU_STACKS), 169)
    assert tb.hu_call.shape == (len(HU_STACKS), 169)
    assert tb.multi_jam.shape == (MAX_BEHIND + 1, len(MULTI_STACKS), 169)
    assert tb.multi_call.shape == (MAX_BEHIND + 1, len(MULTI_STACKS), MAX_BEHIND, 169)
    assert set(float(s) for s in tb.river_stacks) == set(RIVER_STACKS)
    assert tb.meta["convergence"]["pushfold_hu_max_exploitability_bb"] < 1e-3


def test_table_properties() -> None:
    tb = tables()
    aa, kk, s72 = parse_class("AA"), parse_class("KK"), parse_class("72o")
    assert np.all(tb.hu_jam[:, aa] > 0.99)
    assert np.all(tb.hu_call[:, aa] > 0.99)
    assert tb.pushfold_jam(15.0, s72, 2) < 0.01
    assert tb.pushfold_jam(1.0, s72, 2) > 0.5
    assert tb.pushfold_jam(2.0, parse_class("K2o"), 2) > 0.99
    assert tb.pushfold_jam(10.0, aa, 8) > 0.99
    assert tb.pushfold_jam(10.0, s72, 8) < 0.01
    # Equity vs more opponents is lower for every class, and AA stays on top.
    for k in range(1, 9):
        assert np.all(tb.preflop_equity[k] <= tb.preflop_equity[k - 1] + 1e-6)
        assert tb.preflop_equity[k].argmax() == aa
    assert tb.rank_hu[aa] < tb.rank_hu[kk] < tb.rank_hu[s72]
    assert tb.rank_multi[aa] < 0.01 and tb.rank_multi[s72] > 0.95


def test_class_range_helpers() -> None:
    tb = tables()
    aa, s72 = parse_class("AA"), parse_class("72o")
    everything = np.ones(169)
    assert tb.range_mass(aa, everything) == 1.0
    assert 0.80 < tb.equity_vs_class_range(aa, everything) < 0.90
    jam = tb.jam_range(10.0, 2)
    assert tb.equity_vs_class_range(aa, jam) > tb.equity_vs_class_range(s72, jam)
    found = tb.river_lookup(2.0, "x b1")
    assert found is not None
    acts, strat = found
    assert "f" in acts and strat.shape == (tb.river_buckets, len(acts))
