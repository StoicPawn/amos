from __future__ import annotations

from traderx.labeling.triple_barrier import TripleBarrierResult, apply_triple_barrier


def test_triple_barrier_labels_hits():
    close = [100, 102, 101, 103, 104]
    vol = [0.01] * len(close)
    result = apply_triple_barrier(close, pt_mult=1.0, sl_mult=1.0, max_h=2, volatility=vol)
    assert all(isinstance(r, TripleBarrierResult) for r in result)
    assert {r.label for r in result}.issubset({-1, 0, 1})
    assert result[0].label == 1
    assert result[1].t_exit >= 1
