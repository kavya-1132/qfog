"""Tests for the audit-driven adaptive triage (no model files needed).  Run:  pytest -q tests/test_adaptive.py"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.adaptive import AdaptiveTriage, simulate_stream  # noqa: E402


def make_scores(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.46).astype(int)
    p = np.clip(rng.normal(0.30 + 0.25 * y, 0.2), 0, 1)
    return p, y


def test_images_above_threshold_are_never_audited():
    c = AdaptiveTriage(0.4, audit_rate=1.0, seed=0)
    send, audited = c.decide(0.9)
    assert send and not audited


def test_audit_rate_is_respected():
    c = AdaptiveTriage(0.9, audit_rate=0.2, seed=1)              # everything below 0.9 is "kept"
    flags = [c.decide(0.1)[1] for _ in range(4000)]
    assert abs(np.mean(flags) - 0.2) < 0.03


def test_estimate_recovers_true_recall():
    p, y = make_scores(6000)
    c = AdaptiveTriage(0.4, audit_rate=0.25, window=6000, warmup=10 ** 9, seed=2)       # never adapts, only estimates
    true_sent = ((p > 0.4) & (y == 1)).sum()
    for pi, yi in zip(p, y):
        send, audited = c.decide(pi)
        c.update(send and not audited, audited, bool(yi))
    true_recall = true_sent / y.sum()
    assert abs(c.estimated_recall() - true_recall) < 0.05


def test_threshold_moves_down_when_recall_is_too_low():
    p, y = make_scores(2000)
    c = AdaptiveTriage(0.7, target_recall=0.95, audit_rate=0.3, warmup=100, every=25, seed=3)   # a threshold that misses many
    for pi, yi in zip(p, y):
        send, audited = c.decide(pi)
        c.update(send and not audited, audited, bool(yi))
    assert c.threshold < 0.7


def test_threshold_stays_in_bounds():
    p, y = make_scores(3000)
    c = AdaptiveTriage(0.4, target_recall=0.99, audit_rate=0.3, step=0.1, t_min=0.05, t_max=0.6, warmup=50, every=10, seed=4)
    for pi, yi in zip(p, y):
        send, audited = c.decide(pi)
        c.update(send and not audited, audited, bool(yi))
    assert 0.05 <= min(c.history) and max(c.history) <= 0.6


def test_adat_holds_recall_under_shift_where_fixed_fails():
    p, y = make_scores(2000)
    fixed = np.mean([simulate_stream(p, y, 'fixed', shift_at=1000, seed=s)['recall'] for s in range(10)])
    adat = np.mean([simulate_stream(p, y, 'ADAT', shift_at=1000, seed=s)['recall'] for s in range(10)])
    assert adat > fixed + 0.15
    assert adat > 0.85


def test_invalid_audit_rate_rejected():
    with pytest.raises(ValueError):
        AdaptiveTriage(0.4, audit_rate=0)
