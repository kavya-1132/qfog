"""Audit-driven adaptive triage (ADAT): a self-correcting threshold for the fog node.

Problem: the fog threshold is chosen once, on validation data. If the images change (new scanner, new hospital),
the fog node can start missing more abnormal cases and nobody notices.

Idea:
  1. A small random share (audit_rate) of the images the fog rule would KEEP is sent to the cloud anyway (the audit).
  2. For audited images a reference label is obtained (cloud verdict or radiologist). An audited abnormal image
     counts 1 / audit_rate, so the number of missed abnormal images is estimated without bias (inverse-probability weighting).
  3. Recall estimate over the last `window` images  =  caught / (caught + estimated missed).
  4. Every `every` images: estimate below target -> lower the threshold (send more); estimate above target + margin
     -> raise it (send fewer, save cloud calls). The threshold stays inside [t_min, t_max].

Limits (state them when presenting): audited labels are assumed correct; the estimate is noisy when few images
were audited; "recall" means abnormal images that reach the cloud, the cloud model makes its own errors.
"""
from collections import deque

import numpy as np


class AdaptiveTriage:
    def __init__(self, threshold, target_recall=0.90, audit_rate=0.10, step=0.02, margin=0.04, window=300,
                 warmup=200, every=25, t_min=0.02, t_max=0.80, seed=None):
        if not 0 < audit_rate <= 1:
            raise ValueError('audit_rate must be in (0, 1]')
        self.threshold = float(threshold)
        self.initial_threshold = float(threshold)
        self.target, self.q, self.step, self.margin = target_recall, audit_rate, step, margin
        self.window, self.warmup, self.every = window, warmup, every
        self.t_min, self.t_max = t_min, t_max
        self.rng = np.random.default_rng(seed)
        self.buf = deque(maxlen=window)          # per image: (caught abnormal, weighted missed abnormal)
        self.n_seen = 0
        self.n_audits = 0
        self.history = [self.threshold]          # threshold after every image

    # --- decision -------------------------------------------------------------------------------------------
    def decide(self, p_abnormal):
        """Returns (send_to_cloud, audited). `audited` is True when the rule would keep the image but it is sent for audit."""
        send = p_abnormal > self.threshold
        audited = (not send) and bool(self.rng.random() < self.q)
        if audited:
            self.n_audits += 1
        return send or audited, audited

    # --- learning -------------------------------------------------------------------------------------------
    def update(self, rule_sent, audited, abnormal=None):
        """Call once per image. `abnormal` = reference label (True/False), known only if the image reached the cloud;
        pass None when no reference is available (the image is then counted as neutral)."""
        caught = 1.0 if (rule_sent and abnormal) else 0.0
        missed = (1.0 / self.q) if (audited and abnormal) else 0.0
        self.buf.append((caught, missed))
        self.n_seen += 1
        if self.n_seen >= self.warmup and self.n_seen % self.every == 0:
            rec = self.estimated_recall()
            if rec is not None:
                if rec < self.target:
                    self.threshold = max(self.t_min, self.threshold - self.step)
                elif rec > self.target + self.margin:
                    self.threshold = min(self.t_max, self.threshold + self.step)
        self.history.append(self.threshold)

    def estimated_recall(self):
        a = sum(c for c, _ in self.buf)
        b = sum(m for _, m in self.buf)
        return a / (a + b) if (a + b) > 0 else None

    def state(self):
        return {'threshold': self.threshold, 'start threshold': self.initial_threshold, 'estimated recall': self.estimated_recall(),
                'target recall': self.target, 'audit rate': self.q, 'images seen': self.n_seen, 'audits': self.n_audits}


def simulate_stream(p, y, method='ADAT', threshold=0.4, target=0.90, audit_rate=0.10, shift_at=None, shift_scale=0.70, seed=0, **kw):
    """Replay a stream of triage scores p (array) with reference labels y (0/1). Used by the tests and the notebook.
    method: 'fixed' | 'fixed+audit' | 'ADAT'. Returns recall and cloud-calls-avoided on the second half of the stream."""
    rng = np.random.default_rng(seed)
    n = len(p)
    order = rng.permutation(n)
    p, y = np.asarray(p, float)[order].copy(), np.asarray(y)[order]
    if shift_at is not None:
        p[shift_at:] = np.clip(p[shift_at:] * shift_scale, 0, 1)
    ctl = AdaptiveTriage(threshold, target, audit_rate, seed=seed + 1, **kw)
    reached = np.zeros(n, bool)
    for i in range(n):
        if method == 'fixed':
            rule_sent, audited = p[i] > ctl.threshold, False
            reached[i] = rule_sent
            continue
        send, audited = ctl.decide(p[i])
        rule_sent = send and not audited
        reached[i] = send
        if method == 'ADAT':
            ctl.update(rule_sent, audited, bool(y[i]))
        else:                                   # audit only: record, never adapt
            ctl.buf.append(((1.0 if (rule_sent and y[i]) else 0.0), ((1.0 / ctl.q) if (audited and y[i]) else 0.0)))
    h = slice(n // 2, n)
    yb = y[h] == 1
    return {'recall': float((reached[h] & yb).sum() / max(yb.sum(), 1)), 'calls_avoided': float(100 * (1 - reached[h].mean())),
            'final_threshold': ctl.threshold}
