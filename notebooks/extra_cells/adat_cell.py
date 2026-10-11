# ===== NEW CELL: Audit-driven adaptive triage (ADAT): a self-correcting fog threshold =====
# Run AFTER the triage cells (needs tp, YA, THRESHOLD, OUT, FIG). Uses only saved triage probabilities; runs in about a minute.
import numpy as np, pandas as pd, matplotlib.pyplot as plt, os

p_all, y_all = np.asarray(tp['test']), np.asarray(YA['test']).astype(int)
N = len(y_all)

def run_stream(method, seed, shift, target=0.90, q=0.10, t0=None, step=0.02, margin=0.04, W=300, warm=200, every=25):
    """method: 'fixed' (threshold never changes, no audit) | 'fixed+audit' (audit only) | 'ADAT' (audit + adaptive threshold)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(N); p = p_all[order].copy(); y = y_all[order]
    if shift: p[N // 2:] = np.clip(p[N // 2:] * 0.70, 0, 1)        # scanner/population shift: scores shrink in the 2nd half
    t = THRESHOLD if t0 is None else t0
    reached = np.zeros(N, bool); tp_c = np.zeros(N); miss_c = np.zeros(N); ts = np.zeros(N)
    for i in range(N):
        send = p[i] > t
        audit = (not send) and method != 'fixed' and rng.random() < q
        reached[i] = send or audit
        tp_c[i] = float(send and y[i] == 1)                           # abnormal sent by the rule
        miss_c[i] = (1.0 / q) if (audit and y[i] == 1) else 0.0       # inverse-probability weight of audited abnormal kept images
        ts[i] = t
        if method == 'ADAT' and i >= warm and i % every == 0:
            lo = max(0, i - W); a, b = tp_c[lo:i + 1].sum(), miss_c[lo:i + 1].sum()
            rec_hat = a / (a + b) if (a + b) > 0 else 1.0
            if rec_hat < target: t = max(0.02, t - step)
            elif rec_hat > target + margin: t = min(0.80, t + step)
    h = slice(N // 2, N); yb = y[h] == 1                              # evaluate on the 2nd half (after adaptation / after the shift)
    return dict(recall=(reached[h] & yb).sum() / max(yb.sum(), 1), calls_avoided=100 * (1 - reached[h].mean()), final_t=ts[-1])

rows = []
for shift in (False, True):
    for m in ('fixed', 'fixed+audit', 'ADAT'):
        r = pd.DataFrame([run_stream(m, s, shift) for s in range(40)])
        rows.append({'scenario': 'with shift in 2nd half' if shift else 'no shift', 'method': m,
                     'recall (mean)': r.recall.mean(), 'recall (sd)': r.recall.std(),
                     'runs with recall >= 0.88 (%)': 100 * (r.recall >= 0.88).mean(),
                     'cloud calls avoided % (mean)': r.calls_avoided.mean(), 'final threshold (mean)': r.final_t.mean()})
res = pd.DataFrame(rows).set_index(['scenario', 'method'])
pd.set_option('display.width', 220); pd.set_option('display.float_format', lambda v: f'{v:.3f}')
print('Target recall = 0.90, audit rate = 10% of kept images, 40 random stream orders, evaluated on the 2nd half of each stream\n'); print(res)
os.makedirs(f'{OUT}/results', exist_ok=True); res.to_csv(f'{OUT}/results/adat_adaptive_triage.csv')

# one example run: how the threshold moves
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for k, shift in enumerate((False, True)):
    for m, c in (('fixed', '#888'), ('ADAT', '#7B2CBF')):
        rng = np.random.default_rng(0); order = rng.permutation(N)
        # replay ADAT threshold path
        rng = np.random.default_rng(0); order = rng.permutation(N); p = p_all[order].copy(); y = y_all[order]
        if shift: p[N // 2:] = np.clip(p[N // 2:] * 0.70, 0, 1)
        t = THRESHOLD; path = []; tp_c = np.zeros(N); miss_c = np.zeros(N)
        for i in range(N):
            send = p[i] > t; audit = (not send) and m == 'ADAT' and rng.random() < 0.10
            tp_c[i] = float(send and y[i] == 1); miss_c[i] = 10.0 if (audit and y[i] == 1) else 0.0; path.append(t)
            if m == 'ADAT' and i >= 200 and i % 25 == 0:
                lo = max(0, i - 300); a, b = tp_c[lo:i + 1].sum(), miss_c[lo:i + 1].sum(); rh = a / (a + b) if a + b else 1.0
                t = max(0.02, t - 0.02) if rh < 0.90 else (min(0.8, t + 0.02) if rh > 0.94 else t)
        ax[k].plot(path, c=c, label=m)
    ax[k].set_title('Threshold over the stream: ' + ('with shift at the midpoint' if shift else 'no shift')); ax[k].set_xlabel('images processed'); ax[k].set_ylabel('fog threshold'); ax[k].legend()
    if shift: ax[k].axvline(N // 2, c='r', ls='--', lw=0.8)
plt.tight_layout(); plt.savefig(f'{FIG}/adat_threshold_path.png', dpi=150); plt.show()
print('NOTE: audited images are assumed to get a correct abnormal/normal label from the cloud or a radiologist (an upper bound).')
