# ===== NEW CELL: Safety-aware fog triage (borderline band + random audit + online miss-rate estimate) =====
# Paste at the END of the notebook and run it AFTER the triage cells (tp, YA, THRESHOLD, OUT, FIG must exist).
# It uses only the saved triage probabilities, so it runs in seconds. No new training. No new test data.
import numpy as np, pandas as pd, matplotlib.pyplot as plt, os

rng0 = np.random.default_rng(2026)
pv, yv = np.asarray(tp['val']), np.asarray(YA['val']).astype(int)
pt, yt = np.asarray(tp['test']), np.asarray(YA['test']).astype(int)
N, NA = len(yt), int(yt.sum())

def pick_threshold(target):                       # chosen on VALIDATION only
    grid = np.round(np.arange(0.01, 0.80, 0.01), 2)
    ok = [t for t in grid if ((pv > t) & (yv == 1)).sum() / max((yv == 1).sum(), 1) >= target]
    return float(max(ok)) if ok else float(grid.min())

def stats(send, y):
    yb = y == 1
    return dict(recall=(send & yb).sum() / max(yb.sum(), 1), specificity=(~send & ~yb).sum() / max((~yb).sum(), 1), sent=send.mean())

def boot_recall(send, y, B=1000):                 # resamples test IMAGES (same as the report)
    r = np.random.default_rng(1); out = []
    for _ in range(B):
        i = r.integers(0, len(y), len(y)); yb = y[i] == 1
        out.append((send[i] & yb).sum() / max(yb.sum(), 1))
    return np.percentile(out, [2.5, 97.5])

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)

# ---- A. Higher recall targets (thresholds chosen on validation)
rows = []
for name, tgt in [('Current (85% target)', 0.85), ('Safer (90% target)', 0.90), ('Safest (95% target)', 0.95)]:
    t = THRESHOLD if tgt == 0.85 else pick_threshold(tgt)
    send = pt > t; s = stats(send, yt); lo, hi = boot_recall(send, yt)
    rows.append({'setting': name, 'threshold': t, 'recall': s['recall'], 'recall 95% CI': f'[{lo:.3f}, {hi:.3f}]', 'specificity': s['specificity'],
                 'sent to cloud %': 100 * s['sent'], 'cloud calls avoided %': 100 * (1 - s['sent']),
                 'avoided at 20% abnormal %': 100 * (1 - (0.2 * s['recall'] + 0.8 * (1 - s['specificity'])))})
A = pd.DataFrame(rows).set_index('setting')

# ---- B. Borderline band + random audit of what the fog keeps
t_high = THRESHOLD; t_low = pick_threshold(0.90)
def system(audit, reps=300):
    out = []
    for k in range(reps):
        r = np.random.default_rng(k)
        send = pt > t_low
        aud = (~send) & (r.random(N) < audit)     # random audit of images the fog would keep
        s = send | aud; st = stats(s, yt); out.append((st['recall'], st['sent']))
    a = np.array(out); return a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean()
rows = []
for audit in (0.0, 0.02, 0.05, 0.10):
    m, sd, sent = system(audit)
    rows.append({'audit rate of kept images': f'{int(audit * 100)}%', 'recall (mean)': m, 'recall (sd over 300 runs)': sd,
                 'sent to cloud %': 100 * sent, 'cloud calls avoided %': 100 * (1 - sent)})
B = pd.DataFrame(rows).set_index('audit rate of kept images')

# ---- C. Can the audit estimate the fog miss rate online? (share of abnormal among kept images)
kept = pt <= t_low; true_rate = (yt[kept] == 1).mean(); rows = []
for audit in (0.05, 0.10, 0.20):
    est, cover, widths = [], 0, []
    for k in range(500):
        r = np.random.default_rng(10_000 + k); idx = np.where(kept & (r.random(N) < audit))[0]
        n, a = len(idx), int((yt[idx] == 1).sum()); lo, hi = wilson(a, n)
        est.append(a / max(n, 1)); cover += lo <= true_rate <= hi; widths.append(hi - lo)
    rows.append({'audit rate': f'{int(audit * 100)}%', 'true share abnormal among kept': true_rate, 'mean estimate': np.mean(est),
                 'mean 95% interval width': np.mean(widths), 'interval covers truth %': 100 * cover / 500})
C = pd.DataFrame(rows).set_index('audit rate')

pd.set_option('display.width', 200); pd.set_option('display.float_format', lambda v: f'{v:.3f}')
print('A. Higher recall targets (test set, thresholds from validation)\n', A, '\n')
print(f'B. Borderline band: images with p in ({t_low:.2f}, {t_high:.2f}] are sent as "borderline"; random audit of the rest\n', B, '\n')
print('C. Online estimate of the fog miss rate from the audit sample\n', C, '\n')
os.makedirs(f'{OUT}/results', exist_ok=True)
A.to_csv(f'{OUT}/results/safety_recall_targets.csv'); B.to_csv(f'{OUT}/results/safety_band_audit.csv'); C.to_csv(f'{OUT}/results/safety_audit_estimate.csv')

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ts = np.round(np.arange(0.05, 0.80, 0.01), 2)
rc = [stats(pt > t, yt)['recall'] for t in ts]; av = [100 * (1 - (pt > t).mean()) for t in ts]
ax[0].plot(av, rc, '-', c='#555'); 
for nm, r in A.iterrows(): ax[0].scatter(r['cloud calls avoided %'], r['recall'], s=70, zorder=3, label=nm)
ax[0].set_xlabel('cloud calls avoided (%)'); ax[0].set_ylabel('recall of abnormal (test)'); ax[0].set_title('Safety vs saving (threshold sweep)'); ax[0].legend(fontsize=8)
ax[1].bar(B.index, B['recall (mean)'], color='#7B2CBF'); ax[1].set_ylim(0.8, 1.0); ax[1].set_xlabel('audit rate of images the fog keeps'); ax[1].set_ylabel('recall')
for i, (v, c) in enumerate(zip(B['recall (mean)'], B['cloud calls avoided %'])): ax[1].text(i, v + 0.004, f'{v:.3f}\n(avoid {c:.0f}%)', ha='center', fontsize=8)
ax[1].set_title(f'Borderline band + audit (t_low={t_low:.2f})'); plt.tight_layout(); plt.savefig(f'{FIG}/safety_aware_fog.png', dpi=150); plt.show()
print('NOTE: recall here means "abnormal images that reach the cloud". The cloud model itself also makes errors.')
