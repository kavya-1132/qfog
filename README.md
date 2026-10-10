# QFog — Quantum-Enhanced Fog Learning for Early Pulmonary Disorder Detection

A hybrid quantum–classical, fog–cloud chest X-ray decision-support **prototype** with explainable AI (XAI).
Final-year B.Tech Data Science project, VIT Chennai.

> **Academic research prototype. Not a medical device, not clinically validated, not a diagnosis.**

## What it does

1. **Fog node (CPU, MobileNetV3-Small)** screens each X-ray. Cases it flags as possibly abnormal go to the cloud; the rest get a *low-risk screening note* (not a clearance).
2. **Cloud tier** (DenseNet121 → PCA → 8-qubit PennyLane circuit → attention fusion) predicts 14 findings.
3. **XAI**: Grad-CAM++ through the quantum circuit, a quantum-attention map, zone-based reasoning, and an occlusion faithfulness check.
4. **Resilience**: if the cloud is unreachable, flagged cases are kept at the fog and marked for review.

Dataset: NIH ChestX-ray14 sample — 5,606 images, 4,230 patients, 14 findings, patient-level 70/10/20 split
(train 3,931 / val 531 / test 1,144 images).

## Results (test set, all numbers produced by the code in `notebooks/`)

| Model | Mean AUROC (12 reliable classes) |
|---|---|
| DenseNet121 baseline | 0.735 |
| Classical head (pooled) | 0.728 ± 0.006 |
| Attention + classical control | 0.730 ± 0.001 |
| **Attention hybrid (8-qubit quantum)** | **0.721 ± 0.006** |
| Best seed-ensemble | 0.735 (baseline) |

**Honest reading of the quantum result:** the quantum layer did **not** improve accuracy. Against the matched classical control the difference is
−0.009 AUROC (95% CI −0.014 to −0.004), statistically significant *in the wrong direction*. The quantum component is an experimental
part of the architecture, not a source of better accuracy.

**8 vs 16 qubits:** 16 qubits was ~126× slower per training step (11.8 s vs 0.09 s) with no usable benefit, so 8 qubits were used.

**Fog–cloud system**

| Measure | Value |
|---|---|
| Fog triage recall / specificity (threshold 0.4) | 83.8% / 37.8% |
| Fog CPU speed-up vs cloud backbone | 11.0× |
| Cloud calls / upload / cloud compute saved (live run) | 29.3% / 30.2% / 29.6% |
| Projected cloud calls avoided at 20% abnormal prevalence | 33.5% |
| Cloud outage: flagged cases handled by fog fallback | 47 of 60 |

**XAI:** only 2% of analysed explanations were anatomically plausible; highlighted-pixel removal dropped the probability by 0.166 versus −0.078 for random pixels.

## Limitations (read these)

- **The fog screen misses about 16% of abnormal images** (recall 83.8%). This is the central limitation; a 95% recall target would need a lower threshold and sends more cases to the cloud.
- Fog and cloud run on **one machine**; network latency and bandwidth are **emulated**. Nothing is deployed to a real cloud.
- Accuracy is below what a clinical tool needs; AUROC ~0.73 on a small sample. Pneumonia and Hernia have too few test positives to report reliably.
- XAI heatmaps are mostly diffuse; do not read them as localisation evidence.

## Repository layout

```
├── app/              Streamlit website (8 pages)
├── src/              models, FogNode + CloudService pipeline, XAI, optional cloud_server
├── deploy/           trained weights, config, result CSVs, figures, example X-rays
│   ├── models/       cloud_model.pt (32 MB), triage.pt (4 MB)
│   ├── results/      every metric table as CSV + run_metadata.json
│   ├── figures/      EDA, ROC, qubit study, fog triage, system, XAI plots
│   └── examples/     sample chest X-rays for the demo
├── notebooks/        QFog_Complete_v2.ipynb  (full Colab pipeline, includes extra evaluation cells)
├── docs/             QFog_GUIDE.md, QFog_v2_CHANGES.md
├── tests/
├── requirements.txt
└── run_app.bat       Windows: double-click to start
```

## Run the app (Python 3.10–3.12)

```bash
pip install -r requirements.txt
python -m streamlit run app/app.py
```

On Windows, `run_app.bat` does the same. The trained weights are already in `deploy/`.

## Retrain / reproduce

Open `notebooks/QFog_Complete_v2.ipynb` in Google Colab (T4 GPU), mount Drive, and Run all (about 45–60 min).
The NIH ChestX-ray14 images are **not** included (size); download the sample from Kaggle ("NIH Chest X-rays sample") and place it as the notebook describes.
The v2 evaluation cells (operating points, latency breakdown, border-shortcut checks) were added after the main run and are described in `docs/QFog_v2_CHANGES.md`.

## Data and licence

Images: NIH ChestX-ray14 (NIH Clinical Center) — follow the dataset's terms. Code in this repository: academic use.
