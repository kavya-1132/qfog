# QFog — Complete project: run guide, demo script and viva notes

**Files:** `QFog_Complete.ipynb` (the whole project: 42 numbered cells + Colab dashboard), `QFog_App.zip` (stand-alone Streamlit website) and this guide.

## 1. What the notebook contains
| Part | Cells | What it shows |
|---|---|---|
| A Setup | 1–4 | Drive, packages, Kaggle download, configuration |
| B Dataset & EDA | 5–9 | Audit table, class distribution, normal vs abnormal, co-occurrence heatmap, age/gender, one X-ray per class, image properties |
| C Preprocessing | 10–12 | Patient-level 70/10/20 split (no patient leakage), cache, augmentation preview |
| D Cloud baseline | 13–15 | DenseNet121 (CheXNet backbone) training curve, ROC curves |
| E Quantum + **16→8 qubits** | 16–19 | Circuit diagram; measured PCA information, output spread, gradient variance (barren plateau), cost; 8 vs 16 head-to-head |
| F Hybrid models | 20–23 | Quantum-only, concat, attention hybrid, **classical control**, 3 seeds, ensembles, bootstrap significance, per-class AUROC |
| G Fog triage | 24–27 | MobileNetV3-Small, validation-chosen threshold, ROC/sweep/confusion matrix, CPU speed-up, savings at screening prevalence |
| H Fog ↔ cloud integration | 28–32 | Real HTTP cloud service (FastAPI), fog node with emulated network, fallback on outage; fog+cloud vs cloud-only on good/average/poor links |
| I Explainable AI | 33–37 | Grad-CAM++ through the quantum circuit, quantum-attention map, **plain-language reasoning** (zones, plausibility, shortcut warning, occlusion faithfulness), qubit attribution |
| J Reports & results | 38–41 | Clinical reports through the running system, one results table, everything saved to Drive, **deploy bundle for the website** |
| K Live dashboard | 42 | Colab public link with 8 tabs incl. **Live X-ray diagnosis** |
| L Conclusion | — | Limitations + 14 viva Q&As |

## 2. Run it (first time ≈ 1.5–2 h)
1. Upload `QFog_Complete.ipynb` to `My Drive/qfog_project/` → right-click → Open with → **Google Colaboratory**.
2. Runtime → Change runtime type → **T4 GPU** → Save → Runtime → **Run all**.
3. Allow Drive access. Upload `kaggle.json` when asked (kaggle.com → Settings → API → Create New Token). Only asked once.
4. Keep the tab open and the laptop awake. Cell 41 saves `MyDrive/qfog_project/qfog_deploy.zip` (for the website); Cell 42 prints a Colab demo link `https://xxxx.gradio.live`.

Outputs: `My Drive/qfog_project/qfog_complete/` → `results/` (CSV/JSON tables), `figures/` (15 figures for the deck), `checkpoints/` (models).

## 3. The Streamlit website (like the CVD example) — recommended for the demo
1. After the notebook has run once, download **`MyDrive/qfog_project/qfog_deploy.zip`**.
2. Unzip **`QFog_App.zip`** on your laptop, then unzip `qfog_deploy.zip` **into `QFog_App/deploy/`** (so `deploy/deploy_config.json` exists).
3. In a terminal inside `QFog_App`: `python -m venv venv` → `venv\Scripts\activate` → `pip install -r requirements.txt` (once, needs internet).
4. Start it: `streamlit run app/app.py` (or double-click `run_app.bat`). It opens at http://localhost:8501 and runs fully offline — no Colab needed.
5. Optional, to show two real tiers: in a second terminal `python -m uvicorn src.cloud_server:app --port 8001`, then pick
   **HTTP (separate cloud process)** in the Live X-ray Diagnosis sidebar. Stop that terminal during the demo to show the fog fallback.

Sidebar sections: Dashboard · Dataset Explorer · Exploratory Data Analysis · Quantum Design (16 → 8 qubits) · Model Performance ·
Fog–Cloud System · Explainable AI · Live X-ray Diagnosis. The demo script below applies to both the website and the Colab dashboard.

## 3b. Colab dashboard instead (if the laptop setup fails)
Open the notebook → T4 GPU → **Run all**. `REUSE_SAVED = True` loads the saved models, so it is ready in ~15 minutes
(dataset download + feature extraction + system experiment). Open the `gradio.live` link. Keep the Colab tab open.

## 4. Demo script (≈ 6 minutes, follow the tabs)
| Tab | Show | Say |
|---|---|---|
| Dashboard | KPI cards, results table | "Two-tier system: a CPU fog node screens every X-ray; only flagged cases go to the quantum–classical cloud model, and every diagnosis is explained." |
| Dataset explorer | audit, split, example X-rays | "NIH ChestX-ray14 sample: 5,606 images, 14 findings, split by patient so no patient is in both training and test." |
| EDA | class distribution, co-occurrence | "Classes are imbalanced and often co-occur, so it is multi-label; rare classes are flagged as statistically unreliable." |
| Quantum design | four plots + tables | "We tested 4 to 16 qubits. With more qubits the outputs bunch together across patients, gradients vanish — a barren plateau — and simulation cost doubles per qubit. 8 is the largest circuit that stays trainable." |
| Model performance | comparison + significance | "Every quantum model is compared against an identical classical control, over three seeds, with bootstrap confidence intervals." (read the result as printed) |
| Fog–cloud system | triage plots, system table | "The fog threshold is fixed on validation for ≥85% recall. In the running system the fog avoids X% of cloud calls and keeps working when the cloud is down." |
| Explainable AI | Grad-CAM grid, verdicts, faithfulness | "Heatmaps are computed through the quantum circuit; we check anatomical plausibility, flag shortcut learning, and verify faithfulness by occlusion." |
| **Live X-ray diagnosis** | upload/example → Run | Normal example: "handled at the fog in milliseconds, no cloud call." Abnormal example: "sent to the cloud — findings, heatmap, quantum attention and a written reasoning report." Then tick **Simulate cloud outage**: "the fog falls back and flags the case for review." Move the latency slider to show the network effect. |

## 5. What changed from your original notebook, and why
| Original | Now | Why it matters |
|---|---|---|
| Random image split | Patient-level split | Same patient in train and test inflates results |
| `best_state = model.state_dict()` | `copy.deepcopy(...)` | The old "best" model was actually the last epoch |
| Attention over 1 token (softmax = 1) | Attention over 49 image regions | Old attention ignored the quantum branch completely |
| Hard-coded AUROC tables | Everything computed and saved | Results are reproducible |
| No classical control | Matched classical control + 3 seeds + bootstrap | Only fair way to attribute effects to the quantum circuit |
| ResNet18 | DenseNet121 (CheXNet) | Stronger, standard backbone for chest X-rays |
| Grad-CAM on the baseline | Grad-CAM++ through the full hybrid + reasoning text | Explains the model that actually diagnoses |
| Fog simulated by timing two models | Real HTTP fog→cloud services, network emulation, outage fallback | Actual integration of the two tiers |
| "16 qubits gave near-zero outputs" | Measured study: information, spread, gradient variance, cost, head-to-head | Evidence for the 8-qubit decision |

## 6. Presenting the results honestly
Read every number from **Cell 39** and the dashboard. Phrase the quantum result exactly as the significance test says:
- CI excludes 0 and positive → "the quantum attention hybrid improves mean AUROC by +x over the classical control (95% CI [a, b])."
- CI includes 0 → "the quantum layer matches a classical layer of the same size; we found no significant accuracy gain on this dataset."
Strong, defensible claims regardless of that outcome: working two-tier system, measured cloud savings, fault tolerance, 16→8 qubit evidence,
explanations through the quantum circuit with plausibility and faithfulness checks. Do not reuse numbers from the old notebook.

## 7. Troubleshooting
| Problem | Fix |
|---|---|
| Drive mount / `credential propagation` error | Runtime → Disconnect and delete runtime → Run all, same Google account |
| `device: cpu` | Runtime → Change runtime type → T4 GPU |
| Disconnected during training | Run all again — finished models reload; only the rest retrains |
| Out of memory in Cell 18 (16 qubits) | Runtime → Restart, then Run all (models reload) |
| Colab demo link not working | Re-run Cell 42; keep the Colab tab open |
| Retrain from scratch | Set `REUSE_SAVED = False` in Cell 4 |
