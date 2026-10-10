# QFog v2 — evaluation hardening (what changed, how to run, how to report)

Files: `QFog_Complete_v2.ipynb` (replaces `QFog_Complete (1).ipynb`), `QFog_App_patch/` (two files for the Streamlit app), this note.
Cell numbers 1–42 are unchanged; the new cells are **37a, 37b, 37c, 39a**, so the guide's references to Cell 39 and Cell 41 still hold.

## 1. How each review point was handled
| # | Review point | What v2 does |
|---|---|---|
| 1 | Fix the scientific framing | Part L conclusion rewritten: the quantum layer is an *experimental component*, the classical model is the stronger baseline, the quantum hypothesis is *not supported*. Viva Q7 no longer calls 8 qubits "optimal"; Q16 added ("Did the quantum model help? No."). |
| 2 | Evaluate the fog threshold properly | **Cell 37a**: thresholds chosen on *validation* for 85 / 90 / 95% recall (finer grid than Cell 25, whose 0.05-step grid may not reach a 95% target). Reports on *test*: recall with a 95% Wilson CI, specificity, precision, abnormal images kept at the fog (count and %), cloud-call rate, and which findings are kept. **Wording:** the fog now says *"LOW-RISK on the fog screen — not reviewed by the cloud model … not a diagnosis and not a clearance"* instead of *"NORMAL … no cloud call"* (Cell 31, Streamlit `pipeline.py`, app diagram, Part L, title cell). |
| 3 | Strengthen explanation reliability | **Cell 37c**: (a) frame-only, interior-only and full-image fog models (abnormal-vs-normal AUROC with bootstrap CI): do borders alone predict the label? (b) the cloud model evaluated on the original, *frame removed*, *frame swapped between images* and *frame only* test images, with paired-bootstrap CIs, mean probability change and top-finding change. The existing 2% plausibility result stays in the summary table (Cell 39) and in Part L. |
| 4 | Make system savings reproducible | **Cell 37b**: one fixed workload (same images as Cell 32), 3 passes, every image sent through the real fog node and cloud service; latency then composed per policy and link from measured components. Reports mean ± sd over passes and p50 / p95 / p99, with fog inference, network delay, upload, cloud inference and download separated. Network terms are labelled *emulated*. Outputs: `latency_trials.csv`, `latency_breakdown.csv`, `system_profile.csv`, figure `system_latency_trials.png`. |
| 5 | Simplify | No new architecture. Streamlit, 16-vs-8 study, classical control, outage fallback and XAI evaluation are all kept. |

New results are added to `summary.csv` by Cell 39a, saved by the existing Cell 40, and shipped in `qfog_deploy.zip` by Cell 41 (which now also stores `thresholds_by_target` in `deploy_config.json`).

## 2. How to run
1. Upload `QFog_Complete_v2.ipynb`, T4 GPU, **Run all** (`REUSE_SAVED = True` reloads the trained models).
2. Extra time on top of the ~15 min reload run (my estimate, not measured): Cell 37b ≈ 15–20 min (3 passes × 300 images with explanations), Cell 37c ≈ 5–10 min (two small MobileNet retrains + four passes of the cloud model over the test set), Cell 37a seconds.
3. Replace `QFog_App/src/pipeline.py` and `QFog_App/app/app.py` with the files in `QFog_App_patch/` (wording only), then unzip the new `qfog_deploy.zip` into `QFog_App/deploy/`.
4. Guide demo script: change the normal-example line to *"low-risk on the fog screen: no cloud call, not a clearance"*.

## 3. What was and was not tested
- **Tested here (stand-in data, no GPU):** syntax of every new cell; Cell 37a threshold selection, CIs and monotonicity (higher target → lower threshold, fewer misses, fewer calls avoided); Cell 37b latency composition against the node's own arithmetic, global state restored after profiling, table construction; Cell 39a; the border helper functions (frame kept / removed / swapped).
- **NOT tested:** anything that needs PyTorch, PennyLane or your trained models (this sandbox could not install them): Cell 37c's training and `cloud_probs`, and Cell 37b against the real fog node and cloud service. They reuse the notebook's own names (`TriageNet`, `fit`, `CXRDataset`, `backbone.tokens`, `cloud_model.pca`, `cloud_model.heads`, `fog_node`) with the same signatures as Cells 13, 24, 28, 31. If one errors on the real run, paste the traceback.

## 4. Reading the results (no numbers filled in on purpose)
- **Fog:** "At the 85 / 90 / 95% validation-selected operating points, test recall was **x / y / z** (95% CI …), so **n / m / k** abnormal X-rays were kept at the fog without cloud review, and cloud calls avoided were **a / b / c%**." If test recall falls below its target, say so; it is expected to drift from validation.
- **Latency:** quote mean ± sd over passes and p95, name the link, and say the network terms are emulated and fog and cloud share one machine.
- **Borders:** if the frame-only AUROC's CI is above 0.5, say border content carries label information. Use *frame swapped* as the main perturbation (realistic frames); *frame removed* creates a grey frame, which is itself a distribution shift.
- **Quantum:** unchanged from Cell 39. Currently: below the matched classical control, significant, and below the classical backbone.

## 5. One more thing to check before the viva
Cell 29 flags a "possible shortcut" when ≥ 30% of the heat is on the border band, but that band covers about **29% of the image area**, so uniform heat alone nearly meets the threshold. Cell 37 prints this baseline. Treat the border-heat statistic as weak evidence and lean on Cell 37c, which tests border dependence directly.
