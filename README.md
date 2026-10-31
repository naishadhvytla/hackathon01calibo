# FraudGuard — Real-Time Payment Fraud Decision System

**KLU + Path-3 · ML-02 Hackathon · Real-Time UPI / Digital Payment Fraud Detection**

FraudGuard turns a fraud model into a business decision. Every payment gets a fraud score, the score becomes **APPROVE / STEP-UP OTP / BLOCK**, and both cut-offs are set by **expected rupee cost**, not accuracy. A label-free anomaly layer flags unusual behaviour that no labelled fraud has shown yet. The same pipeline runs on two datasets: the ULB credit-card benchmark (primary) and the organiser's UPI 2024 file (negative control).

## Results at a glance

All figures are measured on the held-out test period (the last 20% of the timeline), with every threshold frozen on the validation period first. Every number is in `outputs/summary_*.json`.

| | ULB Credit Card (primary) | UPI 2024 (negative control) |
|---|---|---|
| Test set | 56,746 txns · 74 frauds | 50,000 txns · 96 frauds |
| Do-nothing accuracy (the trap) | 99.87%, 0 fraud caught | 99.81%, 0 fraud caught |
| Final model | CatBoost + class weights | none deployable (permutation p = 0.78) |
| Test PR-AUC (95% CI) | **0.80** (0.71–0.88) vs 0.0013 random | 0.0019 vs 0.0019 random |
| Three-band rule (test) | 60 blocked (54 fraud) · 125 OTP · 56,561 approved | approve everything is cost-optimal |
| Expected cost: no model → 0.5 → cost threshold → three-band | ₹3,70,000 → ₹92,700 → ₹86,600 → **₹82,675** | ₹4,80,000 (no rule beats it) |
| Best label-free detector | COPOD: 59% of fraud in its top 1% | at chance |

The rupee costs are stated assumptions (missed fraud ₹5,000, false block ₹300, OTP ₹15, OTP stops 70% of fraud), so rupee figures are scenario estimates. The organisers published no official values. The notebook config cell and the web app's Policy page recompute everything for any values.

---

## 1 · Quick start

### Option A — Google Colab (recommended)
1. Upload this repository folder to Google Drive, e.g. `MyDrive/fraudguard/`.
2. Put `creditcard.csv` (or `creditcard.zip`) and `upi_transactions 2024.csv` (or its zip) in `fraudguard/data/`. Zips are unpacked automatically.
3. Open `notebooks/fraudguard_pipeline.ipynb` in Colab and run this first cell:
   ```python
   from google.colab import drive; drive.mount('/content/drive')
   %cd /content/drive/MyDrive/fraudguard/notebooks
   !pip -q install xgboost lightgbm catboost shap imbalanced-learn optuna pyod
   ```
4. **Runtime → Run all.** This runs the credit-card dataset. For the UPI control, set `DATASET = "upi"` in the setup cell and run all again.

Runtime is about 17 minutes from scratch on a 2-core CPU, and faster on a Colab GPU (the deep models use CUDA automatically). Expensive steps are cached in `outputs/cache/`; delete that folder to recompute everything. A from-scratch rerun reproduces every reported number exactly (seed 42).

### Option B — local
```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.11+
pip install -r requirements.txt
cd notebooks
FG_DATASET=creditcard jupyter nbconvert --to notebook --execute fraudguard_pipeline.ipynb --output 01_creditcard_executed.ipynb
FG_DATASET=upi        jupyter nbconvert --to notebook --execute fraudguard_pipeline.ipynb --output 02_upi_executed.ipynb
```

### The web app (Netlify + Supabase)
The app in `web/` is a static site with no build step. Scoring, exact SHAP explanations, batch CSV scoring and the policy lab all run in the browser, from the exported CatBoost models in `web/models/`.

1. **Netlify:** *Add new site → Import from Git →* this repo. `netlify.toml` already sets the publish directory to `web`, with no build command.
2. **Supabase** (optional, enables sign-in and saving):
   1. Create a project.
   2. *SQL Editor →* paste and run `supabase/schema.sql`. This creates the tables, row-level security, the 50-scoring and 5-run limits, and a private `batches` storage bucket.
   3. *Authentication → Providers:* enable Email, and Google if you want it.
   4. *Authentication → URL configuration:* add your Netlify URL as Site URL and as a redirect URL.
   5. Copy the *Project URL* and the *anon public key* into `web/config.js`, then commit. The anon key is designed to be public; row-level security keeps each user's rows private.
3. Without keys, the app runs in guest mode. Everything works except saving.

The local Streamlit demo still works: `streamlit run app/streamlit_app.py`.

---

## 2 · Repository layout
```
fraudguard/
├── notebooks/
│   ├── fraudguard_pipeline.ipynb      ← single source pipeline (DATASET = creditcard | upi)
│   ├── fraudguard_pipeline.py         ← same notebook in jupytext percent format
│   ├── 01_creditcard_executed.ipynb   ← executed run with every output (primary)
│   └── 02_upi_executed.ipynb          ← executed run with every output (negative control)
├── src/fraudguard_lib.py              ← loaders + audit, split, cost model, band optimiser, model zoo, anomaly detectors
├── web/                               ← FraudGuard web app (Netlify): index.html, assets/, models/, config.js
├── supabase/schema.sql                ← tables, row-level security, limits, storage bucket
├── app/streamlit_app.py               ← local Streamlit demo
├── docs/
│   ├── ARCHITECTURE.md                ← system diagram + 6 architecture decision records
│   ├── FraudGuard_Business_Report.docx
│   ├── FraudGuard_Model_Comparison.docx
│   ├── FraudGuard_Cost_Analysis_and_Risk_Rule.docx
│   ├── FraudGuard_README.docx
│   └── FraudGuard_Presentation.pptx
├── outputs/
│   ├── figures/{creditcard,upi}/      ← every chart
│   ├── tables/{creditcard,upi}/       ← model comparison, PR-AUC CIs, anomaly table, risk bands, data audit (CSV)
│   └── summary_{creditcard,upi}.json  ← every headline number, machine-readable
├── models/fraudguard_creditcard.joblib
├── data/                              ← put the two CSVs / zips here (not committed)
├── netlify.toml · requirements.txt
```

## 3 · Problem statements → where they are answered

| PS | What we did | Where |
|---|---|---|
| 1 Data quality | Audit of missing values, duplicates, impossible amounts, timestamps, inconsistent categories and leakage columns, with the treatment of each, for both datasets | Notebook §1, `tables/*/data_quality_audit.csv`, Business Report §3 |
| 2 Fraud lens and accuracy trap | Target, fraud rate, class counts; a do-nothing model next to the real model on accuracy, recall, precision and ₹ | §2, §7; `08_accuracy_trap.png` |
| 3 EDA | Fraud rate by amount band, hour, bursts, zero-amount authorisations and a behaviour signature (credit card); by type, merchant, amount, hour, device and age (UPI). All with Wilson 95% CIs and χ² tests | §3 |
| 4 Supervised | 2 rule baselines. LR / RF / XGBoost / LightGBM × {none, class weights, SMOTE inside the pipeline}. CatBoost, Balanced RF, EasyEnsemble, Optuna-tuned LightGBM, focal-loss MLP, FT-Transformer, hybrid, soft-vote ensemble. Time-aware split | §4, §6, Model Comparison doc |
| 5 Anomaly | Isolation Forest, ECOD, COPOD, LOF, autoencoder and DBSCAN, fitted without labels; evaluated afterwards (fraud in top 0.5 / 1 / 5%) and compared with supervised | §5 |
| 6 Cost threshold | PR curves, exact cost curve, threshold vs default 0.5, ₹X × ₹Y grid, ₹Z × OTP-stop grid, bootstrap CIs | §7, Cost Analysis doc |
| 7 Risk rule | Exact two-threshold search on validation; per-band counts, fraud caught, genuine customers challenged or blocked, ₹, cost bridge | §8, Cost Analysis doc |
| 8 Explain and act | TreeSHAP (additivity-checked), three findings → three prioritised actions, limitations | §9, Business Report |

## 4 · Assumptions (editable in the notebook setup cell and the app's Policy page)

| Assumption | Value | Why |
|---|---|---|
| Missed fraud (₹X) | ₹5,000 | Average loss + chargeback and handling; no official figure was published |
| Genuine blocked (₹Y) | ₹300 | Lost transaction margin, support call, churn risk |
| Step-up OTP (₹Z) | ₹15 per challenged txn | Mostly genuine drop-off (≈5% of challenged customers × ₹300), plus SMS (well under ₹1) |
| OTP stops fraud | 70% | Stolen-credential and account-takeover fraud fails OTP; socially engineered victims pass it |
| Credit-card hour of day | capture starts 00:00 | Volume trough at 01–06h supports this |
| Credit-card `Amount` currency | not stated by the source | Rupee costs are per event, never a share of the amount |

## 5 · Guard-rails from the brief, and how the code enforces them
- **No resampling before the split:** SMOTE exists only inside `imblearn.Pipeline`, fitted on train.
- **No training or tuning on test:**
  - Optuna uses `TimeSeriesSplit` inside train.
  - Deep-model early stopping uses the last 15% of train.
  - Model choice, calibration and thresholds use validation.
  - Test is scored once.
- **No leakage columns:** UPI `transaction_status` (post-outcome) and the IDs are excluded. The credit-card data has none.
- **No accuracy-only reporting:** accuracy appears only beside recall, precision, PR-AUC and ₹.
- **No causal claims:** EDA and SHAP are described as associations.
- **Projections labelled:** every per-million-transaction figure is marked as a scenario estimate.
- **Fixed seed:** `SEED = 42` everywhere.

## 6 · Notes for Q&A
- **Why UPI is a control, not a model:**
  - None of 13 χ² tests survives correction.
  - All 22 models score test ROC-AUC 0.47–0.54.
  - The validation "winner" (p = 0.016) falls to p = 0.78 on test, which is what picking the best of 22 models on noise produces.
- **Ensemble members are fixed in advance:** picking the top 3 on validation and then scoring the ensemble there would be selection bias.
- **SMOTE vs class weights:** class weights win clearly for logistic regression (PR-AUC 0.753 vs 0.705). For tree ensembles the regimes are within noise.
- **Deep tabular models:** the FT-Transformer and the focal-loss MLP trail the GBDTs on validation, consistent with Grinsztajn et al. (2022).
- **SHAP in the app:** the browser computes exact path-dependent TreeSHAP. It matches brute-force Shapley enumeration to 2×10⁻¹⁶ and is additive to the model output. CatBoost's built-in SHAP (used for the notebook figures) splits credit differently when a tree reuses a feature; both are exact under their own definitions.

## 7 · Rebuilding `.ipynb` from the `.py` source
```bash
pip install jupytext && jupytext --to ipynb notebooks/fraudguard_pipeline.py
```

## 8 · End-goal checklist (problem statement §7, §12, §13, §14)

| Requirement | Where it is met |
|---|---|
| Data-quality summary: missing values, duplicates, excluded leakage columns, justified treatment | Notebook §1 · Business Report §3 · `tables/*/data_quality_audit.csv` · app Evidence page |
| Accuracy trap: do-nothing model with very high accuracy and zero recall, beside real metrics | Notebook §2/§7 · `08_accuracy_trap.png` · deck slide 3 · Business Report §4 |
| ≥ 3 meaningful EDA fraud comparisons | Amount band, hour of day, bursts over time, zero amounts, behaviour signature (credit card); type, merchant, amount, hour, device, age (UPI) |
| ≥ 3 supervised models vs a rule baseline, with and without imbalance handling | 22 models + 2 rules; LR/RF/XGB/LGBM × none / class weights / SMOTE-in-pipeline · Model Comparison doc |
| Anomaly detector trained without labels, evaluated afterwards | Isolation Forest, ECOD, COPOD, LOF, autoencoder, DBSCAN · fraud in top 0.5/1/5% · compared with supervised |
| Precision–recall curve and cost-versus-threshold chart | `07_pr_curves.png`, `09_cost_threshold_confusion.png` · Cost Analysis doc |
| Risk-score rule with approve / step-up OTP / block bands, counts and costs per band | Notebook §8 · Cost Analysis doc §5 · `tables/creditcard/risk_bands_test.csv` · app Policy page |
| Three evidence-backed findings, three prioritised recommendations, limitations | Business Report §9–§11 · deck slides 23, 25 |
| Source code / notebook, runnable, fixed seed | `notebooks/` (seed 42; from-scratch rerun reproduces every number) |
| Model comparison table with final selection reasoning | `docs/FraudGuard_Model_Comparison.docx` · `tables/*/model_comparison.csv` |
| Cost analysis with ₹ assumptions | `docs/FraudGuard_Cost_Analysis_and_Risk_Rule.docx` (no official ₹ values were published; stated assumptions are used, and the Policy page and config cell recompute for any official values) |
| Business report 4–6 pages | `docs/FraudGuard_Business_Report.docx` (5 pages) |
| Presentation 5–10 min incl. accuracy-trap slide | `docs/FraudGuard_Presentation.pptx` (26 slides incl. 5 section dividers, speaker notes, transitions and animations) |
| README: setup, dependencies, assumptions, execution | this file · `docs/FraudGuard_README.docx` |
| Demo | Web app (`web/`, Netlify + Supabase) · Streamlit (`app/`) |
| Boundaries: no accuracy-only reporting, no resampling before the split, no leakage columns, no tuning on test, no causal claims, no external data, projections labelled | §5 above; enforced in code |

## 9 · Credits
**Datasets:**
- ULB Machine Learning Group and Worldline, *Credit Card Fraud Detection* (Kaggle; Dal Pozzolo et al., 2015).
- *UPI Transactions 2024* (organiser-supplied, simulated).

**Methods:** Chawla et al. 2002 (SMOTE) · Liu et al. 2008 (Isolation Forest) · Lin et al. 2017 (focal loss) · Lundberg & Lee 2017; Lundberg et al. 2020 (SHAP/TreeSHAP) · Li et al. 2020 (COPOD) · Gorishniy et al. 2021 (FT-Transformer) · Li et al. 2022 (ECOD) · Grinsztajn et al. 2022.
