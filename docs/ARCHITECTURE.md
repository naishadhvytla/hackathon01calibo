# FraudGuard — Architecture & Decision Records

System goal: score every payment in real time, and turn the score into **approve / step-up OTP / block**. Choose that rule by expected rupee cost, not accuracy.

```
            ┌──────────────────────── offline (notebook) ───────────────────────┐
 raw CSV ─► │ audit & clean ─► time-aware split ─► feature build                │
            │        │                 │  (train 60 | val 20 | test 20)        │
            │        ▼                 ▼                                        │
            │  leakage list     ┌─────────────┐   ┌──────────────────────┐      │
            │                   │ Supervised  │   │ Unsupervised (no y)  │      │
            │                   │ GBDT zoo,   │   │ IForest, ECOD, COPOD,│      │
            │                   │ LR, RF, MLP │   │ LOF, Autoencoder     │      │
            │                   └─────┬───────┘   └─────────┬────────────┘      │
            │                         └───► hybrid stack ◄──┘                   │
            │                                   │                               │
            │                         isotonic calibration (val)                │
            │                                   │                               │
            │                cost curve ─► T_otp, T_block (val)                 │
            │                                   │                               │
            │                      single honest evaluation (test)              │
            └───────────────────────────────────┬───────────────────────────────┘
                                                ▼
 online (Streamlit demo):  txn ─► features ─► model.predict_proba ─► calibrated p
                           ─► band(p; T_otp, T_block) ─► APPROVE / OTP / BLOCK + SHAP reasons
```

---

## ADR-001: Dataset strategy — run both, and treat UPI as a negative control

**Status:** Accepted · **Date:** 2026-10-31 · **Deciders:** Team

**Context.** The brief offers two datasets: ULB Credit Card (284,807 rows, 492 frauds) and UPI Transactions 2024 (250,000 rows, 480 frauds). It also says the organiser "must inspect [UPI] and confirm the fraud share and columns". We profiled UPI first. Every column's fraud rate falls within chance (chi-square p > 0.1 for 11 of 12 columns, and the remaining p = 0.026 does not survive Bonferroni correction). A 5-fold LightGBM scores ROC-AUC 0.49 and PR-AUC 0.0019, which equals the base rate. UPI also has no sender or account ID, so account-behaviour features cannot be built.

**Decision.** Both datasets go through the identical pipeline. ULB Credit Card is the **primary** dataset: we select the model, set thresholds and make recommendations on it. UPI is reported as a **negative control**: the same code, and evidence that the label carries no learnable signal.

| Option | Complexity | Value | Risk |
|---|---|---|---|
| A. UPI only | Low | On-theme | Every model is at chance, so there is no business value |
| B. Credit card only | Low | Real signal | Ignores the UPI framing in the brief |
| **C. Both, UPI as control (chosen)** | Medium | Real signal, plus a data-trust lesson | Two result sets to explain |

**Consequences.** The UPI result becomes a second "accuracy trap": 99.8% accuracy, and models that learn nothing. Recommendation #1 is therefore about data and labelling, not models.

## ADR-002: Time-aware split; the test set is touched once

**Context.** Fraud patterns drift. A random split leaks future patterns into training, and the brief forbids tuning on test.

**Decision.** Sort by time. Train = first 60%, validation = next 20%, test = last 20%. All hyperparameter search, calibration, model selection and threshold selection use **validation only**. Test is scored once, at the end. We report 1,000-sample bootstrap 95% CIs on test PR-AUC and cost.

**Trade-off.** The training data has fewer frauds than a stratified split would give, but the evaluation mirrors production (train on the past, score the future).

## ADR-003: Imbalance handling stays inside the pipeline

**Decision.** We compare three regimes for every model family: (a) no handling, (b) class weights / `scale_pos_weight`, and (c) SMOTE inside an `imblearn.Pipeline`, so synthetic points are created from training data only and never from validation or test. No resampling happens before the split.

**Trade-off.** SMOTE distorts predicted probabilities, so every candidate is isotonic-calibrated on validation before costs are computed.

## ADR-004: Model family — gradient-boosted trees, with deep and anomaly challengers

**Options.** Logistic Regression (interpretable baseline) · Random Forest / Balanced RF · EasyEnsemble · XGBoost · LightGBM (+ Optuna) · CatBoost · focal-loss MLP · FT-Transformer · soft-vote ensemble (members fixed a priori, to avoid selection bias on validation).

**Decision.** We select on **validation PR-AUC first, then validation expected cost**, with latency and explainability as tie-breakers. A GBDT is the expected winner on tabular PCA features, because it is fast (<1 ms per transaction), has an exact TreeSHAP and is robust. The deep models are included to show we tested the research frontier, not to win by default.

## ADR-005: Label-free anomaly layer

**Decision.** We fit Isolation Forest, ECOD, COPOD, LOF (novelty mode) and a reconstruction autoencoder on **training features with no labels**. Labels are used only afterwards to measure the share of fraud caught in the top 1% of scores, and PR-AUC. The best detector's score is also tested as an extra feature for the supervised model (the hybrid), which covers novel fraud that has no labels yet.

## ADR-006: Decision layer — cost-minimising two thresholds

**Cost model (config cell, editable):**

| Outcome | Default cost |
|---|---|
| Fraud approved (missed) | ₹X = ₹5,000 |
| Genuine blocked | ₹Y = ₹300 |
| Any transaction sent to OTP | ₹Z = ₹15 (SMS + friction) |
| Fraud sent to OTP | X × (1 − OTP stop rate); OTP stop rate = 70% |

**Decision.** (1) A binary threshold minimises `FN·X + FP·Y` on validation (PS6). (2) A grid search over (T_otp < T_block) minimises the full three-outcome cost on validation (PS7). Both are frozen and then reported on test. A sensitivity grid over X and Y shows how the thresholds move.

**Consequences.** Savings are reported as *scenario estimates*, conditional on these assumed costs, and kept separate from the observed confusion-matrix counts.

---

## Outcome log (filled in after the runs)

| ADR | What the data said |
|---|---|
| 001 | UPI is confirmed no-signal on test (all 22 models ROC-AUC 0.47–0.54; permutation p = 0.78). Credit card is the primary dataset. |
| 002 | The validation fraud rate (0.10%) differs from train (0.20%) and test (0.13%): drift is real, and the time split exposes it. |
| 003 | Class weights ≥ SMOTE for 3 of 4 families; SMOTE leads only LightGBM, by 0.002 PR-AUC (noise). |
| 004 | CatBoost (class weights) selected: validation PR-AUC 0.784, lowest validation cost ₹66,300, test PR-AUC 0.80 (95% CI 0.71–0.88). Deep models trail. |
| 005 | COPOD puts 59% of test fraud in its top 1% without labels; as hybrid features, anomaly scores add nothing. Used as a watch-list. |
| 006 | Thresholds T_otp = 0.0093 and T_block = 0.766 → test cost ₹82,690 vs ₹92,700 at 0.5 and ₹3,70,000 with no model. |
