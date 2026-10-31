// Builds the three FraudGuard Word documents from the pipeline outputs (numbers are read, not typed).
const fs = require("fs"), path = require("path");
const L = require("./lib");
const { P, H1, H2, Bul, Num, Caption, Callout, Img, Tbl, makeDoc, save, inr, inrFull, pct, csv, PORTRAIT_W: PW, LANDSCAPE_W: LW, PageBreak } = L;
const ROOT = path.resolve(__dirname, "../..");
const OUT = path.join(ROOT, "docs");
const J = (f) => JSON.parse(fs.readFileSync(f, "utf8").replace(/\bNaN\b/g, "null").replace(/-?Infinity\b/g, "null"));
const S = J(path.join(ROOT, "outputs/summary_creditcard.json"));
const U = J(path.join(ROOT, "outputs/summary_upi.json"));
const F = (d, f) => path.join(ROOT, "outputs/figures", d, f);
const cmp = csv(path.join(ROOT, "outputs/tables/creditcard/model_comparison.csv"));
const ci = Object.fromEntries(csv(path.join(ROOT, "outputs/tables/creditcard/model_pr_auc_ci.csv")).map((r) => [r.model, r]));
const cmpU = csv(path.join(ROOT, "outputs/tables/upi/model_comparison.csv"));
const anom = csv(path.join(ROOT, "outputs/tables/creditcard/anomaly_detectors.csv"));
const anomU = csv(path.join(ROOT, "outputs/tables/upi/anomaly_detectors.csv"));
const auditC = csv(path.join(ROOT, "outputs/tables/creditcard/data_quality_audit.csv"));
const auditU = csv(path.join(ROOT, "outputs/tables/upi/data_quality_audit.csv"));
const FINAL = S.final_model;
const B = Object.fromEntries(S.bands.test.map((b) => [b.band, b]));
const BV = Object.fromEntries(S.bands.val.map((b) => [b.band, b]));
const pol = S.policies_test, approveAll = pol["Approve everything (no model)"].test_cost;
const three = pol["3-band rule (approve / OTP / block)"].test_cost, def05 = pol["Model @ default 0.5 (block/approve)"].test_cost, costThr = pol["Model @ cost threshold (block/approve)"].test_cost;
const T = S.threshold, dn = S.do_nothing_test, md = T.test_default, mc = T.test_cost;
const C = S.costs;
const fmtT = (t) => (t < 0.01 ? t.toExponential(2) : t.toFixed(3));
const ov = S.sup_vs_anom_top1, bestA = anom[0];
const nTest = S.n_test, fTest = S.test_frauds;
const caughtOrChallenged = B["BLOCK"].frauds + B["STEP-UP OTP"].frauds;
const upiThree = U.policies_test["3-band rule (approve / OTP / block)"].test_cost, upiAll = U.policies_test["Approve everything (no model)"].test_cost;
const upiRoc = cmpU.map((r) => r.test_roc_auc), upiAp = cmpU.map((r) => r.test_pr_auc);
const range = (a, d = 2) => `${Math.min(...a).toFixed(d)}–${Math.max(...a).toFixed(d)}`;
const sens = S.sensitivity, otp = S.otp_sensitivity;
const kfmt = (v) => (v >= 1000 ? (v / 1000) + "k" : String(v));

/* ============================================================ 1. MODEL COMPARISON */
async function modelComparison() {
  const fam = (f) => cmp.filter((r) => r.family === f);
  const regimeRow = (f) => { const g = fam(f); const by = (k) => g.find((r) => r.imbalance.startsWith(k)); const v = (r) => (r ? `${r.val_pr_auc.toFixed(3)} / ${inr(r.val_cost)}` : "—"); return [f, v(by("none")), v(by("class weights")), v(by("SMOTE"))]; };
  const rows = cmp.map((r) => [r.model + (r.model === FINAL ? "  ◆" : ""), r.imbalance, r.val_pr_auc.toFixed(3), `${r.test_pr_auc.toFixed(3)} (${ci[r.model].ci_lo.toFixed(2)}–${ci[r.model].ci_hi.toFixed(2)})`,
    r.test_roc_auc.toFixed(3), pct(r.test_precision), pct(r.test_recall), r.test_f1.toFixed(3), inr(r.val_cost), inr(r.test_cost), isNaN(r.ms_per_txn) || r.ms_per_txn === "" ? "—" : (r.ms_per_txn * 1000).toFixed(1)]);
  const finalRow = cmp.find((r) => r.model === FINAL);
  const doc = makeDoc("Model Comparison", "All supervised models, imbalance methods, label-free detectors, metrics and the final selection with reasoning · ULB credit card (primary) and UPI 2024 (negative control) · seed 42", [
    { children: [
      H1("1. The decision in one paragraph"),
      Callout(`**Selected model: ${FINAL}.** It has the best validation PR-AUC (${finalRow.val_pr_auc.toFixed(3)}) and the lowest validation cost (${inrFull(finalRow.val_cost)}) among the shortlisted models. On the untouched test period it reaches PR-AUC ${finalRow.test_pr_auc.toFixed(3)} (95% CI ${T.ci_pr_auc[0].toFixed(2)}–${T.ci_pr_auc[1].toFixed(2)}), against ${(fTest / nTest).toFixed(4)} for a random ranking, and a label-permutation test gives p < 0.002. It scores a transaction in about ${(finalRow.ms_per_txn * 1000).toFixed(1)} µs and has an exact TreeSHAP explanation.`),
      P(`We compared **${cmp.length} supervised candidates** and **${anom.length + 1} label-free detectors**. The selection rule was written before test was opened. The top six models overlap within their confidence intervals on test, so the choice rests on cost and operability, not on a clear accuracy gap.`),
      H1("2. Protocol (how every model was treated)"),
      Tbl(["Split", "Rows", "Frauds", "Fraud rate", "Used for"], S.split.map((s) => [s.split, s.rows.toLocaleString("en-IN"), s.frauds, s.fraud_rate_pct.toFixed(3) + "%",
        { train: "fitting every model; Optuna time-series folds; deep-model early stopping (last 15%)", validation: "model selection, calibration, both thresholds", test: "scored once, at the end" }[s.split]]), [1300, 1300, 1000, 1200, 4514], { right: [1, 2, 3] }),
      Caption("Time-ordered split of the de-duplicated credit-card data (283,726 rows). The fraud rate drifts (0.20% → 0.10% → 0.13%), which a random split would hide."),
      Bul("**Imbalance handling stays inside the training pipeline.** SMOTE runs inside an `imblearn.Pipeline`, so synthetic frauds come only from the training fold. Nothing is resampled before the split."),
      Bul("**Metrics:** PR-AUC (primary, threshold-free), ROC-AUC (with a caution), and precision, recall, F1 and ₹ cost, each at the model's own validation cost-optimal threshold. Test PR-AUC has a 200-sample stratified bootstrap 95% CI."),
      Bul("**Baselines a fraud team could run without ML:** a business rule (night and amount above the 95th percentile) and a learned two-split decision tree (V14 ≤ −1.63, then V10)."),
      H1("3. Full comparison table (credit card)"),
      P("◆ = selected. Validation columns drove the choice; test columns are reported for transparency. Latency is measured on batch scoring, per transaction, on a 2-core CPU."),
    ] },
    { landscape: true, children: [
      Tbl(["Model", "Imbalance handling", "Val PR-AUC", "Test PR-AUC (95% CI)", "Test ROC-AUC", "Precision*", "Recall*", "F1*", "Val ₹ cost*", "Test ₹ cost*", "µs / txn"], rows,
        [2900, 1900, 900, 1600, 950, 950, 850, 750, 1100, 1100, 1246], { size: 15, right: [2, 3, 4, 5, 6, 7, 8, 9, 10], hl: [cmp.findIndex((r) => r.model === FINAL)] }),
      Caption("* at each model's own cost-optimal threshold chosen on validation (missed fraud ₹5,000, false block ₹300). Sorted by validation PR-AUC."),
      Img(F("creditcard", "05c_metric_heat_table.png"), 9.6, 6.0),
      Caption("Figure 1. Metric heat-table for all 22 models (colour = rank within each column)."),
    ] },
    { children: [
      H1("4. Combined views"),
      Img(F("creditcard", "05b_all_models_pr_auc_ci.png"), 6.4, 6.4),
      Caption("Figure 2. Test PR-AUC with 95% bootstrap CI (dots) and validation PR-AUC (bars) for every model. The gradient-boosted trees and forests overlap; rule baselines sit far left."),
      Img(F("creditcard", "07_pr_curves.png"), 6.4),
      Caption("Figure 3. Precision–recall curves on the test period for the main candidates, the best label-free detector and the learned rule."),
      Img(F("creditcard", "05e_roc_curves.png"), 4.8),
      Caption("Figure 4. ROC on test with a log x-axis. Every model looks excellent by ROC-AUC (0.94–0.99), which is why PR-AUC and cost drive selection at 0.13% prevalence."),
      H1("5. Individual views"),
      Img(F("creditcard", "05d_individual_pr_curves.png"), 6.4, 8.0),
      Caption("Figure 5. One precision–recall panel per model: solid = test, dashed = validation. Close solid and dashed curves mean the model generalises to the later period."),
      H1("6. Imbalance methods: class weights vs SMOTE vs none"),
      Tbl(["Family", "No handling (val PR-AUC / val ₹)", "Class weights", "SMOTE in pipeline"], ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"].map(regimeRow), [2000, 2438, 2438, 2438], { right: [1, 2, 3] }),
      Caption("Validation PR-AUC / validation cost at the best threshold. Also tested: Balanced RF and EasyEnsemble (undersampling per tree or member), CatBoost (√-balanced weights), focal loss for the neural models, and a tuned scale_pos_weight in Optuna."),
      Img(F("creditcard", "05_weights_vs_smote.png"), 6.4),
      Caption("Figure 6. Imbalance handling by model family."),
      Bul("**Logistic regression needs imbalance handling.** Class weights lift validation PR-AUC from 0.608 to 0.753; SMOTE reaches 0.705."),
      Bul("**For tree ensembles the three regimes are within noise** (±0.02), well inside the ±0.08 bootstrap CI."),
      Bul("**Decision:** class weights. No synthetic transactions, no extra training time, and probabilities that stay closer to calibrated."),
      H1("7. Research models"),
      P(`The **FT-Transformer** (Gorishniy et al., 2021) and a **focal-loss MLP** (Lin et al., 2017) were trained with early stopping on the last 15% of train. They reached validation PR-AUC ${cmp.find((r) => r.model.startsWith("FT")).val_pr_auc.toFixed(3)} and ${cmp.find((r) => r.model.startsWith("Deep")).val_pr_auc.toFixed(3)}, below every gradient-boosted tree. This matches the published finding that tree ensembles still lead on mid-sized tabular data (Grinsztajn et al., 2022). **Optuna** (30 TPE trials on time-series folds inside train) did not beat default LightGBM on validation. The **hybrid** (tuned LightGBM plus three anomaly scores as features) also added nothing.`),
      H1("8. Label-free anomaly detectors (trained without labels)"),
      Tbl(["Detector", "Test PR-AUC", "Test ROC-AUC", "Fraud in top 0.5%", "Fraud in top 1%", "Precision in top 1%", "Fraud in top 5%"],
        anom.map((a) => [a.model, a.test_pr_auc.toFixed(3), a.test_roc_auc.toFixed(3), pct(a.recall_top_0_5pct, 0), pct(a.recall_top_1pct, 0), pct(a.precision_top_1pct), pct(a.recall_top_5pct, 0)]).concat([["DBSCAN (10k test sample)", "—", "—", "—", `${S.dbscan.fraud_caught}/${S.dbscan.fraud_in_sample} in ${S.dbscan.flagged_pct.toFixed(2)}% flagged`, pct(S.dbscan.precision), "—"]]),
        [2300, 1050, 1050, 1200, 1300, 1250, 1164], { right: [1, 2, 3, 4, 5, 6] }),
      Img(F("creditcard", "06_anomaly_vs_supervised.png"), 6.4),
      Caption("Figure 7. COPOD score distribution (left) and share of test fraud in the top 1% of scores, label-free detectors vs the supervised model (right)."),
      P(`**Comparison with supervised:** COPOD puts ${ov.anom_top1} of ${ov.fraud_total} test frauds in its top 1%, against ${ov.sup_top1} for ${FINAL}. ${ov.both} frauds are in both lists; ${ov.anom_only} is caught only by the detector. The detector therefore works as a **second, label-free watch-list** for new fraud patterns, not as the blocking model.`),
      H1("9. Selection rule and reasoning"),
      Num("Shortlist: every model within 0.01 of the best validation PR-AUC (" + S.shortlist.join(", ") + ")."),
      Num(`Lowest validation ₹ cost at its own cost-optimal threshold: ${FINAL} (${inrFull(finalRow.val_cost)}).`),
      Num("Tie-breakers: a single model over an ensemble (latency, exact SHAP), then the fastest."),
      Num(`Signal check: label-permutation test on validation and test, p < 0.002 for ${FINAL}.`),
      P("**Fixed in advance to avoid selection bias:** the ensemble's members (one per GBDT library) were chosen before any validation score was seen. An ensemble of the top 3 on validation, judged on that same validation set, would look better than it is."),
      H1("10. UPI 2024: the same comparison is a negative control"),
      Callout(`All ${cmpU.length} UPI models score test ROC-AUC ${range(upiRoc)} and test PR-AUC ${range(upiAp, 4)}, against a base rate of 0.0019. The validation "winner" (${U.final_model}, permutation p = ${U.signal.validation.p_value.toFixed(3)}) falls to p = ${U.signal.test.p_value.toFixed(2)} on test, which is what picking the best of ${cmpU.length} models on noise produces. Every anomaly detector catches at most ${Math.max(...anomU.map((a) => Math.round(a.recall_top_1pct * 96)))} of 96 test frauds in its top 1%. **No UPI model is deployable**, and approving everything is the cost-optimal policy.`),
      Img(F("upi", "05b_all_models_pr_auc_ci.png"), 6.4, 6.0),
      Caption("Figure 8. UPI: every model's test PR-AUC sits on the random line."),
      Img(F("upi", "07_pr_curves.png"), 6.4),
      Caption("Figure 9. UPI precision–recall curves (y-axis zoomed to 0–2%)."),
      P("Reproduce with `notebooks/fraudguard_pipeline.ipynb` (seed 42). Tables: `outputs/tables/{creditcard,upi}/model_comparison.csv` and `model_pr_auc_ci.csv`. A from-scratch rerun reproduces every number in this document exactly."),
    ] },
  ]);
  await save(doc, path.join(OUT, "FraudGuard_Model_Comparison.docx"));
}

/* ============================================================ 2. COST ANALYSIS & RISK RULE */
async function costAnalysis() {
  const bandRows = (bb) => ["APPROVE", "STEP-UP OTP", "BLOCK"].map((k) => { const b = bb[k]; return [k, b.transactions.toLocaleString("en-IN"), b.pct_of_txns.toFixed(2) + "%", b.frauds, b.pct_of_all_fraud.toFixed(1) + "%",
    k === "APPROVE" ? `0 (${b.genuine.toLocaleString("en-IN")} pass)` : k === "BLOCK" ? `${b.genuine} blocked` : `${b.genuine} challenged`, b.fraud_rate_in_band_pct.toFixed(2) + "%", inrFull(b.cost_from_fraud), inrFull(b.cost_from_genuine), inrFull(b.total_cost)]; });
  const sensRows = sens.Y.map((y, i) => [`₹${y}`, ...sens.X.map((x, j) => `${sens.thr[i][j] < 0.01 ? sens.thr[i][j].toExponential(1) : sens.thr[i][j].toFixed(3)} · ${pct(sens.recall[i][j], 0)}`)]);
  const otpRows = otp.stop.map((s, i) => [pct(s, 0), ...otp.Z.map((z, j) => `${inr(otp.test_cost[i][j])} · OTP ${otp.otp_n[i][j]}`)]);
  const bv = S.band_value;
  const doc = makeDoc("Cost Analysis & Risk-Score Rule", "From model score to approve / step-up OTP / block, using expected rupee cost · thresholds chosen on validation, frozen, scored once on test · ULB credit card (primary), UPI 2024 (control)", [{ children: [
    H1("1. Result"),
    Callout(`On the unseen test period (${nTest.toLocaleString("en-IN")} payments, ${fTest} frauds, about 7.7 hours of traffic), the three-band rule stops or challenges **${caughtOrChallenged} of ${fTest} frauds (${pct(caughtOrChallenged / fTest)})**. It blocks only 6 genuine customers and challenges ${B["STEP-UP OTP"].genuine} with an OTP. Expected cost falls from **${inrFull(approveAll)}** with no model to **${inrFull(three)}** (95% CI ${inr(S.three_band_ci[0])}–${inr(S.three_band_ci[1])}), a 78% reduction. The default 0.5 threshold costs ${inrFull(def05)}.`),
    H1("2. The cost model"),
    P("Expected cost = missed fraud × ₹X + genuine blocked × ₹Y + every OTP challenge × ₹Z + fraud that passes OTP × ₹X."),
    Tbl(["Assumption", "Value", "Basis"], [
      ["Missed fraud (₹X)", inrFull(C.X), "Average loss + chargeback handling + investigation; the organisers published no official figure"],
      ["Genuine payment blocked (₹Y)", inrFull(C.Y), "Lost transaction margin, a support call, churn risk"],
      ["Step-up OTP (₹Z)", inrFull(C.Z) + " per challenge", "Mostly genuine drop-off: about 5% of challenged customers abandon × ₹300 ≈ ₹15. The SMS itself costs well under ₹1"],
      ["Fraud stopped by OTP", pct(C.otp_stop_rate, 0), "Stolen-credential and account-takeover fraud fails OTP; socially engineered victims type the OTP themselves"],
      ["Cost basis", "per event", "The credit-card Amount column has no stated currency (the source says only \"European cardholders\"), so it is never converted to rupees"],
    ], [2400, 1500, 5414]),
    P("**Official costs.** The brief refers to organiser-published ₹ figures, but none were provided. Every number here comes from one config cell in the notebook (`CostConfig`) and the **Policy** page of the web app. Entering official values there re-optimises both thresholds on validation and re-scores test, with no retraining. Sections 4 and 7 show how far the answer moves when the costs change."),
    H1("3. Cost-based threshold (two outcomes: approve or block)"),
    Img(F("creditcard", "09_cost_threshold_confusion.png"), 6.4),
    Caption("Figure 1. Left: expected cost on validation at every threshold (x = recall reached), with the minimum and the 0.5 default marked. Right: test confusion matrix at the chosen threshold, in rupees."),
    Tbl(["Policy (test period)", "Accuracy", "Recall", "Precision", "Fraud missed", "Genuine blocked", "Expected cost"], [
      ["Do-nothing (approve all)", pct(dn.accuracy, 2), "0%", "—", dn.fn, 0, inrFull(dn.cost_inr)],
      [`${FINAL} @ default 0.5`, pct(md.accuracy, 2), pct(md.recall), pct(md.precision), md.fn, md.fp, inrFull(md.cost_inr)],
      [`${FINAL} @ cost threshold ${T.t_cost.toFixed(3)}`, pct(mc.accuracy, 2), pct(mc.recall), pct(mc.precision), mc.fn, mc.fp, inrFull(mc.cost_inr)],
    ], [2900, 900, 800, 900, 1000, 1100, 1714], { right: [1, 2, 3, 4, 5, 6], hl: [2] }),
    Caption(`Bootstrap 95% CIs on test: recall ${pct(T.ci_recall[0])}–${pct(T.ci_recall[1])}, cost ${inrFull(T.ci_cost[0])}–${inrFull(T.ci_cost[1])}. The cost-optimal threshold is less accurate than 0.5 but ${inrFull(md.cost_inr - mc.cost_inr)} cheaper. Accuracy cannot rank the two.`),
    H1("4. How the threshold moves with ₹X and ₹Y"),
    Tbl(["₹Y \\ ₹X", ...sens.X.map((x) => `₹${x.toLocaleString("en-IN")}`)], sensRows, [1314, 1600, 1600, 1600, 1600, 1600], { boldFirst: true }),
    Caption("Each cell: cost-optimal threshold chosen on validation · test recall at that threshold. Base case: row ₹300, column ₹5,000."),
    Img(F("creditcard", "10_cost_sensitivity.png"), 6.4),
    Caption("Figure 2. Threshold sensitivity. Across 25 scenarios only four distinct thresholds appear, so the decision is stable across the plausible range."),
    H1("5. The risk-score rule (three outcomes)"),
    P(`Both thresholds are found by an **exact search over every distinct validation score in the top 5%**, minimising the full three-outcome cost. The same rule runs in the web app. Result: **approve below ${fmtT(S.bands.T_otp)}; step-up OTP from ${fmtT(S.bands.T_otp)} to ${fmtT(S.bands.T_block)}; block at ${fmtT(S.bands.T_block)} and above.** On the calibrated 0–100 risk score (isotonic regression fitted on validation) that is OTP from ${S.bands.risk_otp.toFixed(1)} and block from ${S.bands.risk_block.toFixed(1)}. As a theory cross-check, Bayes-optimal cut-offs on a calibrated probability are OTP above ${pct(S.bayes_bands.p_otp, 2)} and block above ${pct(S.bayes_bands.p_block)}: the same order of decisions.`),
    Tbl(["Band (test)", "Payments", "Share", "Frauds", "Share of fraud", "Genuine customers", "Fraud rate in band", "₹ from fraud", "₹ from genuine", "Total ₹"], bandRows(B),
      [1150, 950, 750, 650, 850, 1250, 850, 900, 900, 1064], { size: 15, right: [1, 2, 3, 4, 6, 7, 8, 9] }),
    Caption("Test period, thresholds frozen from validation. OTP band: each challenge ₹15; fraud passing OTP (30%) costs ₹5,000."),
    Tbl(["Band (validation)", "Payments", "Share", "Frauds", "Share of fraud", "Genuine customers", "Fraud rate in band", "₹ from fraud", "₹ from genuine", "Total ₹"], bandRows(BV),
      [1150, 950, 750, 650, 850, 1250, 850, 900, 900, 1064], { size: 15, right: [1, 2, 3, 4, 6, 7, 8, 9] }),
    Caption("Validation period, where the thresholds were chosen. Test behaves the same way, which is evidence that the rule generalises."),
    Img(F("creditcard", "11_risk_bands.png"), 6.4),
    Caption("Figure 3. Test scores with the two thresholds (left) and where fraud and genuine customers land (right)."),
    H1("6. Where the money goes"),
    Img(F("creditcard", "11b_cost_waterfall.png"), 6.4),
    Caption("Figure 4. Cost bridge on test. Blocking removes most of the expected loss; OTP removes more; the friction it adds (OTP challenges, 6 blocked customers) is small."),
    Tbl(["Band", "Fraud amount in band", "Share of fraud amount", "Genuine amount in band"], bv.map((b) => [b.band, b.fraud_value.toLocaleString("en-IN", { maximumFractionDigits: 0 }), b.share_of_fraud_value_pct.toFixed(1) + "%", b.genuine_value.toLocaleString("en-IN", { maximumFractionDigits: 0 })]), [2300, 2300, 2300, 2414], { right: [1, 2, 3] }),
    Caption("Transaction value by band, in the dataset's own (unstated) currency units, not rupees. Blocked and OTP payments hold 69% of fraudulent value but only 1.1% of genuine value."),
    H1("7. Stress-testing the OTP assumptions"),
    Tbl(["OTP stops \\ ₹Z", ...otp.Z.map((z) => `₹${z}`)], otpRows, [1414, 1580, 1580, 1580, 1580, 1580], { boldFirst: true, size: 16 }),
    Caption("Each cell: test cost of the three-band rule re-optimised on validation · payments sent to OTP. The block band is 60 payments in every cell. The base case (₹15, 70%) is the middle cell."),
    Img(F("creditcard", "10b_otp_sensitivity.png"), 6.4),
    Caption("Figure 5. A cheaper or stronger OTP widens the OTP band and lowers cost. A dearer or weaker OTP shrinks it, and at ₹50 with 50% the rule falls back to approve/block. Non-monotone cells reflect validation noise (57 frauds)."),
    H1("8. Observed results vs projected savings"),
    Tbl(["Policy", "Observed test cost", "Saving vs no model (observed)", "Cost per 1M payments (scenario estimate)"], [
      ["Approve everything (no model)", inrFull(approveAll), "—", inr(S.projection.per_1m_no_model)],
      ["Model @ default 0.5", inrFull(def05), inrFull(approveAll - def05), inr(def05 / nTest * 1e6)],
      ["Model @ cost threshold", inrFull(costThr), inrFull(approveAll - costThr), inr(costThr / nTest * 1e6)],
      ["Three-band rule", inrFull(three), inrFull(approveAll - three), `${inr(S.projection.per_1m_three_band)} (CI ${inr(S.projection.per_1m_ci[0])}–${inr(S.projection.per_1m_ci[1])})`],
    ], [2600, 1900, 2200, 2614], { right: [1, 2, 3], hl: [3] }),
    Callout(`**Projected (scenario estimate, not observed):** if future traffic looks like the test period and the assumed costs hold, the three-band rule would save about **${inr(S.projection.per_1m_saving)} per million payments** compared with approving everything. The observed columns use held-out counts × assumed costs; the projection additionally assumes the test fraud rate (0.13%) persists.`),
    H1("9. UPI 2024 (negative control)"),
    P(`The same exact search on the UPI validation scores produces thresholds, but on test the resulting three-band rule costs ${inrFull(upiThree)} against ${inrFull(upiAll)} for approving everything. It challenges 212 genuine customers and blocks 5, and catches no fraud. At the default 0.5 threshold, the validation "winner" would cost ${inr(U.policies_test["Model @ default 0.5 (block/approve)"].test_cost)}, because it blocks about 24,700 genuine customers. **For UPI the cost-optimal policy is to approve everything until the data carries signal.** The web app enforces this in UPI mode.`),
    H1("10. How to use official costs on the day"),
    Num("**Notebook:** edit `COST = L.CostConfig(X=…, Y=…, Z=…, otp_stop_rate=…)` in the setup cell and run all. Cached models are reused, so this takes about 3 minutes."),
    Num("**Web app:** open **Policy**, enter the four values, and press **Re-optimise thresholds**. The new thresholds apply to Score and Batch immediately, and the test-period table updates."),
  ] }]);
  await save(doc, path.join(OUT, "FraudGuard_Cost_Analysis_and_Risk_Rule.docx"));
}

/* ============================================================ 3. BUSINESS REPORT */
async function businessReport() {
  const ks = Object.entries(S.eda.ks_top).slice(0, 4).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(", ");
  const night = S.eda.night_01_05, az = S.eda.amount_zero.find((r) => r.amount_zero === 1), low = S.eda.amount_bands[0], hi = S.eda.amount_bands.slice(-2);
  const doc = makeDoc("Business Report", "Real-time payment fraud decisions: approve, step-up OTP or block, set by rupee cost · KLU + Path-3 · ML-02 · 31 October 2026", [{ children: [
    H1("1. Executive summary"),
    Callout(`**A cost-tuned approve / step-up OTP / block rule cuts expected fraud cost by 78% on unseen data and touches only 0.33% of customers.** On the held-out test period it blocks 60 payments, 54 of them fraud. It sends ${B["STEP-UP OTP"].transactions} to OTP, approves the remaining ${B["APPROVE"].transactions.toLocaleString("en-IN")}, and lowers expected cost from ${inrFull(approveAll)} (no model) to ${inrFull(three)}. The organiser's UPI file carries no learnable fraud signal, so the system is proven on the ULB credit-card benchmark, with UPI run through the same pipeline as a negative control.`),
    Bul("**The trap:** a model that approves everything is 99.87% accurate and catches no fraud. We judge every model by recall, precision, PR-AUC and rupee cost."),
    Bul(`**The model:** CatBoost with class weights, chosen from 22 supervised candidates by a rule fixed before testing. Test PR-AUC is 0.80 (95% CI 0.71–0.88), against 0.0013 for random ranking.`),
    Bul(`**Unknown fraud:** COPOD, trained with no labels, ranks ${pct(bestA.recall_top_1pct, 0)} of test fraud in its top 1% of scores. It is a ready-made watch-list for fraud patterns that have no labels yet.`),
    Bul("**Three actions:** (1) replace the 0.5 threshold with the cost-based three-band rule; (2) fix UPI data capture before modelling UPI; (3) target card testing and night-time bursts operationally."),
    H1("2. The datasets"),
    Tbl(["", "ULB Credit Card Fraud (primary)", "UPI Transactions 2024 (control)"], [
      ["Source", "Worldline and the Machine Learning Group, Université Libre de Bruxelles; published on Kaggle (mlg-ulb/creditcardfraud)", "Supplied by the organisers; simulated UPI-style data"],
      ["Scope", "Card payments by European cardholders, September 2013, two days", "250,000 UPI payments, 1 Jan – 30 Dec 2024"],
      ["Size and fraud", "284,807 rows, 31 columns; 492 frauds (0.172%)", "250,000 rows, 17 columns; 480 frauds (0.192%)"],
      ["Columns", "Time (seconds since first txn), V1–V28 (PCA components), Amount (currency not stated), Class", "Transaction type, merchant category, amount (₹), sender/receiver age group, sender state, sender/receiver bank, device, network, status, hour, weekday, weekend, fraud flag"],
      ["Confidentiality", "Original features and cardholder identities withheld; only PCA components released", "No names, account numbers, VPAs or phone numbers; banks and states only as categories"],
    ], [1500, 3907, 3907], { boldFirst: true, size: 16 }),
    H1("3. Data quality and cleaning"),
    Tbl(["Check", "Credit card: finding → treatment", "UPI: finding → treatment"], [
      ["Missing values", "None", "None"],
      ["Duplicates", "1,081 exact duplicates (19 fraud) → dropped, keep first", "None"],
      ["Impossible amounts", "No negatives; 1,808 zero amounts (25 fraud) → kept, flagged as card-testing signal", "None (₹10 to ₹42,099); log-transformed"],
      ["Timestamps", "Seconds since start → hour of day derived; raw Time dropped", "All valid; hour and weekday consistent"],
      ["Inconsistent categories", "None (PCA features)", "All 112,445 P2P rows carry a merchant category; amounts identical across types → flagged as simulated"],
      ["Leakage columns", "Only Class excluded", "transaction_status (known only after the decision) and transaction id excluded"],
    ], [1700, 3807, 3807], { boldFirst: true, size: 16 }),
    P("After cleaning, both files are split **by time** (first 60% train, next 20% validation, last 20% test) before any resampling or tuning."),
    H1("4. The accuracy trap"),
    Tbl(["Test period", "Accuracy", "Recall", "Precision", "Missed", "Blocked genuine", "Expected cost"], [
      ["Do-nothing (approve all)", pct(dn.accuracy, 2), "0%", "—", dn.fn, 0, inrFull(dn.cost_inr)],
      ["Final model, default 0.5", pct(md.accuracy, 2), pct(md.recall), pct(md.precision), md.fn, md.fp, inrFull(md.cost_inr)],
      ["Final model, cost threshold", pct(mc.accuracy, 2), pct(mc.recall), pct(mc.precision), mc.fn, mc.fp, inrFull(mc.cost_inr)],
    ], [2700, 1000, 900, 1000, 900, 1200, 1614], { right: [1, 2, 3, 4, 5, 6] }),
    P("Accuracy separates useless from useful by 0.08 points, and ranks the cheaper threshold lower. It is reported only to make this point."),
    H1("5. Where fraud lives (credit card)"),
    Img(F("creditcard", "02_fraud_by_amount_hour.png"), 6.4),
    Caption("Figure 1. Fraud rate by amount octile and by hour, with Wilson 95% CIs (orange = whole interval above the overall rate)."),
    Bul(`**Tiny amounts:** payments of 1.51 or less (currency not stated) have a ${low.fraud_rate_pct.toFixed(2)}% fraud rate, 3× the 0.17% average, and hold ${low.frauds} of ${S.n_fraud} frauds. Zero-amount authorisations run at ${az.fraud_rate_pct.toFixed(2)}% (8×). This fits card testing.`),
    Bul(`**Large amounts:** above 77.51 the rate rises again (${hi.map((h) => h.fraud_rate_pct.toFixed(2) + "%").join(" and ")}).`),
    Bul(`**Night window 01–05h:** ${night.share_txns.toFixed(1)}% of payments but ${night.share_fraud.toFixed(1)}% of fraud (rate ${night.rate.toFixed(2)}% vs ${night.rate_day.toFixed(2)}% otherwise).`),
    Bul("**Bursts:** 43 frauds arrived in a single hour (Figure 2), so fraud comes in campaigns."),
    Bul(`**Behaviour signature:** a few anonymised components separate fraud strongly (Kolmogorov–Smirnov: ${ks}).`),
    Img(F("creditcard", "04_fraud_over_time.png"), 6.4),
    Caption("Figure 2. Payments and frauds per hour over the two days; shaded = 01–05h."),
    P("**UPI:** no transaction type, merchant, amount band, hour, device or age group differs from the overall rate once multiple testing is accounted for (Figure 3). These are associations, not causes."),
    Img(F("upi", "02_upi_fraud_rates.png"), 6.2),
    Caption("Figure 3. UPI fraud rate by type, merchant, amount, hour, sender age and device, with 95% CIs."),
    H1("6. What we built"),
    Bul("**Supervised:** two rule baselines and 22 models (logistic regression, random forest, XGBoost, LightGBM × none / class weights / SMOTE inside the pipeline; CatBoost; Balanced RF; EasyEnsemble; Optuna-tuned LightGBM; focal-loss MLP; FT-Transformer; hybrid; soft-vote ensemble). Detail in the Model Comparison document."),
    Bul("**Label-free:** Isolation Forest, ECOD, COPOD, LOF, autoencoder and DBSCAN, trained without labels and evaluated afterwards."),
    Bul("**Decision layer:** isotonic calibration, then two thresholds from an exact search minimising expected rupee cost on validation, frozen and scored once on test. Detail in the Cost Analysis document."),
    Bul("**Explanation:** exact TreeSHAP; in the web app it runs in the browser for every payment."),
    Img(F("creditcard", "07_pr_curves.png"), 6.0),
    Caption("Figure 4. Precision–recall on the test period."),
    H1("7. The decision rule and its cost"),
    Tbl(["Band", "Rule (model score)", "Payments", "Fraud", "Genuine customers", "Expected cost"], [
      ["APPROVE", `< ${fmtT(S.bands.T_otp)}`, B.APPROVE.transactions.toLocaleString("en-IN"), `${B.APPROVE.frauds} missed`, "pass", inrFull(B.APPROVE.total_cost)],
      ["STEP-UP OTP", `${fmtT(S.bands.T_otp)} – ${fmtT(S.bands.T_block)}`, B["STEP-UP OTP"].transactions, `${B["STEP-UP OTP"].frauds} challenged`, `${B["STEP-UP OTP"].genuine} challenged`, inrFull(B["STEP-UP OTP"].total_cost)],
      ["BLOCK", `≥ ${fmtT(S.bands.T_block)}`, B.BLOCK.transactions, `${B.BLOCK.frauds} stopped`, `${B.BLOCK.genuine} blocked`, inrFull(B.BLOCK.total_cost)],
    ], [1400, 1900, 1200, 1400, 1700, 1714], { right: [2, 5], boldFirst: true }),
    P(`Total ${inrFull(three)} against ${inrFull(def05)} at the default 0.5 threshold and ${inrFull(approveAll)} with no model (observed counts × assumed costs). **Scenario estimate:** about ${inr(S.projection.per_1m_saving)} saved per million payments.`),
    Img(F("creditcard", "11b_cost_waterfall.png"), 6.2),
    Caption("Figure 5. From approve-everything to the three-band rule on test."),
    H1("8. Why the model flags a payment"),
    Img(F("creditcard", "12_shap_beeswarm.png"), 4.6),
    Caption("Figure 6. SHAP summary for the final model (log-odds of fraud). Low V14 and high V4 push towards fraud; hour of day ranks 6th."),
    P("The model relies on the same components the exploration found (V14, V4, V12, V10, hour of day), so it has learned real behaviour, not an artefact. SHAP shows how the model uses inputs, not what causes fraud."),
    H1("9. Three findings → three actions"),
    Tbl(["Priority", "Finding (observed)", "Evidence", "Action", "Projected impact"], [
      ["1 · this quarter", "A cost-tuned three-band rule beats the 0.5 threshold", `${inrFull(three)} vs ${inrFull(def05)} on test; blocks are 90% fraud; ${pct(caughtOrChallenged / fTest)} of fraud stopped or challenged`, "Shadow-run CatBoost for 2–4 weeks, then go live with the two thresholds; re-fit them monthly from the cost curve", `≈ ${inr(S.projection.per_1m_saving)} per million payments (scenario)`],
      ["2 · before any UPI model", "The UPI data cannot support a fraud model", "22 models at chance (ROC-AUC 0.47–0.54); permutation p = 0.78; no account or device IDs; post-outcome status field", "Log sender, receiver and device IDs and dated confirmed-fraud labels; keep post-outcome fields out; run COPOD as a watch-list; hold out a small random slice from blocking", "Not estimable until labels carry signal; avoids ₹4,680+ of pure friction per 50,000 payments"],
      ["3 · next sprint", "Fraud clusters in card testing, nights and bursts", "≤ 1.51 amounts: 3× rate, 37% of fraud; zero amounts 8×; 01–05h: 23% of fraud in 5.7% of traffic; 43 frauds in one hour", "Velocity limit on micro and zero-amount authorisations per card/device; burst alerts at 3× trailing volume of OTP+block; night staffing; A/B-test first", "Faster response to new campaigns; not separately quantified (associations)"],
    ], [1150, 1700, 2350, 2550, 1564], { size: 15, boldFirst: true }),
    H1("10. Special points"),
    Bul("**Confidentiality:** the card data's real features are hidden behind PCA by the data owner, so reasons are given as V-components. The UPI file holds no personal identifiers. The web app scores in the browser; saved records sit in each user's own Supabase rows, protected by row-level security."),
    Bul("**Compliance for a real rollout:** customer consent, purpose limitation and retention under India's Digital Personal Data Protection Act 2023; RBI fraud-reporting and customer-liability rules for blocked or disputed payments; an appeal path for blocked customers."),
    Bul("**Currency:** the credit-card source never states the currency of Amount, so amounts are shown in source units and rupee costs are applied per event."),
    Bul("**Official costs:** none were published. All rupee figures use stated assumptions and can be recomputed instantly in the notebook or the app's Policy page."),
    Bul("**Feedback loop:** blocked fraud is never confirmed, which biases future labels. A small random holdout of unblocked traffic keeps them honest."),
    H1("11. Limitations"),
    Bul("About two days of 2013 European card data; 57 validation and 74 test frauds, so costs carry wide CIs (reported throughout)."),
    Bul("Anonymised features prevent business-readable reason codes and transaction-type analysis on card data."),
    Bul("Costs and the 70% OTP stop rate are assumptions; savings are scenario estimates."),
    Bul("UPI patterns cannot be studied with this file; card patterns may not transfer to UPI."),
    H1("12. Deliverables"),
    Bul("Notebook (both datasets, seed 42, reproducible from scratch), source library, README, architecture decision records."),
    Bul("Web app (Netlify + Supabase): Credit Card and UPI modes, live scoring with SHAP, batch CSV scoring, policy lab, evidence pages, sign-in with saved scorings and audit log."),
    Bul("Model Comparison and Cost Analysis & Risk-Score Rule documents; presentation deck."),
  ] }]);
  await save(doc, path.join(OUT, "FraudGuard_Business_Report.docx"));
}

(async () => { await modelComparison(); await costAnalysis(); await businessReport(); })();
