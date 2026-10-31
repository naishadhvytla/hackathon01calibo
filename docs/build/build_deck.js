// FraudGuard presentation — structured pptxgenjs deck; numbers are read from the pipeline outputs.
const fs = require("fs"), path = require("path");
const pptxgen = require("pptxgenjs");
const SK = "/root/.claude/skills/synced/21f66b5d-6ebb-47b1-8886-6ec218504b4e_80327f49-d0d7-4365-a95e-465d7a6ce699/pptx";
const { applyTheme } = require(SK + "/scripts/apply_theme.js");
const ROOT = path.resolve(__dirname, "../..");
const J = (f) => JSON.parse(fs.readFileSync(f, "utf8").replace(/\bNaN\b/g, "null").replace(/-?Infinity\b/g, "null"));
const S = J(path.join(ROOT, "outputs/summary_creditcard.json")), U = J(path.join(ROOT, "outputs/summary_upi.json"));
const FIG = (d, f) => path.join(ROOT, "outputs/figures", d, f), SHOT = (f) => path.join(__dirname, "shots", f);
const pngSize = (f) => { const b = fs.readFileSync(f); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; };

const THEME = { name: "FraudGuard", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "14284B", lt1: "FFFFFF", dk2: "2E3B52", lt2: "F2F4F7", accent1: "D8A21B", accent2: "1E7A52", accent3: "9A6B00", accent4: "B03A22", accent5: "5B6B85", accent6: "8FA3C2", hlink: "2F6FEB", folHlink: "6B4FBB" } };
const HEX = { navy: "14284B", ink: "2E3B52", muted: "5B6B85", paper: "F2F4F7", line: "D8DEE7", gold: "D8A21B", green: "1E7A52", amber: "9A6B00", red: "B03A22", ice: "C9D5E8", white: "FFFFFF" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; pres.title = "FraudGuard — payment fraud decisions"; pres.author = "FraudGuard team"; pres.company = "KLU + Path-3";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333, H = 7.5, MX = 0.6;

// ---------- layouts
pres.defineSlideMaster({ title: "DARK", background: { color: HEX.navy }, objects: [
  { placeholder: { options: { name: "title", type: "title", x: MX, y: 2.2, w: 9.6, h: 1.9, fontFace: "Cambria", fontSize: 44, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
  { placeholder: { options: { name: "body", type: "body", x: MX, y: 4.25, w: 9.6, h: 1.2, fontFace: "Calibri", fontSize: 18, color: C.accent6, valign: "top", margin: 0 }, text: "" } } ] });
pres.defineSlideMaster({ title: "SECTION", background: { color: HEX.navy }, slideNumber: { x: W - 1.0, y: H - 0.5, w: 0.5, h: 0.3, fontSize: 10, color: C.accent6 }, objects: [
  { placeholder: { options: { name: "title", type: "title", x: MX, y: 2.6, w: 10, h: 1.2, fontFace: "Cambria", fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
  { placeholder: { options: { name: "body", type: "body", x: MX, y: 3.95, w: 10, h: 0.9, fontFace: "Calibri", fontSize: 18, color: C.accent6, valign: "top", margin: 0 }, text: "" } } ] });
pres.defineSlideMaster({ title: "CONTENT", background: { color: HEX.white }, slideNumber: { x: W - 1.0, y: H - 0.45, w: 0.5, h: 0.3, fontSize: 10, color: C.accent5 },
  objects: [ { text: { text: "FraudGuard · KLU + Path-3 · ML-02", options: { x: MX, y: H - 0.45, w: 5, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 } } },
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.4, w: W - 2 * MX, h: 0.95, fontFace: "Cambria", fontSize: 30, bold: true, color: C.text1, valign: "top", align: "left", margin: 0 }, text: "" } } ] });

// ---------- helpers
let anim = 0; const A = () => "anim_" + String(++anim).padStart(3, "0");   // objectNames the post-processor animates, in order
const T = (s, text, o) => s.addText(text, { isTextBox: true, margin: 0, fontFace: "Calibri", fontSize: 16, color: C.text2, valign: "top", ...o });
function card(s, x, y, w, h, fill = HEX.paper, name) { s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill }, objectName: name }); }
function stat(s, x, y, w, big, small, color = C.text1, animate = true) {
  T(s, [{ text: big, options: { fontFace: "Cambria", fontSize: 40, bold: true, color, breakLine: true } }, { text: small, options: { fontSize: 14, color: C.accent5 } }], { x, y, w, h: 1.4, objectName: animate ? A() : undefined });
}
function img(s, file, x, y, w, h, o = {}) {
  const { w: pw, h: ph } = pngSize(file); let iw = w, ih = w * ph / pw; if (ih > h) { ih = h; iw = h * pw / ph; }
  s.addImage({ path: file, x: x + (o.align === "left" ? 0 : (w - iw) / 2), y, w: iw, h: ih, altText: o.alt || path.basename(file), objectName: o.name });
  return { iw, ih };
}
const inr = (v) => { v = Number(v); return Math.abs(v) >= 1e5 ? "₹" + (v / 1e5).toFixed(2) + " L" : "₹" + Math.round(v).toLocaleString("en-IN"); };
const pct = (v, d = 1) => (100 * v).toFixed(d) + "%";
const chartBase = (title) => ({ showTitle: true, title, titleFontFace: "+mn-lt", titleFontSize: 14, titleColor: HEX.navy, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  catAxisLabelColor: HEX.ink, valAxisLabelColor: HEX.muted, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, valGridLine: { color: HEX.line, size: 0.5 }, catGridLine: { style: "none" },
  dataLabelFontFace: "+mn-lt", dataLabelFontSize: 12, dataLabelColor: HEX.ink, legendFontFace: "+mn-lt", legendFontSize: 12, legendColor: HEX.ink });
const sec = (title) => pres.addSection({ title });
const content = (title, section) => { const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section }); s.addText(title, { placeholder: "title" }); return s; };

const B = Object.fromEntries(S.bands.test.map((b) => [b.band, b])), pol = S.policies_test;
const approveAll = pol["Approve everything (no model)"].test_cost, three = pol["3-band rule (approve / OTP / block)"].test_cost, d05 = pol["Model @ default 0.5 (block/approve)"].test_cost, cthr = pol["Model @ cost threshold (block/approve)"].test_cost;
const dn = S.do_nothing_test, md = S.threshold.test_default, mc = S.threshold.test_cost, FINAL = S.final_model;
const cmp = S.comparison, fm = cmp.find((r) => r.model === FINAL);
const nT = S.n_test, fT = S.test_frauds, caught = B.BLOCK.frauds + B["STEP-UP OTP"].frauds;
const fmtT = (t) => (t < 0.01 ? t.toFixed(4) : t.toFixed(3));

/* ===================== 1 · Title */
sec("Opening");
let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Opening" });
s.addText("FraudGuard", { placeholder: "title" });
s.addText("Every payment gets a fraud score; the score becomes approve, step-up OTP or block, with the two cut-offs set by rupee cost", { placeholder: "body" });
T(s, "KLU + Path-3 · ML-02 · Real-time UPI / digital payment fraud detection", { x: MX, y: 0.6, w: 10, h: 0.4, fontSize: 14, color: C.accent1 });
[[pct(1 - three / approveAll, 0), "lower expected fraud cost on unseen data"], [pct((B["STEP-UP OTP"].transactions + B.BLOCK.transactions) / nT, 2), "of customers see any friction"], [`${caught} / ${fT}`, "test frauds stopped or challenged"]].forEach(([b, t], i) =>
  T(s, [{ text: b, options: { fontFace: "Cambria", fontSize: 34, bold: true, color: C.background1, breakLine: true } }, { text: t, options: { fontSize: 13, color: C.accent6 } }], { x: MX + i * 3.4, y: 5.75, w: 3.1, h: 1.1, objectName: A() }));
T(s, "[Team name] · 31 October 2026", { x: W - 4.6, y: 0.6, w: 4.0, h: 0.4, fontSize: 14, color: C.accent6, align: "right" });
s.addNotes("Open with the decision problem: every payment needs a decision in a fraction of a second. FraudGuard scores it and turns the score into approve, step-up OTP or block, with thresholds set by rupee cost. Headline: on the held-out test period, expected cost fell 78% against approving everything, and only 0.33% of customers felt any friction.");

/* ===================== 2 · The decision problem */
sec("The problem");
s = content("Each payment needs a decision in milliseconds, and both mistakes cost money", "The problem");
[["₹5,000", "Fraud approved", "Loss, chargeback, investigation", HEX.red], ["₹300", "Genuine payment blocked", "Friction, support call, churn risk", HEX.ink], ["₹15", "Step-up OTP challenge", "Mostly genuine drop-off (≈5% × ₹300); SMS < ₹1", HEX.amber]].forEach(([v, k, d, col], i) => {
  const x = MX + i * 4.1; card(s, x, 1.7, 3.8, 1.9, HEX.paper, A());
  T(s, [{ text: v, options: { fontFace: "Cambria", fontSize: 36, bold: true, color: col, breakLine: true } }, { text: k, options: { fontSize: 17, bold: true, color: C.text1, breakLine: true } }, { text: d, options: { fontSize: 14, color: C.accent5 } }], { x: x + 0.3, y: 1.9, w: 3.3, h: 1.8 });
});
card(s, MX, 4.2, W - 2 * MX, 1.25, "EAF0F8");
T(s, [{ text: "What is step-up OTP?  ", options: { bold: true, color: C.text1 } }, { text: "Extra proof of identity asked only when a payment looks risky. The real owner types the one-time password and the payment goes through; a fraudster with stolen details usually cannot, because they don't have the phone. It does not stop scams where the victim is talked into typing the OTP, so we assume it stops 70% of fraud." }], { x: MX + 0.3, y: 4.35, w: W - 2 * MX - 0.6, h: 1.0, fontSize: 15 });
T(s, [{ text: "Expected cost = ", options: { bold: true } }, { text: "missed fraud × ₹X  +  genuine blocked × ₹Y  +  OTP challenges × ₹Z  +  fraud that passes OTP × ₹X" }], { x: MX, y: 5.75, w: W - 2 * MX, h: 0.4, fontSize: 16, color: C.text1 });
T(s, "No official ₹ figures were published, so these are stated assumptions. The notebook and the app's Policy page recompute every threshold for any values.", { x: MX, y: 6.25, w: W - 2 * MX, h: 0.4, fontSize: 13, color: C.accent5 });
s.addNotes("Frame fraud loss against customer friction. We put a rupee price on each outcome: ₹5,000 for missed fraud, ₹300 for a wrongly blocked customer, ₹15 for an OTP challenge. The ₹15 is mostly genuine customers who abandon the payment (about 5% × ₹300), not the SMS. No official costs were given, so these are stated assumptions; slide 16 shows the decision barely moves when they change.");

/* ===================== 3 · Accuracy trap */
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "The problem" });
s.background = { color: HEX.navy };
s.addText([{ text: "The accuracy trap: 99.87% accurate, zero fraud caught", options: { color: HEX.white } }], { placeholder: "title" });
s.addChart(pres.charts.BAR, [
  { name: "Accuracy", labels: ["Do-nothing", "Model @ 0.5", "Model @ cost threshold"], values: [dn.accuracy, md.accuracy, mc.accuracy] },
  { name: "Recall (fraud caught)", labels: ["Do-nothing", "Model @ 0.5", "Model @ cost threshold"], values: [0, md.recall, mc.recall] },
  { name: "Precision", labels: ["Do-nothing", "Model @ 0.5", "Model @ cost threshold"], values: [0, md.precision, mc.precision] },
], { x: MX, y: 1.5, w: 7.4, h: 5.2, barDir: "col", barGrouping: "clustered", chartColors: [HEX.muted, HEX.gold, HEX.ice], ...chartBase(""), showTitle: false,
  catAxisLabelColor: HEX.white, valAxisLabelColor: HEX.ice, legendColor: HEX.white, dataLabelColor: HEX.white, showLegend: true, legendPos: "b", showValue: true, dataLabelFormatCode: "0.00%", valAxisLabelFormatCode: "0%", valAxisMinVal: 0, valAxisMaxVal: 1.1, valGridLine: { color: "2A3F66", size: 0.5 }, objectName: "chart_trap" });
[[`${pct(dn.accuracy, 2)}`, "accuracy of a model that approves everything"], ["0 of 74", "frauds it catches; ₹3.70 L lost"], [`₹${(md.cost_inr - mc.cost_inr).toLocaleString("en-IN")}`, "saved by the cost threshold, which is less accurate"]].forEach(([b, t], i) =>
  T(s, [{ text: b, options: { fontFace: "Cambria", fontSize: 32, bold: true, color: i === 1 ? "F08A70" : HEX.white, breakLine: true } }, { text: t, options: { fontSize: 14, color: HEX.ice } }], { x: 8.5, y: 1.7 + i * 1.6, w: 4.2, h: 1.4, objectName: A() }));
T(s, "So we select on PR-AUC and rupee cost; accuracy appears only to make this point.", { x: 8.5, y: 6.35, w: 4.2, h: 0.6, fontSize: 13, color: HEX.ice });
s.addNotes("The wow-factor slide. Fraud is 0.13% of the test period, so labelling everything genuine is 99.87% accurate and catches nothing. Accuracy can't even rank our own two settings: the cost threshold is less accurate than 0.5 but ₹6,100 cheaper. Hence recall, precision, PR-AUC and rupees.");

/* ===================== 4 · Section: data */
sec("Data");
s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Data" });
s.addText("Can the data be trusted?", { placeholder: "title" });
s.addText("Two datasets, one pipeline: one carries signal, one does not", { placeholder: "body" });

/* ===================== 5 · Datasets */
s = content("Two datasets, run through the identical pipeline", "Data");
const dsCard = (x, title, sub, rows, name) => { card(s, x, 1.55, 5.9, 4.6, HEX.paper, name);
  T(s, [{ text: title, options: { fontFace: "Cambria", fontSize: 22, bold: true, color: C.text1, breakLine: true } }, { text: sub, options: { fontSize: 13, color: C.accent5 } }], { x: x + 0.3, y: 1.75, w: 5.3, h: 0.9 });
  T(s, rows.map(([k, v], i) => ({ text: "", options: {} }) && [{ text: k + "  ", options: { bold: true, color: C.text1 } }, { text: v, options: { breakLine: i < rows.length - 1 } }]).flat(), { x: x + 0.3, y: 2.75, w: 5.3, h: 3.3, fontSize: 15, paraSpaceAfter: 8 }); };
dsCard(MX, "ULB Credit Card Fraud", "Worldline + Machine Learning Group, ULB · Kaggle (mlg-ulb/creditcardfraud)", [
  ["Scope", "European cardholders, September 2013, 2 days"], ["Size", "284,807 payments · 492 frauds (0.172%)"], ["Columns", "Time, Amount (currency not stated), V1–V28, Class"],
  ["Confidential", "V1–V28 are PCA components; original features withheld"], ["Cleaned", "1,081 duplicates dropped → 283,726 rows, 473 frauds"], ["Role", "Primary: model, thresholds, actions"]], A());
dsCard(MX + 6.2, "UPI Transactions 2024", "Organiser-supplied, simulated UPI-style file", [
  ["Scope", "1 Jan – 30 Dec 2024"], ["Size", "250,000 payments · 480 frauds (0.192%)"], ["Columns", "Type, merchant, amount ₹, ages, state, banks, device, network, status, hour, weekday"],
  ["Missing", "No sender, receiver or device IDs"], ["Leakage", "transaction_status is known only after the decision → excluded"], ["Role", "Negative control: same code, tests for signal"]], A());
s.addNotes("The brief told us to inspect the UPI file before choosing it, so we ran both datasets through the same code. Credit card: the Kaggle ULB benchmark with anonymised PCA features. UPI: the organiser's simulated file, which turns out to have no learnable signal.");

/* ===================== 6 · Cleaning & leakage */
s = content("Data quality audit: what we kept, dropped and excluded", "Data");
const hdr = (t) => ({ text: t, options: { bold: true, color: HEX.white, fill: { color: HEX.navy } } });
s.addTable([
  [hdr("Check"), hdr("Credit card"), hdr("UPI 2024")],
  ["Missing values", "None", "None"],
  ["Duplicates", "1,081 exact duplicates (19 fraud) → dropped", "None"],
  ["Impossible amounts", "None negative; 1,808 zero amounts (25 fraud) → kept, flagged (card testing)", "None; ₹10–₹42,099 → log-transformed"],
  ["Timestamps", "Seconds since start → hour of day; raw Time dropped", "All valid; hour and weekday consistent"],
  ["Inconsistent categories", "None (PCA)", "All 112,445 P2P rows have a merchant category; amounts identical across types"],
  ["Leakage columns", "Only Class excluded", "transaction_status (post-decision), transaction id excluded"],
  ["Split", "Time-ordered 60 / 20 / 20, before any resampling", "Same"],
], { x: MX, y: 1.55, w: W - 2 * MX, colW: [2.4, 5.2, 4.53], fontFace: "Calibri", fontSize: 14, color: HEX.ink, border: { type: "solid", pt: 0.75, color: HEX.line }, valign: "middle", rowH: 0.56, objectName: A() });
s.addNotes("Every check the brief lists, for both datasets, with the treatment. Two things to stress: zero amounts are kept because they're a card-testing signal, not an error; and UPI's status column is excluded because a blocked fraud would show up as FAILED, so it leaks the outcome.");

/* ===================== 7 · UPI has no signal */
s = content("The UPI fraud label carries no learnable signal", "Data");
img(s, FIG("upi", "07_pr_curves.png"), MX, 1.5, 7.4, 4.8, { name: A(), alt: "UPI precision-recall curves all on the random line" });
[["13", "χ² tests; none survives multiple-testing correction"], ["0.47–0.54", "test ROC-AUC across all 22 models"], ["p = 0.78", "the validation 'winner' on test (p = 0.016 on validation)"], ["Approve all", "is the cost-optimal UPI policy: ₹4.80 L vs ₹4.85 L for any rule"]].forEach(([b, t], i) =>
  T(s, [{ text: b, options: { fontFace: "Cambria", fontSize: 26, bold: true, color: i === 3 ? HEX.green : C.text1, breakLine: true } }, { text: t, options: { fontSize: 14, color: C.accent5 } }], { x: 8.4, y: 1.55 + i * 1.25, w: 4.3, h: 1.15, objectName: A() }));
s.addNotes("Every test came back at chance. The interesting one: the model that won on validation had p = 0.016, but on test p = 0.78. That's what picking the best of 22 models on noise looks like. So no UPI model is deployable; our recommendation 2 is about fixing UPI data capture.");

/* ===================== 8 · Where fraud lives */
s = content("On card data, fraud clusters in micro-amounts, the night and bursts", "Data");
const ab0 = S.eda.amount_bands[0], az = S.eda.amount_zero.find((r) => r.amount_zero === 1), nt = S.eda.night_01_05;
[[`${(ab0.fraud_rate_pct / (100 * S.n_fraud / S.n)).toFixed(0)}× · ${(az.fraud_rate_pct / (100 * S.n_fraud / S.n)).toFixed(0)}×`, "fraud rate at amounts ≤ 1.51 and at zero (card testing)"], [`${nt.share_fraud.toFixed(0)}% in ${nt.share_txns.toFixed(1)}%`, "of fraud in the 01–05h window, which carries 5.7% of traffic"], ["43 in 1 hour", "frauds arriving together: campaigns, not noise"]].forEach(([b, t], i) => {
  card(s, MX + i * 4.1, 1.5, 3.8, 1.35, HEX.paper, A()); T(s, [{ text: b, options: { fontFace: "Cambria", fontSize: 26, bold: true, color: HEX.red, breakLine: true } }, { text: t, options: { fontSize: 13, color: C.accent5 } }], { x: MX + i * 4.1 + 0.25, y: 1.62, w: 3.35, h: 1.15 }); });
img(s, FIG("creditcard", "02_fraud_by_amount_hour.png"), MX, 3.1, W - 2 * MX, 3.75, { name: A(), alt: "Fraud rate by amount band and hour with confidence intervals" });
s.addNotes("Three patterns with 95% confidence intervals: card testing at tiny and zero amounts, a night window, and bursts. These are associations, not causes; say that explicitly. They feed recommendation 3.");

/* ===================== 9 · Bursts + behaviour */
s = content("Fraud arrives in night-time bursts with a distinct behaviour signature", "Data");
img(s, FIG("creditcard", "04_fraud_over_time.png"), MX, 1.5, W - 2 * MX, 3.4, { name: A() });
img(s, FIG("creditcard", "03_behaviour_signature.png"), MX, 5.0, W - 2 * MX, 1.9, { name: A() });
s.addNotes("Top: payments per hour in grey, frauds in orange, night window shaded. Bottom: four anonymised components separate fraud almost on their own (KS statistic up to 0.84); the model uses exactly these.");

/* ===================== 10 · Section: models */
sec("Models");
s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Models" });
s.addText("How the system decides", { placeholder: "title" });
s.addText("22 supervised models, 6 label-free detectors, one cost-set rule", { placeholder: "body" });

/* ===================== 11 · Pipeline */
s = content("One pipeline: labelled and label-free scores feed a cost-set rule", "Models");
const boxes = [["Audit & split", "dedupe · leakage out · time split 60/20/20"], ["Supervised model", "CatBoost + class weights"], ["Calibrate", "isotonic → risk 0–100"], ["₹ threshold search", "2 cut-offs on validation"], ["Decide", "approve · OTP · block"]];
boxes.forEach(([t, d], i) => { const x = MX + i * 2.5; card(s, x, 2.2, 2.15, 1.75, i === 4 ? "E7F2EC" : HEX.paper, A());
  T(s, [{ text: t, options: { bold: true, fontSize: 16, color: C.text1, breakLine: true } }, { text: d, options: { fontSize: 13, color: C.accent5 } }], { x: x + 0.18, y: 2.35, w: 1.8, h: 1.5 });
  if (i < 4) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 2.17, y: 2.92, w: 0.3, h: 0.3, fill: { color: HEX.muted }, line: { color: HEX.muted } }); });
card(s, MX + 2.5, 4.35, 2.15, 1.3, "FBF1D6", A());
T(s, [{ text: "Label-free watch-list", options: { bold: true, fontSize: 15, color: C.text1, breakLine: true } }, { text: "COPOD, no labels", options: { fontSize: 13, color: C.accent5 } }], { x: MX + 2.68, y: 4.5, w: 1.8, h: 1.0 });
s.addShape(pres.shapes.LINE, { x: MX + 3.575, y: 3.95, w: 0, h: 0.4, line: { color: HEX.muted, width: 1.25, dashType: "dash" } });
T(s, [{ text: "Guard-rails enforced in code: ", options: { bold: true, color: C.text1 } }, { text: "SMOTE only inside the training pipeline · Optuna tunes on time-series folds inside train · deep-model early stopping on the last 15% of train · model choice, calibration and thresholds on validation · test scored once · seed 42" }], { x: MX, y: 5.95, w: W - 2 * MX, h: 0.8, fontSize: 14 });
s.addNotes("Walk left to right. Judges look for leakage, so stress the guard-rails: no resampling before the split, nothing tuned on test, a time-aware split. The label-free detector runs alongside as a watch-list.");

/* ===================== 12 · 22 models */
s = content(`22 models compared; ${FINAL.split(" |")[0]} wins on validation PR-AUC and cost`, "Models");
img(s, FIG("creditcard", "05b_all_models_pr_auc_ci.png"), MX, 1.45, 7.6, 5.5, { name: A(), align: "left" });
T(s, [{ text: "Selection rule, fixed before test", options: { bold: true, fontSize: 17, color: C.text1, breakLine: true } },
  { text: "Within 0.01 of best validation PR-AUC (6 models)", options: { bullet: { type: "number" }, breakLine: true } },
  { text: `Lowest validation ₹ cost: ${FINAL.split(" |")[0]} ₹66,300`, options: { bullet: { type: "number" }, breakLine: true } },
  { text: "Single model over ensemble (latency, exact SHAP)", options: { bullet: { type: "number" }, breakLine: true } },
  { text: "Permutation test p < 0.002", options: { bullet: { type: "number" } } }], { x: 8.5, y: 1.55, w: 4.2, h: 2.6, fontSize: 14, paraSpaceAfter: 6, objectName: A() });
[["0.80", `test PR-AUC (95% CI ${S.threshold.ci_pr_auc[0].toFixed(2)}–${S.threshold.ci_pr_auc[1].toFixed(2)}) vs 0.0013 random`], ["< 1 µs", "per payment on a laptop CPU"]].forEach(([b, t], i) =>
  T(s, [{ text: b, options: { fontFace: "Cambria", fontSize: 30, bold: true, color: C.text1, breakLine: true } }, { text: t, options: { fontSize: 13, color: C.accent5 } }], { x: 8.5, y: 4.4 + i * 1.25, w: 4.2, h: 1.15, objectName: A() }));
s.addNotes("Dots are test PR-AUC with 95% bootstrap intervals; bars are validation, which drove the choice. The top models overlap on test, so the choice rests on cost and operability. The ensemble's members were fixed in advance to avoid selection bias.");

/* ===================== 13 · weights vs SMOTE */
s = content("Class weights beat SMOTE; imbalance handling matters most for linear models", "Models");
const fam = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"], reg = (f, k) => { const r = cmp.find((m) => m.family === f && m.imbalance.startsWith(k) && !m.model.includes("Optuna")); return r ? r.val_pr_auc : 0; };
s.addChart(pres.charts.BAR, ["none", "class weights", "SMOTE"].map((k) => ({ name: k === "SMOTE" ? "SMOTE (in-pipeline)" : k, labels: ["LogReg", "Random Forest", "XGBoost", "LightGBM"], values: fam.map((f) => reg(f, k)) })),
  { x: MX, y: 1.5, w: 7.6, h: 5.3, barDir: "col", barGrouping: "clustered", chartColors: [HEX.ice, HEX.navy, HEX.gold], ...chartBase("Validation PR-AUC by model family and imbalance handling"), showLegend: true, legendPos: "b", showValue: true, dataLabelFormatCode: "0.000", dataLabelFontSize: 10, valAxisMinVal: 0, valAxisMaxVal: 0.9, valAxisLabelFormatCode: "0.0", objectName: "chart_smote" });
T(s, [{ text: "LogReg: weights 0.753 · SMOTE 0.705 · none 0.608", options: { bold: true, color: C.text1, breakLine: true } }, { text: "A linear model needs imbalance handling; class weights do it best.", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } }, { text: "Trees: all three regimes within ±0.02", options: { bold: true, color: C.text1, breakLine: true } }, { text: "Well inside the ±0.08 bootstrap interval, so the difference is noise.", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } }, { text: "Decision: class weights", options: { bold: true, color: HEX.green, breakLine: true } }, { text: "No synthetic transactions, no extra training time, probabilities closer to calibrated. SMOTE always ran inside the training fold only." }],
  { x: 8.5, y: 1.6, w: 4.2, h: 5.0, fontSize: 15, objectName: A() });
s.addNotes("This answers 'does SMOTE beat class weights?' from the brief. Not here. For logistic regression class weights win clearly; for tree ensembles the three are within noise.");

/* ===================== 14 · research models */
s = content("We tried research-grade models; tuned trees still won", "Models");
const rm = (n) => cmp.find((r) => r.model.startsWith(n));
s.addTable([[hdr("Model"), hdr("What it adds"), hdr("Val PR-AUC"), hdr("Test PR-AUC"), hdr("Verdict")],
  ["FT-Transformer (Gorishniy et al., 2021)", "Attention over feature tokens; focal loss", rm("FT").val_pr_auc.toFixed(3), rm("FT").test_pr_auc.toFixed(3), "Below every GBDT"],
  ["Deep MLP, focal loss (Lin et al., 2017)", "Loss-level imbalance handling", rm("Deep").val_pr_auc.toFixed(3), rm("Deep").test_pr_auc.toFixed(3), "Weaker on validation"],
  ["LightGBM + Optuna (30 TPE trials)", "Tuned on time-series folds in train", rm("LightGBM | Optuna").val_pr_auc.toFixed(3), rm("LightGBM | Optuna").test_pr_auc.toFixed(3), "No gain over defaults"],
  ["Hybrid: LightGBM + anomaly scores", "Label-free scores as features", rm("Hybrid").val_pr_auc.toFixed(3), rm("Hybrid").test_pr_auc.toFixed(3), "Adds nothing"],
  ["Soft-vote ensemble (fixed members)", "CatBoost + XGBoost + LightGBM", rm("Ensemble").val_pr_auc.toFixed(3), rm("Ensemble").test_pr_auc.toFixed(3), "No gain over CatBoost"],
  [{ text: "CatBoost, class weights (selected)", options: { bold: true } }, "Ordered boosting, √-balanced weights", { text: fm.val_pr_auc.toFixed(3), options: { bold: true } }, { text: fm.test_pr_auc.toFixed(3), options: { bold: true } }, { text: "Best val PR-AUC and cost", options: { bold: true, color: HEX.green } }],
], { x: MX, y: 1.55, w: W - 2 * MX, colW: [3.6, 3.4, 1.4, 1.4, 2.33], fontFace: "Calibri", fontSize: 14, color: HEX.ink, border: { type: "solid", pt: 0.75, color: HEX.line }, valign: "middle", rowH: 0.62, objectName: A() });
T(s, "Consistent with Grinsztajn et al. (2022): tree ensembles still lead on mid-sized tabular data. The brief rewards business value, not complexity, so complexity had to earn its place.", { x: MX, y: 6.15, w: W - 2 * MX, h: 0.6, fontSize: 14, color: C.accent5 });
s.addNotes("We went beyond the brief's LR, RF, XGBoost: transformers, focal loss, Bayesian tuning, a hybrid and an ensemble. None beat CatBoost on validation, which matches the literature, so the simple model wins and we can say why.");

/* ===================== 15 · anomaly */
s = content("Without any labels, COPOD finds 59% of fraud in its top 1%", "Models");
const an = S.anomaly;
s.addChart(pres.charts.BAR, [{ name: "Fraud in top 1% of scores", labels: [...an.map((a) => a.model), `${FINAL.split(" |")[0]} (supervised)`], values: [...an.map((a) => a.recall_top_1pct), S.sup_vs_anom_top1.sup_top1 / S.sup_vs_anom_top1.fraud_total] }],
  { x: MX, y: 1.5, w: 7.6, h: 5.3, barDir: "bar", chartColors: [HEX.gold, HEX.gold, HEX.gold, HEX.gold, HEX.gold, HEX.navy], ...chartBase("Fraud in top 1% of scores (gold = no labels used)"), showLegend: false, showValue: true, dataLabelFormatCode: "0%", valAxisLabelFormatCode: "0%", valAxisMaxVal: 1, catAxisOrientation: "maxMin", objectName: "chart_anom" });
[[`${S.sup_vs_anom_top1.anom_top1} of ${S.sup_vs_anom_top1.fraud_total}`, "test frauds in COPOD's top 1%; precision there is 60× random"], [`${S.sup_vs_anom_top1.both} overlap`, "with the supervised top 1%: it mostly re-finds known fraud"], ["Watch-list", "its job: surface new fraud patterns that have no labels yet, for analysts, not for blocking"]].forEach(([b, t], i) =>
  T(s, [{ text: b, options: { fontFace: "Cambria", fontSize: 26, bold: true, color: C.text1, breakLine: true } }, { text: t, options: { fontSize: 14, color: C.accent5 } }], { x: 8.5, y: 1.6 + i * 1.6, w: 4.2, h: 1.4, objectName: A() }));
s.addNotes("This answers PS5. Detectors fit on training features only; labels used afterwards. Also tested: ECOD, Isolation Forest, autoencoder, LOF and DBSCAN, which flags 0.6% as noise and catches 6 of 12 frauds in its sample.");

/* ===================== 16 · Section: decision */
sec("Decision");
s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Decision" });
s.addText("From score to rupees", { placeholder: "title" });
s.addText("A cost-based threshold, then three bands", { placeholder: "body" });

/* ===================== 17 · cost threshold */
s = content("The cheapest cut-off is 0.119, not 0.5", "Decision");
img(s, FIG("creditcard", "09_cost_threshold_confusion.png"), MX, 1.45, W - 2 * MX, 4.2, { name: A() });
[[`${mc.tp} of ${fT}`, "frauds caught on test at 0.119"], [`${mc.fp}`, "genuine payments blocked"], [inr(mc.cost_inr), `test cost vs ${inr(md.cost_inr)} at 0.5`], [`${inr(S.threshold.ci_cost[0])}–${inr(S.threshold.ci_cost[1])}`, "bootstrap 95% CI of that cost"]].forEach(([b, t], i) =>
  T(s, [{ text: b, options: { fontFace: "Cambria", fontSize: 24, bold: true, color: C.text1, breakLine: true } }, { text: t, options: { fontSize: 13, color: C.accent5 } }], { x: MX + i * 3.05, y: 5.85, w: 2.9, h: 1.0, objectName: A() }));
s.addNotes("Left: expected cost on validation at every threshold; the minimum is 0.119, the square is the 0.5 default. Right: the test confusion matrix in rupees. We chose on validation and scored test once.");

/* ===================== 18 · sensitivity */
s = content("Wrong cost guesses barely move the decision", "Decision");
img(s, FIG("creditcard", "10_cost_sensitivity.png"), MX, 1.4, 8.4, 2.75, { name: A(), align: "left" });
img(s, FIG("creditcard", "10b_otp_sensitivity.png"), MX, 4.2, 8.4, 2.75, { name: A(), align: "left" });
T(s, [{ text: "₹X × ₹Y: 25 scenarios", options: { bold: true, color: C.text1, breakLine: true } }, { text: "Only four distinct thresholds appear; recall stays between 73% and 86%.", options: { breakLine: true } }, { text: " ", options: { breakLine: true } },
  { text: "₹Z × OTP stop rate", options: { bold: true, color: C.text1, breakLine: true } }, { text: "The block band stays at 60 payments in every case. A dearer or weaker OTP only shrinks the OTP band.", options: { breakLine: true } }, { text: " ", options: { breakLine: true } },
  { text: "Official costs on the day?", options: { bold: true, color: HEX.green, breakLine: true } }, { text: "Type them into the app's Policy page: thresholds re-optimise in under a second." }], { x: 9.2, y: 1.5, w: 3.6, h: 5.3, fontSize: 14, objectName: A() });
s.addNotes("This defends the assumed costs. Top: missed-fraud cost against block cost. Bottom: OTP cost against how much fraud OTP stops. The block decision never moves, and the app recomputes everything live for any official numbers.");

/* ===================== 19 · risk bands */
s = content("Three bands: block 60, challenge 125, approve the rest", "Decision");
const bandColor = { APPROVE: "E3F2EA", "STEP-UP OTP": "FAEFD2", BLOCK: "F7E0D9" };
s.addTable([[hdr("Band"), hdr("Rule (model score)"), hdr("Payments"), hdr("Fraud"), hdr("Genuine customers"), hdr("Expected cost")],
  ...["APPROVE", "STEP-UP OTP", "BLOCK"].map((k) => [{ text: k, options: { bold: true, fill: { color: bandColor[k] } } },
    k === "APPROVE" ? `< ${fmtT(S.bands.T_otp)}` : k === "BLOCK" ? `≥ ${fmtT(S.bands.T_block)}` : `${fmtT(S.bands.T_otp)} – ${fmtT(S.bands.T_block)}`,
    B[k].transactions.toLocaleString("en-IN"), k === "APPROVE" ? `${B[k].frauds} missed` : k === "BLOCK" ? `${B[k].frauds} stopped (${pct(B[k].frauds / fT, 0)})` : `${B[k].frauds} challenged`,
    k === "APPROVE" ? "pass" : k === "BLOCK" ? `${B[k].genuine} blocked` : `${B[k].genuine} challenged`, "₹" + Math.round(B[k].total_cost).toLocaleString("en-IN")])],
  { x: MX, y: 1.5, w: 7.7, colW: [1.5, 1.55, 1.05, 1.4, 1.25, 0.95], fontFace: "Calibri", fontSize: 13, color: HEX.ink, border: { type: "solid", pt: 0.75, color: HEX.line }, valign: "middle", rowH: 0.62, objectName: A() });
img(s, FIG("creditcard", "11_risk_bands.png"), MX, 4.25, 7.7, 2.65, { name: A(), align: "left" });
[[pct(B.BLOCK.fraud_rate_in_band_pct / 100, 0), "of blocked payments are fraud"], [pct(caught / fT, 0), "of fraud stopped or challenged"], ["0.33%", "of customers see any friction"], [`${S.bands.risk_otp.toFixed(1)} / ${S.bands.risk_block.toFixed(1)}`, "calibrated risk (0–100) where OTP / block start"]].forEach(([b, t], i) =>
  T(s, [{ text: b, options: { fontFace: "Cambria", fontSize: 28, bold: true, color: i === 0 ? HEX.red : C.text1, breakLine: true } }, { text: t, options: { fontSize: 13, color: C.accent5 } }], { x: 8.75, y: 1.55 + i * 1.32, w: 4.0, h: 1.2, objectName: A() }));
s.addNotes("This is PS7. Two thresholds found by an exact search over validation scores, minimising approve + OTP + block cost, then frozen. Read the table: 60 blocks, 54 of them fraud; 125 OTPs, 119 genuine customers challenged; 14 frauds slip through.");

/* ===================== 20 · money */
s = content(`Expected cost falls from ${inr(approveAll)} to ${inr(three)} on unseen data`, "Decision");
s.addChart(pres.charts.BAR, [{ name: "Expected cost on test", labels: ["No model (approve all)", "Model @ 0.5", "Cost threshold", "Three-band rule"], values: [approveAll, d05, cthr, three] }],
  { x: MX, y: 1.5, w: 6.2, h: 5.3, barDir: "col", chartColors: [HEX.muted, HEX.ice, HEX.ice, HEX.green], ...chartBase("Expected cost on the test period (observed counts × assumed ₹)"), showLegend: false, showValue: true, dataLabelFormatCode: "₹#,##0", valAxisLabelFormatCode: "₹#,##0", objectName: "chart_cost" });
img(s, FIG("creditcard", "11b_cost_waterfall.png"), 7.0, 1.6, 5.75, 2.6, { name: A() });
card(s, 7.0, 4.55, 5.75, 2.2, "E7F2EC", A());
T(s, [{ text: "Projected (scenario estimate)", options: { bold: true, color: HEX.green, breakLine: true } }, { text: `≈ ${inr(S.projection.per_1m_saving)} saved per million payments`, options: { fontFace: "Cambria", fontSize: 24, bold: true, color: C.text1, breakLine: true } },
  { text: "if future traffic looks like the test period and the assumed costs hold. Everything else on this slide is observed on held-out data.", options: { fontSize: 13, color: C.accent5 } }], { x: 7.25, y: 4.7, w: 5.3, h: 1.95, fontSize: 15 });
s.addNotes("Left: four policies on the same test period. Right: the bridge from approve-everything, where blocking does most of the work. The per-million figure is a projection and is labelled as one, as the brief requires.");

/* ===================== 21 · SHAP */
s = content("The model relies on the same behaviour signals the data showed", "Decision");
img(s, FIG("creditcard", "12_shap_beeswarm.png"), MX, 1.4, 6.6, 5.5, { name: A(), align: "left" });
T(s, [{ text: "Low V14 → up to +6 log-odds of fraud", options: { bold: true, color: C.text1, breakLine: true } }, { text: "the strongest single signal", options: { breakLine: true } }, { text: " ", options: { breakLine: true } },
  { text: "High V4, low V12 and V10 push towards fraud", options: { bold: true, color: C.text1, breakLine: true } }, { text: "the same components that separated fraud in the EDA", options: { breakLine: true } }, { text: " ", options: { breakLine: true } },
  { text: "Hour of day ranks 6th", options: { bold: true, color: C.text1, breakLine: true } }, { text: "matching the night window", options: { breakLine: true } }, { text: " ", options: { breakLine: true } },
  { text: "Every payment explained live", options: { bold: true, color: HEX.green, breakLine: true } }, { text: "The web app computes exact TreeSHAP in the browser. Associations, not causes." }], { x: 7.6, y: 1.55, w: 5.1, h: 5.3, fontSize: 15, objectName: A() });
s.addNotes("SHAP agrees with the EDA, so the model learned real behaviour, not an artefact. Be upfront that V1–V28 are anonymised by the bank, so we can't name them in business terms.");

/* ===================== 22 · Section: action */
sec("Action");
s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Action" });
s.addText("Three findings, three actions", { placeholder: "title" });
s.addText("Each tied to observed evidence; projections labelled", { placeholder: "body" });

/* ===================== 23 · actions */
s = content("Three actions, each tied to observed evidence", "Action");
[["1 · This quarter", "Retire 0.5; deploy the three-band rule", `${inr(three)} vs ${inr(d05)} on test; blocks are 90% fraud`, "Shadow-run 2–4 weeks, then go live; re-fit thresholds monthly from the cost curve", HEX.green],
 ["2 · Before any UPI model", "Fix UPI data, then model it", "22 models at chance; no IDs; a leakage field", "Log account and device IDs and dated labels; COPOD watch-list meanwhile; random holdout", HEX.navy],
 ["3 · Next sprint", "Target card testing and bursts", "≤ 1.51 amounts hold 37% of fraud; 01–05h = 23% of fraud in 5.7% of traffic", "Velocity limits on micro-authorisations; burst alerts; night staffing; A/B test first", HEX.amber]].forEach(([p, t, e, a, col], i) => {
  const x = MX + i * 4.1; card(s, x, 1.5, 3.8, 4.3, HEX.paper, A());
  T(s, [{ text: p, options: { bold: true, fontSize: 14, color: col, breakLine: true } }, { text: t, options: { fontFace: "Cambria", bold: true, fontSize: 20, color: C.text1, breakLine: true } }, { text: " ", options: { fontSize: 8, breakLine: true } },
    { text: "Evidence (observed)", options: { bold: true, fontSize: 13, color: C.accent5, breakLine: true } }, { text: e, options: { fontSize: 14, breakLine: true } }, { text: " ", options: { fontSize: 8, breakLine: true } },
    { text: "Action", options: { bold: true, fontSize: 13, color: C.accent5, breakLine: true } }, { text: a, options: { fontSize: 14 } }], { x: x + 0.28, y: 1.7, w: 3.25, h: 4.0 });
});
s.addNotes("Exactly three, each linked to evidence. One has the largest observed value. Two unlocks UPI at all. Three is operational and must be A/B tested, because the patterns are associations.");

/* ===================== 24 · demo */
s = content("Live demo: FraudGuard on the web, in both modes", "Action");
img(s, SHOT("app_score_cc.png"), MX, 1.45, 7.9, 4.95, { name: A(), align: "left", alt: "FraudGuard web app scoring a credit-card payment" });
img(s, SHOT("app_score_upi.png"), 8.75, 1.45, 4.0, 2.5, { name: A(), alt: "UPI mode with the no-signal policy" });
img(s, SHOT("app_policy.png"), 8.75, 4.05, 4.0, 2.5, { name: A(), alt: "Policy page re-optimising thresholds" });
T(s, "[Netlify URL] · Score · Batch CSV · Policy (enter official ₹ costs) · Evidence · sign-in with saved scorings and audit log (Supabase)", { x: MX, y: 6.55, w: W - 2 * MX, h: 0.4, fontSize: 13, color: C.accent5 });
s.addNotes("Switch to the live app. Load a test fraud: BLOCK with the reasons. Load a genuine: APPROVE. Edit V14 down and watch the risk climb. Open Policy and type a different missed-fraud cost to show the thresholds re-optimise. Switch to UPI mode to show the no-signal banner.");

/* ===================== 25 · special points + limitations */
s = content("Special points and limitations", "Action");
[["Confidentiality", ["Card features are PCA-anonymised by the data owner", "UPI file has no personal identifiers", "App scores in the browser; saves are row-level-secured per user"]],
 ["Compliance for rollout", ["DPDP Act 2023: consent, purpose, retention", "RBI fraud-reporting and customer-liability rules", "An appeal path for blocked customers"]],
 ["Limitations", ["~2 days of 2013 card data; 74 test frauds → wide CIs", "Amount currency not stated; costs per event", "Assumed ₹ costs → savings are scenarios", "Blocked fraud is never confirmed → keep a random holdout"]]].forEach(([t, items], i) => {
  const x = MX + i * 4.1; card(s, x, 1.5, 3.8, 4.0, HEX.paper, A());
  T(s, [{ text: t, options: { fontFace: "Cambria", bold: true, fontSize: 20, color: C.text1, breakLine: true } }, ...items.map((it, j) => ({ text: it, options: { bullet: true, breakLine: j < items.length - 1 } }))], { x: x + 0.28, y: 1.7, w: 3.25, h: 3.7, fontSize: 15, paraSpaceAfter: 8 });
});
s.addNotes("Close the analysis on honesty: what the data can and can't support, and what a real rollout would need.");

/* ===================== 26 · Close */
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Action" });
s.addText("From 99.87% accuracy to rupees saved", { placeholder: "title" });
s.addText("Recall, precision and PR-AUC instead of accuracy · a cost-based threshold instead of 0.5 · three bands instead of one score · a label-free watch-list for unseen fraud · a fair comparison of 22 models · three prioritised actions", { placeholder: "body" });
T(s, "Questions?", { x: MX, y: 6.1, w: 6, h: 0.6, fontFace: "Cambria", fontSize: 28, bold: true, color: C.accent1, objectName: A() });
s.addNotes("Recap the brief's 'from → to' table in one line and open for questions. Likely questions: why not UPI (slide 7), what if costs are wrong (slide 18), does SMOTE help (slide 13), how was leakage prevented (slide 11).");

(async () => {
  const out = path.join(ROOT, "docs/FraudGuard_Presentation.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out, "slides:", pres.slides.length);
})();
