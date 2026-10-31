/* FraudGuard web app — routing, scoring UI, batch, policy lab, evidence, Supabase auth + saves. */
(function () {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const inr = (v) => {
    v = Number(v); const a = Math.abs(v);
    if (a >= 1e7) return "₹" + (v / 1e7).toFixed(2) + " Cr";
    if (a >= 1e5) return "₹" + (v / 1e5).toFixed(2) + " L";
    return "₹" + Math.round(v).toLocaleString("en-IN");
  };
  const pct = (v, d = 1) => (100 * v).toFixed(d) + "%";
  const BANDCLS = { "APPROVE": "APPROVE", "STEP-UP OTP": "OTP", "BLOCK": "BLOCK" };
  const LIMITS = { scorings: 50, batch_runs: 5 };

  const S = { mode: localGet("fg-mode") || "creditcard", spec: {}, model: {}, policy: {}, sb: null, user: null, form: {}, lastLoaded: {} };

  function localGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
  function localSet(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } }
  function toast(msg) {
    const t = document.createElement("div"); t.className = "toast"; t.textContent = msg; t.setAttribute("role", "status");
    document.body.appendChild(t); setTimeout(() => t.remove(), 2600);
  }

  // ---------------------------------------------------------------- data loading
  async function load(mode) {
    if (S.spec[mode]) return;
    const [spec, cb] = await Promise.all([
      fetch(`models/${mode}.json`).then((r) => r.json()),
      fetch(`models/catboost_${mode}.json`).then((r) => r.json()),
    ]);
    S.spec[mode] = spec; S.model[mode] = FG.prepareModel(cb);
    S.policy[mode] = { name: "Frozen policy (validation-optimised)", costs: { ...spec.costs }, T_otp: spec.thresholds.T_otp, T_block: spec.thresholds.T_block, frozen: true };
  }
  const spec = () => S.spec[S.mode];
  const model = () => S.model[S.mode];
  const pol = () => S.policy[S.mode];

  // ---------------------------------------------------------------- scoring
  function scoreRow(row, mode = S.mode) {
    const sp = S.spec[mode], m = S.model[mode];
    const x = FG.featurize(row, sp);
    const p = FG.predictProba(m, x);
    const risk = FG.calibrate(sp, p);
    const P = S.policy[mode];
    let band = FG.band(p, P);
    if (sp.policy === "approve_all" && P.frozen) band = "APPROVE";
    return { x, p, risk, band };
  }
  function displayName(feat, sp) {
    if (feat === "log_amount") return "Amount";
    if (feat === "amount_zero") return "Zero-amount flag";
    if (feat === "hour_sin" || feat === "hour_cos") return "Hour of day";
    if (feat === "is_weekend") return "Weekend";
    for (const c of Object.keys(sp.cat_features || {})) if (feat.startsWith(c + "_")) return c.replace(/_/g, " ");
    return feat;
  }
  function groupedShap(x, mode = S.mode) {
    const sp = S.spec[mode], { phi, base } = FG.shapValues(S.model[mode], x);
    const g = new Map();
    sp.feature_order.forEach((f, i) => { const k = displayName(f, sp); g.set(k, (g.get(k) || 0) + phi[i]); });
    return { base, items: [...g.entries()].map(([name, v]) => ({ name, v })).sort((a, b) => Math.abs(b.v) - Math.abs(a.v)) };
  }

  // ---------------------------------------------------------------- views
  const views = {};

  views.score = function () {
    const sp = spec(), cc = sp.dataset === "creditcard";
    const main = $("#main");
    main.innerHTML = `
      <div class="split">
        <div>
          <h1>${cc ? "Should this card payment go through?" : "Should this UPI payment go through?"}</h1>
          <p class="lede">${cc
            ? "Enter a transaction or load one from the held-out test period. The model scores it, the policy turns the score into approve, step-up OTP or block, and the right-hand slip shows why."
            : "UPI mode runs the same pipeline on the organiser's UPI file. That file's fraud label is unrelated to every column, so the model cannot tell fraud apart, and the evidence-based policy approves every payment."}</p>
          <form id="txn" class="panel stack" autocomplete="off">${cc ? ccForm(sp) : upiForm(sp)}
            <div class="row">
              <button class="btn ghost" type="button" id="load-fraud">Load a test fraud</button>
              <button class="btn ghost" type="button" id="load-genuine">Load a test genuine</button>
            </div>
          </form>
        </div>
        <div id="slip"></div>
      </div>`;
    const form = $("#txn");
    form.addEventListener("input", debounce(() => renderSlip(readForm(form, sp)), 120));
    $("#load-fraud").onclick = () => loadDemo(form, sp, 1);
    $("#load-genuine").onclick = () => loadDemo(form, sp, 0);
    const prev = S.form[S.mode];
    if (prev) fillForm(form, prev.row, sp), renderSlip(prev.row, prev.label);
    else loadDemo(form, sp, 1, true);
  };

  function ccForm(sp) {
    const v = Array.from({ length: 28 }, (_, i) => `<div><label for="V${i + 1}">V${i + 1}</label><input type="number" step="any" id="V${i + 1}" name="V${i + 1}" value="0"></div>`).join("");
    return `<div class="grid2">
        <div><label for="Amount">Amount <span class="hint">(source currency not stated)</span></label><input type="number" step="0.01" min="0" id="Amount" name="Amount" value="10"></div>
        <div><label for="hour">Hour of day <span class="hint">(0–23)</span></label><input type="number" min="0" max="23" step="1" id="hour" name="hour" value="12"></div>
      </div>
      <details open><summary>Behaviour components V1–V28 <span class="hint">(anonymised by the bank with PCA for confidentiality)</span></summary><div class="vgrid">${v}</div></details>`;
  }
  function upiForm(sp) {
    const sel = (c, label) => `<div><label for="${cssId(c)}">${label}</label><select id="${cssId(c)}" name="${esc(c)}">${sp.cat_features[c].map((o) => `<option>${esc(o)}</option>`).join("")}</select></div>`;
    return `<div class="grid2">
      ${sel("transaction type", "Transaction type")}${sel("merchant_category", "Merchant category")}
      <div><label for="amt">Amount (₹)</label><input type="number" min="1" step="1" id="amt" name="amount (INR)" value="500"></div>
      <div><label for="hod">Hour of day</label><input type="number" min="0" max="23" id="hod" name="hour_of_day" value="14"></div>
      ${sel("day_of_week", "Day of week")}${sel("device_type", "Device")}
      ${sel("network_type", "Network")}${sel("sender_age_group", "Sender age group")}
      ${sel("receiver_age_group", "Receiver age group")}${sel("sender_state", "Sender state")}
      ${sel("sender_bank", "Sender bank")}${sel("receiver_bank", "Receiver bank")}
    </div>`;
  }
  const cssId = (c) => "f-" + c.replace(/[^a-z0-9]+/gi, "-");
  function readForm(form, sp) {
    const fd = new FormData(form), row = {};
    for (const [k, v] of fd.entries()) row[k] = v;
    if (sp.dataset === "creditcard") { row.Time = (+row.hour || 0) * 3600; delete row.hour; }
    else row.is_weekend = ["Saturday", "Sunday"].includes(row.day_of_week) ? 1 : 0;
    return row;
  }
  function fillForm(form, row, sp) {
    if (sp.dataset === "creditcard") {
      form.Amount.value = (+row.Amount).toFixed(2); form.hour.value = Math.floor(+row.Time / 3600) % 24;
      for (let i = 1; i <= 28; i++) form["V" + i].value = (+row["V" + i]).toFixed(4);
    } else {
      for (const el of $$("select, input", form)) if (el.name in row) el.value = row[el.name];
    }
  }
  function loadDemo(form, sp, label, first) {
    let pool = sp.demo.filter((r) => r.label === label);
    if (first && sp.has_signal) pool = pool.filter((r) => r.py_score >= sp.thresholds.T_block);   // first view: a fraud the model catches
    const row = pool[Math.floor(Math.random() * pool.length)];
    fillForm(form, row, sp);
    const parsed = readForm(form, sp);
    // keep exact source values (form rounding would perturb V's slightly)
    const exact = { ...parsed, ...pick(row, Object.keys(parsed)) };
    S.form[S.mode] = { row: exact, label };
    renderSlip(exact, label);
  }
  const pick = (o, keys) => Object.fromEntries(keys.filter((k) => k in o).map((k) => [k, o[k]]));

  function renderSlip(row, label) {
    const sp = spec(), P = pol(), r = scoreRow(row);
    if (label === undefined) S.form[S.mode] = { row, label: undefined };
    const cls = BANDCLS[r.band], c = P.costs;
    const pr = r.risk;
    const exp = { APPROVE: pr * c.X, "STEP-UP OTP": c.Z + pr * (1 - c.otp_stop_rate) * c.X, BLOCK: (1 - pr) * c.Y };
    const sh = groupedShap(r.x);
    const top = sh.items.slice(0, 8), maxAbs = Math.max(...top.map((t) => Math.abs(t.v)), 1e-9);
    const noSignal = sp.policy === "approve_all" && P.frozen;
    $("#slip").innerHTML = `
      <section class="panel slip" aria-live="polite">
        <div class="slip-head">
          <div><p class="muted small" style="margin:0 0 6px">Decision</p><span class="stamp ${cls} reveal">${r.band}</span></div>
          <div class="riskfig"><p class="muted small" style="margin:0">Risk score</p><strong>${(pr * 100).toFixed(pr < 0.01 ? 2 : 1)}</strong><span class="muted small">calibrated chance of fraud, out of 100</span></div>
        </div>
        ${noSignal ? `<div class="notice bad"><strong>No learnable signal in this data.</strong> On the held-out UPI test set this model's PR-AUC is ${sp.metrics.test_pr_auc.toFixed(4)} against a base rate of ${sp.metrics.base_rate.toFixed(4)} (permutation p = ${sp.metrics.signal_p.toFixed(2)}). Any block or OTP would only add friction, so the policy approves everything. The score is shown for transparency.</div>` : ""}
        <div class="track">${trackSvg(r.p, P)}</div>
        <p class="small muted" style="margin:0">Raw model score ${r.p.toExponential(2)} · thresholds: OTP from ${fmtT(P.T_otp)}, block from ${fmtT(P.T_block)} · ${esc(P.name)}</p>
        <div class="costs" role="group" aria-label="Expected cost of each action">
          ${["APPROVE", "STEP-UP OTP", "BLOCK"].map((b) => `<div class="${b === r.band ? "chosen" : ""}"><div class="k">${b === "STEP-UP OTP" ? "Step-up OTP" : b[0] + b.slice(1).toLowerCase()}</div><div class="v">${inr(exp[b])}</div></div>`).join("")}
        </div>
        <p class="small muted" style="margin-top:6px">Expected cost of each action for this payment, from its calibrated risk and the policy's costs (missed fraud ${inr(c.X)}, wrong block ${inr(c.Y)}, OTP ${inr(c.Z)}, OTP stops ${pct(c.otp_stop_rate, 0)} of fraud).</p>
        <div class="why">
          <h3>Why — what moved this score</h3>
          ${top.map((t) => `<div class="bar"><span class="name" title="${esc(t.name)}">${esc(t.name)}</span><span class="rail"><span class="fill ${t.v > 0 ? "up" : "down"}" style="width:${(50 * Math.abs(t.v) / maxAbs).toFixed(1)}%"></span></span><span class="val">${t.v > 0 ? "+" : ""}${t.v.toFixed(2)}</span></div>`).join("")}
          <p class="small muted">Exact TreeSHAP contributions in log-odds, computed in your browser (red pushes towards fraud, green towards genuine). They show how the model uses each input, not what causes fraud.</p>
        </div>
        ${label !== undefined ? `<p class="truth">Loaded from the held-out test period. Actual label: <strong>${label ? "fraud" : "genuine"}</strong>.</p>` : ""}
        <div class="row" style="margin-top:10px"><button class="btn" id="save-scoring" type="button" ${S.user ? "" : "disabled"}>Save scoring</button>${S.user ? "" : `<span class="small muted">${S.sb ? '<a href="#login">Sign in</a> to save scorings.' : "Saving needs Supabase keys in config.js."}</span>`}</div>
      </section>`;
    const btn = $("#save-scoring");
    if (btn) btn.onclick = () => saveScoring(row, r, label);
  }
  const fmtT = (t) => (t >= 1.5 ? "never" : t < 0.01 ? t.toExponential(2) : t.toFixed(3));

  function trackSvg(p, P) {
    const W = 600, x0 = 10, x1 = 590, lo = -6, hi = 0;
    const X = (s) => x0 + (Math.min(Math.max(Math.log10(Math.max(s, 1e-6)), lo), hi) - lo) / (hi - lo) * (x1 - x0);
    const to = P.T_otp >= 1.5 ? x1 : X(P.T_otp), tb = P.T_block >= 1.5 ? x1 : X(P.T_block), xm = X(p);
    const ticks = [-6, -5, -4, -3, -2, -1, 0];
    return `<svg viewBox="0 0 ${W} 96" role="img" aria-label="Score ${p.toExponential(2)} on a log scale with the OTP and block thresholds">
      <rect x="${x0}" y="24" width="${to - x0}" height="14" rx="3" fill="var(--approve-tint)"/>
      <rect x="${to}" y="24" width="${Math.max(tb - to, 0)}" height="14" fill="var(--otp-tint)"/>
      <rect x="${tb}" y="24" width="${Math.max(x1 - tb, 0)}" height="14" rx="3" fill="var(--block-tint)"/>
      ${P.T_otp < 1.5 ? `<line x1="${to}" x2="${to}" y1="18" y2="44" stroke="var(--otp)" stroke-width="2"/><text x="${to}" y="12" text-anchor="middle" font-size="12" fill="var(--otp)" font-weight="600">OTP</text>` : ""}
      ${P.T_block < 1.5 ? `<line x1="${tb}" x2="${tb}" y1="18" y2="44" stroke="var(--block)" stroke-width="2"/><text x="${tb}" y="12" text-anchor="middle" font-size="12" fill="var(--block)" font-weight="600">Block</text>` : ""}
      ${ticks.map((t) => `<text x="${X(10 ** t)}" y="76" text-anchor="middle" font-size="11" fill="var(--muted)">${t === 0 ? "1" : "10" + sup(t)}</text>`).join("")}
      <text x="${x0}" y="92" font-size="11" fill="var(--muted)">model score (log scale)</text>
      <g class="track-marker" style="transform: translateX(${xm}px)"><line x1="0" x2="0" y1="20" y2="42" stroke="var(--ink)" stroke-width="2.5"/><path d="M0 46 L-7 60 L7 60 Z" fill="var(--ink)"/></g>
    </svg>`;
  }
  const sup = (n) => String(n).replace("-", "⁻").replace(/\d/g, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]);

  // ---------------------------------------------------------------- batch
  const REQUIRED = {
    creditcard: ["Time", "Amount", ...Array.from({ length: 28 }, (_, i) => "V" + (i + 1))],
    upi: ["transaction type", "merchant_category", "amount (INR)", "sender_age_group", "receiver_age_group", "sender_state", "sender_bank", "receiver_bank", "device_type", "network_type", "hour_of_day", "day_of_week", "is_weekend"],
  };
  const LABEL = { creditcard: "Class", upi: "fraud_flag" };

  views.batch = function () {
    const m = S.mode;
    $("#main").innerHTML = `
      <h1>Score a whole file</h1>
      <p class="lede">Upload a CSV in the ${m === "creditcard" ? "ULB credit-card" : "UPI 2024"} format. Every row is scored in your browser. If the file has the <code>${LABEL[m]}</code> label, you also get recall, precision and the rupee cost of each band.</p>
      <div class="panel">
        <div class="drop" id="drop"><p><strong>Drop a CSV here</strong> or choose one.</p><input type="file" id="file" accept=".csv,text/csv"><p class="small muted">Needs these ${REQUIRED[m].length} columns: ${REQUIRED[m].map(esc).join(", ")}.</p>
        <button class="btn ghost" type="button" id="sample">Use the ${spec().demo.length}-row held-out sample</button></div>
        <div id="batch-out"></div>
      </div>`;
    const drop = $("#drop"), file = $("#file");
    file.onchange = () => file.files[0] && parseFile(file.files[0]);
    drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
    drop.ondragleave = () => drop.classList.remove("over");
    drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); e.dataTransfer.files[0] && parseFile(e.dataTransfer.files[0]); };
    $("#sample").onclick = () => runBatch(spec().demo.map((r) => { const o = { ...r }; o[LABEL[m]] = r.label; return o; }), "held-out-sample.csv");
  };
  function parseFile(f) {
    $("#batch-out").innerHTML = `<p class="muted">Reading ${esc(f.name)}…</p>`;
    Papa.parse(f, { header: true, dynamicTyping: true, skipEmptyLines: true, worker: false,
      complete: (res) => runBatch(res.data, f.name),
      error: (e) => { $("#batch-out").innerHTML = `<div class="notice bad">Could not read the file: ${esc(e.message)}</div>`; } });
  }
  async function runBatch(rows, name) {
    const m = S.mode, out = $("#batch-out");
    const cols = rows.length ? Object.keys(rows[0]) : [];
    const missing = REQUIRED[m].filter((c) => !cols.includes(c));
    if (missing.length) { out.innerHTML = `<div class="notice bad">This file is missing ${missing.length} required column${missing.length > 1 ? "s" : ""}: ${missing.map(esc).join(", ")}. Switch mode if this is the other dataset.</div>`; return; }
    const hasLabel = cols.includes(LABEL[m]);
    const scored = new Array(rows.length);
    const t0 = performance.now();
    for (let i = 0; i < rows.length; i += 2000) {
      for (let j = i; j < Math.min(i + 2000, rows.length); j++) { const r = scoreRow(rows[j], m); scored[j] = { p: r.p, risk: r.risk, band: r.band }; }
      out.innerHTML = `<p class="muted">Scoring… ${Math.min(i + 2000, rows.length).toLocaleString()} of ${rows.length.toLocaleString()}</p>`;
      await new Promise((r) => setTimeout(r));
    }
    const ms = performance.now() - t0, c = pol().costs;
    const bands = ["APPROVE", "STEP-UP OTP", "BLOCK"].map((b) => ({ band: b, txns: 0, frauds: 0, genuine: 0, cost: 0 }));
    const bi = { "APPROVE": 0, "STEP-UP OTP": 1, "BLOCK": 2 };
    scored.forEach((s, i) => {
      const B = bands[bi[s.band]]; B.txns++;
      if (hasLabel) {
        const y = +rows[i][LABEL[m]] === 1;
        if (y) { B.frauds++; B.cost += s.band === "APPROVE" ? c.X : s.band === "BLOCK" ? 0 : c.Z + (1 - c.otp_stop_rate) * c.X; }
        else { B.genuine++; B.cost += s.band === "STEP-UP OTP" ? c.Z : s.band === "BLOCK" ? c.Y : 0; }
      }
    });
    let metrics = null;
    if (hasLabel) {
      const y = rows.map((r) => (+r[LABEL[m]] === 1 ? 1 : 0)), nF = y.reduce((a, b) => a + b, 0);
      const caught = bands[1].frauds + bands[2].frauds, flagged = bands[1].txns + bands[2].txns;
      metrics = { nF, recall: nF ? caught / nF : 0, precision: flagged ? caught / flagged : 0, ap: avgPrecision(y, scored.map((s) => s.p)),
                  total: bands.reduce((a, b) => a + b.cost, 0), approveAll: nF * c.X };
    }
    const summary = { name, rows: rows.length, hasLabel, bands, metrics, policy: { name: pol().name, T_otp: pol().T_otp, T_block: pol().T_block, costs: c }, ms };
    S.lastBatch = { rows, scored, summary };
    out.innerHTML = `
      <h2 style="margin-top:22px">${esc(name)}: ${rows.length.toLocaleString()} payments scored in ${ms < 1000 ? Math.max(1, Math.round(ms)) + " ms" : (ms / 1000).toFixed(1) + " s"}</h2>
      ${name === "held-out-sample.csv" ? `<div class="notice">This sample keeps every test fraud plus 200 random genuine payments, so fraud is far more common here than in real traffic and precision looks higher than it is. The Evidence page has the full test-period numbers.</div>` : ""}
      <div class="table-wrap"><table><thead><tr><th>Band</th><th class="num">Payments</th><th class="num">Share</th>${hasLabel ? '<th class="num">Fraud</th><th class="num">Genuine</th><th class="num">Expected cost</th>' : ""}</tr></thead><tbody>
      ${bands.map((b) => `<tr><td><span class="pill ${BANDCLS[b.band]}">${b.band}</span></td><td class="num">${b.txns.toLocaleString()}</td><td class="num">${pct(b.txns / rows.length, 2)}</td>${hasLabel ? `<td class="num">${b.frauds}</td><td class="num">${b.genuine.toLocaleString()}</td><td class="num">${inr(b.cost)}</td>` : ""}</tr>`).join("")}
      </tbody></table></div>
      ${metrics ? `<p style="margin-top:12px">Fraud caught or challenged: <strong>${pct(metrics.recall)}</strong> of ${metrics.nF} · precision of OTP+block: <strong>${pct(metrics.precision)}</strong> · PR-AUC <strong>${metrics.ap.toFixed(3)}</strong> · expected cost <strong>${inr(metrics.total)}</strong> against ${inr(metrics.approveAll)} if everything were approved.</p>` : `<p class="muted">No <code>${LABEL[m]}</code> column, so only the band counts are shown.</p>`}
      <div class="row"><button class="btn" id="dl" type="button">Download scored CSV</button><button class="btn ghost" id="save-batch" type="button" ${S.user ? "" : "disabled"}>Save this run</button></div>`;
    $("#dl").onclick = () => download(name.replace(/\.csv$/i, "") + "_scored.csv", toCsv(rows, scored));
    $("#save-batch").onclick = () => saveBatch();
  }
  function avgPrecision(y, s) {
    const o = s.map((v, i) => [v, y[i]]).sort((a, b) => b[0] - a[0]);
    const P = y.reduce((a, b) => a + b, 0); if (!P) return 0;
    let tp = 0, fp = 0, ap = 0, prevR = 0;
    for (let i = 0; i < o.length; i++) {
      o[i][1] ? tp++ : fp++;
      if (i === o.length - 1 || o[i + 1][0] !== o[i][0]) { const r = tp / P; ap += (r - prevR) * (tp / (tp + fp)); prevR = r; }
    }
    return ap;
  }
  function toCsv(rows, scored) {
    const data = rows.map((r, i) => ({ ...r, fraud_score: scored[i].p, risk_0_100: +(scored[i].risk * 100).toFixed(3), decision: scored[i].band }));
    return Papa.unparse(data);
  }
  function download(name, text) {
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type: "text/csv" })); a.download = name; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }

  // ---------------------------------------------------------------- policy lab
  views.policy = function () {
    const sp = spec(), P = pol(), c = P.costs;
    $("#main").innerHTML = `
      <h1>Set the rupee costs, get the thresholds</h1>
      <p class="lede">The organisers published no official costs, so FraudGuard ships stated assumptions. Enter your own: both thresholds are re-chosen on the validation period, then scored on the untouched test period.</p>
      <div class="split">
        <form class="panel stack" id="costs">
          <div class="grid2">
            <div><label for="cX">Missed fraud (₹X)</label><input id="cX" type="number" min="0" step="100" value="${c.X}"></div>
            <div><label for="cY">Genuine payment blocked (₹Y)</label><input id="cY" type="number" min="0" step="10" value="${c.Y}"></div>
            <div><label for="cZ">One OTP challenge (₹Z)</label><input id="cZ" type="number" min="0" step="1" value="${c.Z}"></div>
            <div><label for="cS">Share of fraud OTP stops</label><input id="cS" type="number" min="0" max="1" step="0.05" value="${c.otp_stop_rate}"></div>
          </div>
          <div class="row"><button class="btn" type="submit">Re-optimise thresholds</button><button class="btn ghost" type="button" id="reset">Back to the frozen policy</button></div>
          <p class="small muted">Default ₹5,000 / ₹300 / ₹15 / 70%. ₹15 per OTP is mainly genuine drop-off (about 5% of challenged customers × ₹300), not the SMS itself.</p>
          ${S.user ? `<div class="row"><input type="text" id="pname" placeholder="Policy name" style="max-width:240px"><button class="btn ghost" type="button" id="psave">Save policy</button></div>` : ""}
        </form>
        <div id="policy-out"></div>
      </div>
      <section class="block" id="saved-policies"></section>`;
    const read = () => ({ X: +$("#cX").value, Y: +$("#cY").value, Z: +$("#cZ").value, otp_stop_rate: Math.min(Math.max(+$("#cS").value, 0), 1) });
    $("#costs").onsubmit = (e) => { e.preventDefault(); const cc = read(); const best = FG.optimisePolicy(sp, cc); S.policy[S.mode] = { name: "Custom costs (re-optimised)", costs: cc, T_otp: best.T_otp, T_block: best.T_block, frozen: false, valCost: best.cost }; renderPolicy(); logAudit("policy_changed", { mode: S.mode, costs: cc, T_otp: best.T_otp, T_block: best.T_block }); };
    $("#reset").onclick = () => { S.policy[S.mode] = { name: "Frozen policy (validation-optimised)", costs: { ...sp.costs }, T_otp: sp.thresholds.T_otp, T_block: sp.thresholds.T_block, frozen: true }; views.policy(); };
    if ($("#psave")) $("#psave").onclick = () => savePolicy($("#pname").value || "Untitled policy");
    renderPolicy(); listPolicies();
  };
  function renderPolicy() {
    const sp = spec(), P = pol(), c = P.costs;
    const test = FG.policyCost(sp.search.test, Math.max(P.T_otp, sp.search.test.cutoff), Math.max(P.T_block, sp.search.test.cutoff), c);
    const allApprove = sp.search.test.n_fraud * c.X;
    const frozen = FG.policyCost(sp.search.test, Math.max(sp.thresholds.T_otp, sp.search.test.cutoff), Math.max(sp.thresholds.T_block, sp.search.test.cutoff), c);
    const below = P.T_otp < sp.search.test.cutoff;
    $("#policy-out").innerHTML = `<section class="panel">
      <h2>${esc(P.name)}</h2>
      <dl class="kv"><dt>Step-up OTP from</dt><dd>score ${fmtT(P.T_otp)}</dd><dt>Block from</dt><dd>score ${fmtT(P.T_block)}</dd>
      <dt>Test-period cost</dt><dd><strong>${inr(test.cost)}</strong> (approve everything: ${inr(allApprove)}; frozen thresholds under these costs: ${inr(frozen.cost)})</dd></dl>
      ${below ? `<div class="notice">The OTP threshold sits below the stored top-5% slice of test scores, so test counts are approximate.</div>` : ""}
      ${sp.policy === "approve_all" ? `<div class="notice bad">UPI has no learnable signal, so any band other than approve only adds friction. A cost setting that moves thresholds here is fitting noise in the validation period.</div>` : ""}
      <div class="table-wrap"><table><thead><tr><th>Band (test)</th><th class="num">Payments</th><th class="num">Fraud</th><th class="num">Genuine affected</th><th class="num">Cost</th></tr></thead><tbody>
      ${test.bands.map((b) => `<tr><td><span class="pill ${BANDCLS[b.band]}">${b.band}</span></td><td class="num">${b.txns.toLocaleString()}</td><td class="num">${b.frauds}</td><td class="num">${b.band === "APPROVE" ? "0" : b.genuine.toLocaleString()}</td><td class="num">${inr(b.cost)}</td></tr>`).join("")}
      </tbody></table></div>
      <p class="small muted">Scoring on the Score and Batch pages now uses this policy. Observed counts on held-out data × your assumed costs; savings are scenario estimates.</p>
    </section>`;
  }

  // ---------------------------------------------------------------- evidence
  views.evidence = function () {
    const sp = spec(), cc = sp.dataset === "creditcard", fig = (f, cap) => `<figure><img loading="lazy" src="assets/fig/${sp.dataset}/${f}" alt="${esc(cap)}"><figcaption>${esc(cap)}</figcaption></figure>`;
    const tr = sp.trap || [];
    $("#main").innerHTML = `
      <h1>${cc ? "Evidence: credit-card fraud model" : "Evidence: UPI negative control"}</h1>
      <p class="lede">${cc ? "Everything below comes from the notebook run with seed 42: a time-ordered 60/20/20 split, thresholds frozen on validation, and test scored once." : "Same notebook, same code, the UPI file. Every check comes back at chance, which is itself the finding."}</p>
      <section class="block cols2">
        <div class="panel"><h2>The datasets</h2>${datasetsHtml()}</div>
        <div class="panel"><h2>Confidentiality and data handling</h2>
          <ul><li>The credit-card features V1–V28 were anonymised by the data owner with PCA. Original fields and cardholder identities are not released, so model reasons cannot be named in business terms.</li>
          <li>The UPI file holds no names, account numbers, VPAs or phone numbers. Banks and states appear only as categories.</li>
          <li>The app scores in the browser. Signed-in saves go to your own Supabase rows, protected by row-level security. A real deployment would also need to meet India's DPDP Act 2023 consent and retention rules and RBI fraud-reporting norms.</li></ul></div>
      </section>
      <section class="block panel"><h2>The accuracy trap (test period)</h2>
        <div class="table-wrap"><table><thead><tr><th>Model</th><th class="num">Accuracy</th><th class="num">Recall</th><th class="num">Precision</th><th class="num">Missed fraud</th><th class="num">Genuine blocked</th><th class="num">Expected cost</th></tr></thead><tbody>
        ${tr.map((t) => `<tr><td>${esc(t.name)}</td><td class="num">${pct(t.accuracy, 2)}</td><td class="num">${pct(t.recall)}</td><td class="num">${t.tp + t.fp ? pct(t.precision) : "—"}</td><td class="num">${t.fn}</td><td class="num">${t.fp}</td><td class="num">${inr(t.cost_inr)}</td></tr>`).join("")}
        </tbody></table></div><p class="small muted">A model that approves everything is ${tr[0] ? pct(tr[0].accuracy, 2) : ""} accurate and catches no fraud. FraudGuard is judged on recall, precision, PR-AUC and rupee cost instead.</p></section>
      <section class="block panel"><h2>Data quality audit</h2><div class="table-wrap"><table><thead><tr><th>Check</th><th>Finding</th><th>Treatment</th></tr></thead><tbody>
        ${(sp.audit || []).map((a) => `<tr><td>${esc(a.Check)}</td><td>${esc(a.Finding)}</td><td>${esc(a.Treatment)}</td></tr>`).join("")}</tbody></table></div>
        <h3 style="margin-top:14px">Columns excluded from the model</h3><ul>${(sp.excluded || []).map((e) => `<li><strong>${esc(e.Column)}</strong>: ${esc(e.Reason)}</li>`).join("")}</ul></section>
      <section class="block panel"><h2>Model comparison (${sp.comparison.length} models; selection on validation)</h2>
        <div class="table-wrap"><table><thead><tr><th>Model</th><th>Imbalance handling</th><th class="num">Val PR-AUC</th><th class="num">Test PR-AUC</th><th class="num">Test ROC-AUC</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">Test cost</th></tr></thead><tbody>
        ${sp.comparison.map((r) => `<tr class="${r.model === sp.final_model ? "hl" : ""}"><td>${esc(r.model)}${r.model === sp.final_model ? " (selected)" : ""}</td><td>${esc(r.imbalance)}</td><td class="num">${r.val_pr_auc.toFixed(3)}</td><td class="num">${r.test_pr_auc.toFixed(3)}</td><td class="num">${r.test_roc_auc.toFixed(3)}</td><td class="num">${pct(r.test_precision)}</td><td class="num">${pct(r.test_recall)}</td><td class="num">${inr(r.test_cost)}</td></tr>`).join("")}
        </tbody></table></div><p class="small muted">Precision, recall and cost at each model's own cost-optimal threshold, chosen on validation.</p></section>
      <section class="block panel"><h2>Label-free anomaly detectors</h2><div class="table-wrap"><table><thead><tr><th>Detector (no labels used)</th><th class="num">Test PR-AUC</th><th class="num">Fraud in top 1%</th><th class="num">Fraud in top 5%</th></tr></thead><tbody>
        ${sp.anomaly.map((a) => `<tr><td>${esc(a.model)}</td><td class="num">${a.test_pr_auc.toFixed(3)}</td><td class="num">${pct(a.recall_top_1pct, 0)}</td><td class="num">${pct(a.recall_top_5pct, 0)}</td></tr>`).join("")}</tbody></table></div></section>
      <section class="block panel"><h2>Risk-score rule on the test period</h2><div class="table-wrap"><table><thead><tr><th>Band</th><th class="num">Payments</th><th class="num">Fraud</th><th class="num">Share of fraud</th><th class="num">Genuine affected</th><th class="num">Expected cost</th></tr></thead><tbody>
        ${sp.test_bands.map((b) => `<tr><td><span class="pill ${BANDCLS[b.band]}">${b.band}</span></td><td class="num">${b.transactions.toLocaleString()}</td><td class="num">${b.frauds}</td><td class="num">${b.pct_of_all_fraud.toFixed(1)}%</td><td class="num">${b.band === "APPROVE" ? 0 : b.genuine}</td><td class="num">${inr(b.total_cost)}</td></tr>`).join("")}
        </tbody></table></div>
        <ul>${Object.entries(sp.policies).map(([k, v]) => `<li>${esc(k)}: <strong>${inr(v)}</strong></li>`).join("")}</ul>
        ${sp.policy === "approve_all" ? `<div class="notice bad">No model-driven rule beats approving everything on UPI. The app therefore enforces approve-all for this mode.</div>` : ""}</section>
      <section class="block"><h2>Figures</h2><div class="figs">${(cc ? [
          ["01_class_imbalance.png", "Class imbalance (log scale)"], ["08_accuracy_trap.png", "The accuracy trap"],
          ["02_fraud_by_amount_hour.png", "Fraud rate by amount band and hour, with 95% CIs"], ["04_fraud_over_time.png", "Fraud over time: bursts at night"],
          ["03_behaviour_signature.png", "Behaviour signature (KS statistic)"], ["05_weights_vs_smote.png", "Class weights vs SMOTE"],
          ["05b_all_models_pr_auc_ci.png", "All models: PR-AUC with 95% CI"], ["07_pr_curves.png", "Precision–recall curves"],
          ["06_anomaly_vs_supervised.png", "Label-free detectors vs supervised"], ["09_cost_threshold_confusion.png", "Cost vs threshold and rupee confusion matrix"],
          ["10_cost_sensitivity.png", "Threshold sensitivity to ₹X and ₹Y"], ["10b_otp_sensitivity.png", "Sensitivity to OTP cost and stop rate"],
          ["11_risk_bands.png", "Risk bands on test"], ["11b_cost_waterfall.png", "Cost bridge: approve-all to 3-band rule"],
          ["12_shap_beeswarm.png", "SHAP summary"], ["13_shap_waterfall.png", "One blocked fraud explained"]] : [
          ["01_class_imbalance.png", "Class imbalance"], ["02_upi_fraud_rates.png", "Fraud rate by type, merchant, amount, hour, age, device"],
          ["04_upi_fraud_by_month.png", "Fraud rate by month"], ["05b_all_models_pr_auc_ci.png", "All models at the random line"],
          ["07_pr_curves.png", "Precision–recall: every model at chance"], ["06_anomaly_vs_supervised.png", "Anomaly detectors at chance"]]
        ).map(([f, c]) => fig(f, c)).join("")}</div></section>
      <section class="block panel"><h2>Limitations</h2><ul>
        <li>Credit card: about two days of September 2013 data from European cardholders; 57 validation and 74 test frauds, so costs carry wide confidence intervals.</li>
        <li>The <code>Amount</code> column's currency is not stated by the source, so rupee costs are applied per event, never as a share of the amount.</li>
        <li>Rupee costs and the 70% OTP stop rate are stated assumptions (no official figures were published); the Policy page recomputes everything for any values.</li>
        <li>UPI: the file is simulated, has no account or device IDs, and its label carries no signal, so no UPI model can be deployed until labels and IDs are collected.</li>
        <li>Blocked fraud is never confirmed in production. A random holdout of unblocked traffic is needed to keep future labels unbiased.</li></ul></section>`;
  }
  function datasetsHtml() {
    const c = S.spec.creditcard, u = S.spec.upi;
    return `<h3>ULB Credit Card Fraud <a class="small" href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" target="_blank" rel="noopener">(Kaggle)</a></h3>
      <p>Card transactions made by European cardholders in September 2013, collected by Worldline and the Machine Learning Group of ULB (Brussels). There are 284,807 rows over two days, 492 of them fraud (0.172%). <code>Time</code> is seconds since the first transaction. <code>Amount</code> is the transaction amount, with no currency stated. V1–V28 are PCA components; the original features are withheld for confidentiality. <code>Class</code> = 1 for fraud. After removing 1,081 exact duplicates: ${c ? c.n_rows.toLocaleString() : "283,726"} rows, ${c ? c.n_fraud : 473} frauds.</p>
      <h3>UPI Transactions 2024 (organiser-supplied)</h3>
      <p>A simulated UPI-style file: 250,000 transactions dated 1 Jan – 30 Dec 2024, 17 columns, 480 frauds (0.192%). It has transaction type (P2P, P2M, bill payment, recharge), merchant category, amount in ₹, sender and receiver age groups, sender state, sender and receiver banks, device, network, status, hour, weekday and a fraud flag. It has no sender, receiver or device identifiers.</p>`;
  }

  // ---------------------------------------------------------------- auth + history (Supabase)
  views.login = () => authView("login");
  views.signup = () => authView("signup");
  function authView(kind) {
    if (!S.sb) { $("#main").innerHTML = `<div class="auth panel"><h1>Sign-in is off</h1><p>This deployment has no Supabase keys yet. Add them to <code>config.js</code> (see README). Scoring, batch, policy and evidence all work without an account.</p></div>`; return; }
    $("#main").innerHTML = `<form class="auth panel stack" id="authf"><h1>${kind === "login" ? "Sign in" : "Create an account"}</h1>
      <div><label for="em">Email</label><input id="em" type="email" required autocomplete="email"></div>
      <div><label for="pw">Password</label><input id="pw" type="password" minlength="8" required autocomplete="${kind === "login" ? "current-password" : "new-password"}"></div>
      <button class="btn" type="submit">${kind === "login" ? "Sign in" : "Create account"}</button>
      <button class="btn ghost" type="button" id="google">Continue with Google</button>
      <p class="small">${kind === "login" ? 'New here? <a href="#signup">Create an account</a>' : 'Have an account? <a href="#login">Sign in</a>'}</p>
      <p class="small muted" id="authmsg"></p></form>`;
    $("#authf").onsubmit = async (e) => {
      e.preventDefault(); const email = $("#em").value, password = $("#pw").value;
      const { error } = kind === "login" ? await S.sb.auth.signInWithPassword({ email, password }) : await S.sb.auth.signUp({ email, password, options: { emailRedirectTo: location.origin } });
      $("#authmsg").textContent = error ? error.message : (kind === "signup" ? "Check your inbox to confirm your email, then sign in." : "");
      if (!error && kind === "login") location.hash = "#score";
    };
    $("#google").onclick = () => S.sb.auth.signInWithOAuth({ provider: "google", options: { redirectTo: location.origin } });
  }
  async function logAudit(action, detail) { if (S.sb && S.user) await S.sb.from("audit_log").insert({ user_id: S.user.id, action, detail }); }
  async function countRows(table) { const { count } = await S.sb.from(table).select("id", { count: "exact", head: true }).eq("user_id", S.user.id); return count || 0; }
  async function saveScoring(row, r, label) {
    if (!S.user) return;
    if (await countRows("scorings") >= LIMITS.scorings) { toast(`You have ${LIMITS.scorings} saved scorings. Delete some in History first.`); return; }
    const { error } = await S.sb.from("scorings").insert({ user_id: S.user.id, mode: S.mode, input: row, score: r.p, risk: r.risk, decision: r.band, policy: { name: pol().name, T_otp: pol().T_otp, T_block: pol().T_block }, label: label ?? null });
    if (error) return toast(error.message);
    await logAudit("scoring_saved", { mode: S.mode, decision: r.band }); toast("Scoring saved");
  }
  async function saveBatch() {
    if (!S.user || !S.lastBatch) return;
    if (await countRows("batch_runs") >= LIMITS.batch_runs) { toast(`You have ${LIMITS.batch_runs} saved runs. Delete one in History first.`); return; }
    const { rows, scored, summary } = S.lastBatch;
    const path = `${S.user.id}/${Date.now()}_${summary.name.replace(/[^\w.-]+/g, "_")}`;
    const up = await S.sb.storage.from("batches").upload(path, new Blob([toCsv(rows, scored)], { type: "text/csv" }));
    if (up.error) return toast(up.error.message);
    const { error } = await S.sb.from("batch_runs").insert({ user_id: S.user.id, mode: S.mode, file_name: summary.name, n_rows: summary.rows, summary, storage_path: path });
    if (error) return toast(error.message);
    await logAudit("batch_saved", { mode: S.mode, rows: summary.rows }); toast("Run saved");
  }
  async function savePolicy(name) {
    const P = pol();
    const { error } = await S.sb.from("policies").insert({ user_id: S.user.id, mode: S.mode, name, costs: P.costs, t_otp: P.T_otp, t_block: P.T_block });
    if (error) return toast(error.message);
    await logAudit("policy_saved", { mode: S.mode, name }); toast("Policy saved"); listPolicies();
  }
  async function listPolicies() {
    const el = $("#saved-policies"); if (!el || !S.user) return;
    const { data } = await S.sb.from("policies").select("*").eq("mode", S.mode).order("created_at", { ascending: false });
    el.innerHTML = data && data.length ? `<h2>Your saved policies</h2><div class="table-wrap"><table><tbody>${data.map((p) => `<tr><td>${esc(p.name)}</td><td class="small muted">X ₹${p.costs.X} · Y ₹${p.costs.Y} · Z ₹${p.costs.Z} · stop ${p.costs.otp_stop_rate}</td><td><button class="btn ghost" data-apply="${p.id}" type="button">Apply</button></td></tr>`).join("")}</tbody></table></div>` : "";
    $$("[data-apply]", el).forEach((b) => (b.onclick = () => { const p = data.find((d) => d.id === b.dataset.apply); S.policy[S.mode] = { name: p.name, costs: p.costs, T_otp: p.t_otp, T_block: p.t_block, frozen: false }; logAudit("policy_applied", { name: p.name }); views.policy(); }));
  }
  views.history = async function () {
    if (!S.user) { location.hash = "#login"; return; }
    $("#main").innerHTML = `<h1>Your history</h1><p class="lede">Saved scorings (up to ${LIMITS.scorings}), batch runs (up to ${LIMITS.batch_runs}), policies, and the audit log of every save, delete and policy change.</p><div id="hist" class="stack"><p class="muted">Loading…</p></div>`;
    const [sc, br, po, au] = await Promise.all(["scorings", "batch_runs", "policies", "audit_log"].map((t) => S.sb.from(t).select("*").order("created_at", { ascending: false }).limit(t === "audit_log" ? 100 : 50)));
    const when = (d) => new Date(d).toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" });
    $("#hist").innerHTML = `
      <section class="panel"><h2>Scorings</h2>${tbl(sc.data, ["When", "Mode", "Decision", "Risk", ""], (r) => [when(r.created_at), r.mode, `<span class="pill ${BANDCLS[r.decision]}">${r.decision}</span>`, (r.risk * 100).toFixed(2), del("scorings", r.id)])}</section>
      <section class="panel"><h2>Batch runs</h2>${tbl(br.data, ["When", "File", "Rows", "Mode", "", ""], (r) => [when(r.created_at), esc(r.file_name), r.n_rows.toLocaleString(), r.mode, `<button class="btn ghost" type="button" data-dl="${esc(r.storage_path)}">Download</button>`, del("batch_runs", r.id, r.storage_path)])}</section>
      <section class="panel"><h2>Policies</h2>${tbl(po.data, ["When", "Name", "Mode", "Thresholds", ""], (r) => [when(r.created_at), esc(r.name), r.mode, `${fmtT(r.t_otp)} / ${fmtT(r.t_block)}`, del("policies", r.id)])}</section>
      <section class="panel"><h2>Audit log</h2>${tbl(au.data, ["When", "Action", "Detail"], (r) => [when(r.created_at), esc(r.action), `<code class="small">${esc(JSON.stringify(r.detail))}</code>`])}</section>`;
    $$("[data-del]").forEach((b) => (b.onclick = async () => {
      const [t, id, path] = b.dataset.del.split("|");
      if (path) await S.sb.storage.from("batches").remove([path]);
      const { error } = await S.sb.from(t).delete().eq("id", id); if (error) return toast(error.message);
      await logAudit("deleted", { table: t, id }); toast("Deleted"); views.history();
    }));
    $$("[data-dl]").forEach((b) => (b.onclick = async () => { const { data, error } = await S.sb.storage.from("batches").download(b.dataset.dl); if (error) return toast(error.message); download(b.dataset.dl.split("/").pop(), await data.text()); }));
  };
  const del = (t, id, path = "") => `<button class="btn ghost" type="button" data-del="${t}|${id}|${esc(path)}">Delete</button>`;
  const tbl = (rows, head, f) => rows && rows.length ? `<div class="table-wrap"><table><thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${f(r).map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table></div>` : `<p class="muted">Nothing saved yet.</p>`;

  // ---------------------------------------------------------------- shell
  function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
  function setMode(mode) {
    S.mode = mode; localSet("fg-mode", mode);
    $$(".mode button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === mode)));
    const ban = $("#banner");
    ban.hidden = mode !== "upi";
    ban.innerHTML = mode === "upi" ? "UPI mode: the organiser's UPI file has no learnable fraud signal. Every model scores at chance, so payments are approved by policy. Use this mode to see the evidence, not to stop fraud." : "";
  }
  async function route() {
    let r = location.hash.replace(/^#\/?/, "").split("?")[0] || "score";
    if (r === "main") return;   // skip-to-content link
    $$(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.route === r));
    $("#main").innerHTML = `<p class="loading">Loading the ${S.mode === "upi" ? "UPI" : "credit-card"} model…</p>`;
    await Promise.all([load(S.mode), load(S.mode === "upi" ? "creditcard" : "upi")]);
    (views[r] || views.score)();
    $("#main").focus({ preventScroll: true });
  }
  async function initAuth() {
    const cfg = window.FG_CONFIG || {};
    if (!cfg.SUPABASE_URL || !cfg.SUPABASE_ANON_KEY || !window.supabase) { document.body.classList.remove("signed-in"); return; }
    S.sb = window.supabase.createClient(cfg.SUPABASE_URL, cfg.SUPABASE_ANON_KEY);
    const { data } = await S.sb.auth.getSession(); S.user = data.session ? data.session.user : null;
    document.body.classList.toggle("signed-in", !!S.user);
    S.sb.auth.onAuthStateChange((_e, session) => { S.user = session ? session.user : null; document.body.classList.toggle("signed-in", !!S.user); });
  }
  document.addEventListener("DOMContentLoaded", async () => {
    $$(".mode button").forEach((b) => (b.onclick = () => { setMode(b.dataset.mode); route(); }));
    $("#nav-logout").onclick = async () => { if (S.sb) await S.sb.auth.signOut(); location.hash = "#score"; };
    setMode(S.mode);
    try { await initAuth(); } catch (e) { console.warn("Supabase unavailable", e); }
    window.addEventListener("hashchange", route);
    route();
  });
})();
