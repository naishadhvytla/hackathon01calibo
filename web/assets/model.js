/* FraudGuard in-browser model runtime.
 * - Evaluates a CatBoost model exported with save_model(format="json") (oblivious trees).
 * - Computes exact TreeSHAP (Lundberg et al. 2018, path-dependent, algorithm 2) per tree.
 * - Reproduces the notebook's feature engineering for both datasets.
 * Works in the browser (window.FG) and in Node (module.exports) for verification tests.
 */
(function (root) {
  "use strict";

  // ---------- CatBoost oblivious trees -> explicit binary trees (for TreeSHAP) ----------
  function prepareModel(cb) {
    const trees = cb.oblivious_trees.map((t) => {
      const splits = t.splits.map((s) => ({ f: s.float_feature_index, b: s.border }));
      const d = splits.length;
      const L = t.leaf_values.length;           // 2^d
      const w = t.leaf_weights;
      // node arrays: internal nodes 0..L-2 (heap order by depth), leaves appended after
      // Build recursively: depth j splits on splits[j]; leaf index = sum(bit_j << j)
      const nodes = [];
      function build(j, idx) {
        const id = nodes.length;
        nodes.push(null);
        if (j === d) {
          nodes[id] = { leaf: true, value: t.leaf_values[idx], cover: Math.max(w[idx], 0) };
          return id;
        }
        const left = build(j + 1, idx);              // x <= border -> bit 0
        const right = build(j + 1, idx | (1 << j));   // x  > border -> bit 1
        nodes[id] = { leaf: false, f: splits[j].f, b: splits[j].b, left, right,
                      cover: nodes[left].cover + nodes[right].cover };
        return id;
      }
      build(0, 0);
      return { splits, values: t.leaf_values, nodes };
    });
    const sb = cb.scale_and_bias || [1, [0]];
    const bias = Array.isArray(sb[1]) ? sb[1][0] : sb[1];
    return { trees, scale: sb[0], bias };
  }

  function rawScore(model, x) {
    let s = 0;
    for (const t of model.trees) {
      let idx = 0;
      for (let j = 0; j < t.splits.length; j++) if (x[t.splits[j].f] > t.splits[j].b) idx |= 1 << j;
      s += t.values[idx];
    }
    return model.scale * s + model.bias;
  }
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));
  const predictProba = (model, x) => sigmoid(rawScore(model, x));

  // ---------- exact TreeSHAP (path-dependent) ----------
  function treeShap(tree, x, phi) {
    const nodes = tree.nodes;
    // path elements: {d: feature, z: zero fraction, o: one fraction, w: weight}
    function extend(path, pz, po, pi) {
      const l = path.length;
      path.push({ d: pi, z: pz, o: po, w: l === 0 ? 1 : 0 });
      for (let i = l - 1; i >= 0; i--) {
        path[i + 1].w += po * path[i].w * (i + 1) / (l + 1);
        path[i].w = pz * path[i].w * (l - i) / (l + 1);
      }
    }
    function unwind(path, i) {
      const l = path.length - 1;
      let n = path[l].w;
      const o = path[i].o, z = path[i].z;
      for (let j = l - 1; j >= 0; j--) {
        if (o !== 0) {
          const t = path[j].w;
          path[j].w = n * (l + 1) / ((j + 1) * o);
          n = t - path[j].w * z * (l - j) / (l + 1);
        } else {
          path[j].w = path[j].w * (l + 1) / (z * (l - j));
        }
      }
      for (let j = i; j < l; j++) { path[j].d = path[j + 1].d; path[j].z = path[j + 1].z; path[j].o = path[j + 1].o; }
      path.pop();
    }
    function unwoundSum(path, i) {
      const l = path.length - 1;
      const o = path[i].o, z = path[i].z;
      let total = 0, n = path[l].w;
      if (o !== 0) {
        for (let j = l - 1; j >= 0; j--) {
          const t = n / ((j + 1) * o);
          total += t;
          n = path[j].w - t * z * (l - j);
        }
      } else {
        for (let j = l - 1; j >= 0; j--) total += path[j].w / (z * (l - j));
      }
      return total * (l + 1);
    }
    function recurse(j, path, pz, po, pi) {
      path = path.map((p) => ({ ...p }));
      extend(path, pz, po, pi);
      const node = nodes[j];
      if (node.leaf) {
        for (let i = 1; i < path.length; i++) {
          const w = unwoundSum(path, i);
          phi[path[i].d] += w * (path[i].o - path[i].z) * node.value;
        }
        return;
      }
      const hot = x[node.f] > node.b ? node.right : node.left;
      const cold = hot === node.right ? node.left : node.right;
      const rc = node.cover > 0 ? nodes[hot].cover / node.cover : 0.5;
      const cc = node.cover > 0 ? nodes[cold].cover / node.cover : 0.5;
      let iz = 1, io = 1, k = 0;
      for (; k < path.length; k++) if (path[k].d === node.f) break;
      if (k !== path.length) { iz = path[k].z; io = path[k].o; unwind(path, k); }
      // a branch with zero mass under both "feature absent" (z) and "feature present" (o) contributes nothing:
      // skip it (oblivious trees reuse features, so this happens on the hot side too)
      if (iz * rc > 0 || io > 0) recurse(hot, path, iz * rc, io, node.f);
      if (iz * cc > 0) recurse(cold, path, iz * cc, 0, node.f);
    }
    recurse(0, [], 1, 1, -1);
  }
  function expectedValue(model) {
    // cover-weighted mean leaf value per tree (what TreeSHAP's baseline is)
    let s = 0;
    for (const t of model.trees) {
      const root0 = t.nodes[0]; let acc = 0;
      for (const n of t.nodes) if (n.leaf) acc += n.value * n.cover;
      s += root0.cover > 0 ? acc / root0.cover : 0;
    }
    return model.scale * s + model.bias;
  }
  function shapValues(model, x) {
    const phi = new Float64Array(x.length);
    for (const t of model.trees) treeShap(t, x, phi);
    for (let i = 0; i < phi.length; i++) phi[i] *= model.scale;
    return { phi: Array.from(phi), base: expectedValue(model) };
  }

  // ---------- feature engineering (mirrors src/fraudguard_lib.py) ----------
  function featuresCreditCard(row, spec) {
    const amount = +row.Amount, time = +row.Time;
    const hour = Math.floor(time / 3600) % 24;
    const f = {};
    for (let i = 1; i <= 28; i++) f["V" + i] = +row["V" + i];
    f.log_amount = Math.log1p(amount);
    f.amount_zero = amount === 0 ? 1 : 0;
    f.hour_sin = Math.sin(2 * Math.PI * hour / 24);
    f.hour_cos = Math.cos(2 * Math.PI * hour / 24);
    return spec.feature_order.map((k) => f[k]);
  }
  function featuresUPI(row, spec) {
    const hour = +row.hour_of_day;
    const f = {
      log_amount: Math.log1p(+row["amount (INR)"]),
      hour_sin: Math.sin(2 * Math.PI * hour / 24), hour_cos: Math.cos(2 * Math.PI * hour / 24),
      is_weekend: +row.is_weekend,
    };
    for (const [col, cats] of Object.entries(spec.cat_features))
      for (const c of cats) f[col + "_" + c] = String(row[col]) === c ? 1 : 0;
    return spec.feature_order.map((k) => (k in f ? f[k] : 0));
  }
  const featurize = (row, spec) => (spec.dataset === "creditcard" ? featuresCreditCard(row, spec) : featuresUPI(row, spec));

  // ---------- calibration, bands, policy search ----------
  function calibrate(spec, s) {               // isotonic = piecewise-linear interpolation, clipped
    const X = spec.calibrator.x, Y = spec.calibrator.y;
    if (s <= X[0]) return Y[0];
    if (s >= X[X.length - 1]) return Y[Y.length - 1];
    let lo = 0, hi = X.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (X[m] <= s) lo = m; else hi = m; }
    const t = (s - X[lo]) / (X[hi] - X[lo] || 1);
    return Y[lo] + t * (Y[hi] - Y[lo]);
  }
  function band(s, th) { return s >= th.T_block ? "BLOCK" : s >= th.T_otp ? "STEP-UP OTP" : "APPROVE"; }

  // cost of a (t_otp, t_block) policy on a slice {scores (desc-sorted top k), labels, n, n_fraud}
  function sliceCounts(sl, t) {        // frauds/genuine with score >= t (t must be >= sl.cutoff)
    let f = 0, g = 0;
    for (let i = 0; i < sl.scores.length && sl.scores[i] >= t; i++) sl.labels[i] ? f++ : g++;
    return { f, g };
  }
  function policyCost(sl, tO, tB, c) {
    const A = sliceCounts(sl, tO), B = sliceCounts(sl, tB);
    const fB = B.f, gB = B.g, fO = A.f - B.f, gO = A.g - B.g, fA = sl.n_fraud - A.f, gA = sl.n - sl.n_fraud - A.g;
    const cost = fA * c.X + fO * (c.Z + (1 - c.otp_stop_rate) * c.X) + gO * c.Z + gB * c.Y;
    return { cost, bands: [
      { band: "APPROVE", txns: fA + gA, frauds: fA, genuine: gA, cost: fA * c.X },
      { band: "STEP-UP OTP", txns: fO + gO, frauds: fO, genuine: gO, cost: fO * (c.Z + (1 - c.otp_stop_rate) * c.X) + gO * c.Z },
      { band: "BLOCK", txns: fB + gB, frauds: fB, genuine: gB, cost: gB * c.Y } ] };
  }
  // re-optimise both thresholds on the validation slice for any cost assumptions
  function optimisePolicy(spec, c) {
    const sl = spec.search.val;
    const cand = Array.from(new Set(sl.scores)).sort((a, b) => a - b);
    cand.push(2.0);                                // 2.0 = "never" (approve everything)
    // prefix counts over descending scores for O(1) lookups
    const desc = sl.scores, lab = sl.labels;
    const geF = new Map(), geG = new Map();
    let f = 0, g = 0, i = 0;
    for (const t of [...cand].reverse()) {
      while (i < desc.length && desc[i] >= t) { lab[i] ? f++ : g++; i++; }
      geF.set(t, f); geG.set(t, g);
    }
    let best = { cost: Infinity };
    for (let a = 0; a < cand.length; a++) {
      const tO = cand[a];
      for (let b = a; b < cand.length; b++) {
        const tB = cand[b];
        const fB = geF.get(tB), gB = geG.get(tB), fO = geF.get(tO) - fB, gO = geG.get(tO) - gB, fA = sl.n_fraud - geF.get(tO);
        const cost = fA * c.X + fO * (c.Z + (1 - c.otp_stop_rate) * c.X) + gO * c.Z + gB * c.Y;
        if (cost < best.cost) best = { cost, T_otp: tO, T_block: tB };
      }
    }
    // never cut below the slice cutoff (counts below it are not stored)
    best.T_otp = Math.max(best.T_otp, sl.cutoff); best.T_block = Math.max(best.T_block, best.T_otp);
    return best;
  }

  const FG = { prepareModel, rawScore, predictProba, shapValues, expectedValue, featurize, calibrate, band, policyCost, optimisePolicy, sigmoid };
  if (typeof module !== "undefined" && module.exports) module.exports = FG; else root.FG = FG;
})(typeof window !== "undefined" ? window : globalThis);
