"""
FraudGuard — Streamlit demo.
Pick (or edit) a transaction → risk score 0–100 → APPROVE / STEP-UP OTP / BLOCK, with the SHAP reasons.

Run:  streamlit run app/streamlit_app.py      (from the repo root, after running the notebook once)
"""
import os, sys
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import fraudguard_lib as L

st.set_page_config(page_title="FraudGuard", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_bundle():
    # NOTE: joblib/pickle can execute code on load — only load the bundle this repo's notebook produced.
    return joblib.load(os.path.join(ROOT, "models", "fraudguard_creditcard.joblib"))


@st.cache_data
def load_demo():
    return pd.read_csv(os.path.join(ROOT, "app", "demo_transactions.csv"))


B = load_bundle()
demo = load_demo()
costs = L.CostConfig(**B["costs"])


def model_input(raw: pd.DataFrame) -> pd.DataFrame:
    f = L.featurize_creditcard(raw)
    if B["is_hybrid"]:
        for name, col in B["hybrid_features"].items():
            f[col] = B["anomaly_detectors"][name].score(f)
    return f


def score(raw: pd.DataFrame) -> float:
    f = model_input(raw)
    if B.get("ensemble"):                       # soft-vote ensemble: mean member probability
        return float(np.mean([m.predict_proba(f)[:, 1][0] for m in B["ensemble"].values()]))
    return float(B["pipeline"].predict_proba(f)[:, 1][0])


def decide(s):
    if s >= B["T_block"]: return "BLOCK", "#D55E00"
    if s >= B["T_otp"]: return "STEP-UP OTP", "#E69F00"
    return "APPROVE", "#009E73"


# ── sidebar: pick a transaction ───────────────────────────────────────────────
st.sidebar.title("🛡️ FraudGuard")
st.sidebar.caption("Group 8 · Matrix Mechanics — N. Varshitha · M. Kumara Swamy · V. Naishadh · M. Suman Srinivas  \nLive web app: https://calibofraudguard.netlify.app/")
st.sidebar.caption(f"Model: **{B['final']}**  \nThresholds frozen on validation: T_otp={B['T_otp']:.4f}, T_block={B['T_block']:.4f}")
kind = st.sidebar.radio("Load a held-out test transaction", ["Random fraud", "Random genuine", "Pick by row"])
if "row" not in st.session_state or st.session_state.get("kind") != kind or st.sidebar.button("🎲 New random transaction"):
    st.session_state.kind = kind
    pool = demo.index[demo.Class == (1 if kind == "Random fraud" else 0)]
    st.session_state.row = int(np.random.choice(pool))
if kind == "Pick by row":
    st.session_state.row = st.sidebar.number_input("row", 0, len(demo) - 1, st.session_state.row)
base = demo.iloc[[st.session_state.row]].copy()

st.sidebar.markdown("---\n**Edit the transaction** (what-if)")
amount = st.sidebar.number_input("Amount (currency not stated in source)", 0.0, 30000.0, float(base.Amount.iloc[0]), step=1.0)
hour = st.sidebar.slider("Hour of day", 0, 23, int((base.Time.iloc[0] // 3600) % 24))
shap_top = ["V14", "V4", "V12", "V10", "V17", "V3"]
edits = {v: st.sidebar.slider(v, -25.0, 15.0, float(base[v].iloc[0]), 0.1) for v in shap_top}

raw = base.drop(columns=["Class"]).copy()
raw["Amount"] = amount
raw["Time"] = (raw.Time.iloc[0] // 86400) * 86400 + hour * 3600
for k, v in edits.items(): raw[k] = v

# ── main panel ────────────────────────────────────────────────────────────────
s = score(raw)
risk = 100 * float(B["calibrator"].predict([s])[0])
decision, colour = decide(s)
truth = "FRAUD" if base.Class.iloc[0] == 1 else "genuine"

st.title("Real-time payment risk decision")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Decision", decision)
c2.metric("Risk score (calibrated, 0–100)", f"{risk:.1f}")
c3.metric("Raw model score", f"{s:.5f}")
c4.metric("Actual label (held-out)", truth)
st.markdown(f"<div style='padding:14px;border-radius:8px;background:{colour};color:white;font-size:22px;font-weight:700'>"
            f"{decision}</div>", unsafe_allow_html=True)

exp_cost = {"APPROVE": f"if fraud: ₹{costs.X:,.0f} loss", "STEP-UP OTP": f"₹{costs.Z:,.0f} friction; OTP stops ~{costs.otp_stop_rate:.0%} of fraud",
            "BLOCK": f"if genuine: ₹{costs.Y:,.0f} customer cost"}[decision]
st.caption(f"Cost logic for this band — {exp_cost}. Costs are scenario assumptions (editable in the notebook config cell).")

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Why? — SHAP contributions (log-odds of fraud)")
    try:
        import shap
        pipe = B["shap_pipeline"]
        f = model_input(raw)
        Xp = pd.DataFrame(pipe.named_steps["pre"].transform(f), columns=pipe.named_steps["pre"].get_feature_names_out())
        ex = shap.TreeExplainer(pipe.named_steps["clf"], feature_perturbation="tree_path_dependent")(Xp)
        if ex.values.ndim == 3: ex = ex[..., 1]
        fig = plt.figure()
        shap.plots.waterfall(ex[0], max_display=10, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
        if B["explain_model"] != B["final"]:
            st.caption(f"Explained model: {B['explain_model']} (closest tree model to the final choice).")
    except Exception as e:  # SHAP is optional for the decision itself
        st.info(f"SHAP explanation unavailable: {e}")
with right:
    st.subheader("Policy")
    st.markdown(f"""
| Band | Rule | Action |
|---|---|---|
| 🟢 APPROVE | score < {B['T_otp']:.4f} | pass instantly |
| 🟠 STEP-UP OTP | {B['T_otp']:.4f} ≤ score < {B['T_block']:.4f} | extra verification |
| 🔴 BLOCK | score ≥ {B['T_block']:.4f} | decline + alert |
""")
    st.markdown("**Transaction (edited)**")
    st.dataframe(raw[["Time", "Amount"] + shap_top].T.rename(columns={raw.index[0]: "value"}))

st.markdown("---")
st.caption("Demo only. Credit-card data (ULB, 2013) with anonymised PCA features V1–V28; associations, not causes. "
           "Savings and costs are scenario estimates under assumed ₹X/₹Y/₹Z.")
