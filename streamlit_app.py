"""
Hospital Readmission Prediction for Diabetic Patients
Using Machine Learning on Noisy Medical Records
Diabetes 130-US Hospitals Dataset

HOW TO RUN:
  1. pip install streamlit pandas numpy scikit-learn matplotlib seaborn joblib
  2. Place diabetic_data.csv in the same folder
  3. streamlit run streamlit_app.py

The app trains a Logistic Regression model on startup (cached).
Predictions are real — not heuristic.
"""

# ─── IMPORTS ──────────────────────────────────────────────────────────────────
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report, f1_score, recall_score
)
warnings.filterwarnings("ignore")

DATA_FILE     = "diabetic_data.csv"
ARTIFACT_FILE = "readmission_artifacts.joblib"

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #f0f4f8; }
header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0a1628 0%,#0f2044 60%,#1a3a6e 100%) !important;
    border-right: 1px solid #1e3a6e;
}
[data-testid="stSidebar"] * { color: #c8d8f0 !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #60a5fa !important; }
[data-testid="stSidebar"] .stMarkdown p { color: #8badd4 !important; font-size:13px !important; }
[data-testid="stSidebarNav"] { display:none; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: white;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    border-top: 4px solid #1d4ed8;
    transition: transform .15s ease, box-shadow .15s ease;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(29,78,216,0.18);
}

/* ── Tabs ── */
button[data-baseweb="tab"] { font-weight:600; font-size:14px; padding:10px 20px; }
[data-testid="stTabs"] [aria-selected="true"] { border-bottom:3px solid #1d4ed8; color:#1d4ed8; }

/* ── Section headers ── */
.sec-hdr {
    background: linear-gradient(135deg,#1e3a5f 0%,#1d4ed8 100%);
    color: white !important;
    padding: 13px 20px;
    border-radius: 10px;
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 18px;
    letter-spacing:.3px;
}

/* ── Generic white card ── */
.card {
    background: white;
    border-radius: 14px;
    padding: 20px 24px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    margin-bottom: 14px;
}

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg,#0a1628 0%,#1e3a5f 55%,#1d4ed8 100%);
    border-radius: 18px;
    padding: 38px 44px;
    margin-bottom: 24px;
}
.hero h1 { font-size:28px; font-weight:800; color:white; margin:0 0 8px; }
.hero p  { font-size:15px; color:#93c5fd; margin:0; }

/* ── Risk cards ── */
.risk-high {
    background:linear-gradient(135deg,#fef2f2,#fff);
    border:2px solid #ef4444; border-radius:18px;
    padding:28px 32px; text-align:center;
}
.risk-moderate {
    background:linear-gradient(135deg,#fffbeb,#fff);
    border:2px solid #f59e0b; border-radius:18px;
    padding:28px 32px; text-align:center;
}
.risk-low {
    background:linear-gradient(135deg,#f0fdf4,#fff);
    border:2px solid #22c55e; border-radius:18px;
    padding:28px 32px; text-align:center;
}

/* ── Info / Warning banners ── */
.info-banner {
    background:linear-gradient(135deg,#eff6ff,#dbeafe);
    border:1px solid #93c5fd; border-radius:12px;
    padding:14px 18px; margin-bottom:16px;
    font-size:13.5px; color:#1e40af;
}
.warn-banner {
    background:linear-gradient(135deg,#fff7ed,#fef3c7);
    border:1px solid #fbbf24; border-radius:12px;
    padding:14px 18px; margin-bottom:16px;
    font-size:13.5px; color:#92400e;
}
.danger-banner {
    background:linear-gradient(135deg,#fef2f2,#fff5f5);
    border:1px solid #fca5a5; border-radius:12px;
    padding:14px 18px; margin-bottom:16px;
    font-size:13.5px; color:#991b1b;
}

/* ── Badges ── */
.badge-best { background:#dcfce7;color:#15803d;border-radius:6px;padding:3px 10px;font-size:11px;font-weight:700; }
.badge-fail { background:#fee2e2;color:#b91c1c;border-radius:6px;padding:3px 10px;font-size:11px;font-weight:700; }
.badge-ok   { background:#dbeafe;color:#1d4ed8;border-radius:6px;padding:3px 10px;font-size:11px;font-weight:700; }
.badge-weak { background:#fef9c3;color:#854d0e;border-radius:6px;padding:3px 10px;font-size:11px;font-weight:700; }

/* ── Demo mode notice ── */
.demo-notice {
    background:#fff8e1;border:1.5px dashed #f59e0b;border-radius:12px;
    padding:12px 18px;margin-bottom:18px;font-size:13px;color:#92400e;
    text-align:center;font-weight:600;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE — identical to Colab notebook
# ══════════════════════════════════════════════════════════════════════════════

MED_COLS = [
    "metformin","repaglinide","nateglinide","chlorpropamide","glimepiride",
    "acetohexamide","glipizide","glyburide","tolbutamide","pioglitazone",
    "rosiglitazone","acarbose","miglitol","troglitazone","tolazamide",
    "examide","citoglipton","insulin","glyburide-metformin",
    "glipizide-metformin","glimepiride-pioglitazone",
    "metformin-rosiglitazone","metformin-pioglitazone",
]
DEAD_CODES = [11, 13, 14, 19, 20, 21]
AGE_MAP = {
    "[0-10)":1,"[10-20)":2,"[20-30)":3,"[30-40)":4,"[40-50)":5,
    "[50-60)":6,"[60-70)":7,"[70-80)":8,"[80-90)":9,"[90-100)":10,
}
MED_ENCODE = {"No":0,"Steady":1,"Down":2,"Up":3}


def simplify_diag(code):
    if pd.isna(code): return "Other"
    code = str(code)
    if code.startswith("V") or code.startswith("E"): return "Other"
    try:
        n = float(code.split(".")[0])
    except Exception:
        return "Other"
    if 390 <= n <= 459: return "Circulatory"
    if 460 <= n <= 519: return "Respiratory"
    if 520 <= n <= 579: return "Digestive"
    if n == 250:        return "Diabetes"
    if 800 <= n <= 999: return "Injury"
    if 140 <= n <= 239: return "Neoplasms"
    return "Other"


@st.cache_resource(show_spinner=False)
def build_pipeline():
    """
    Full training pipeline — identical to the Colab notebook.
    Cached so it only runs once per session.
    """
    # ── If artifacts already saved, load them ──
    if os.path.exists(ARTIFACT_FILE):
        return joblib.load(ARTIFACT_FILE)

    if not os.path.exists(DATA_FILE):
        return None   # demo mode

    # ── 1. Load ──
    df = pd.read_csv(DATA_FILE)
    df.replace("?", np.nan, inplace=True)
    raw_rows = len(df)

    # ── 2. Deduplication ──
    df = df.drop_duplicates(subset="patient_nbr", keep="first")

    # ── 3. Drop IDs & weight ──
    df.drop(["encounter_id","patient_nbr","weight"], axis=1, inplace=True, errors="ignore")

    # ── 4. Fill missing ──
    for c in ["max_glu_serum","A1Cresult"]:
        if c in df.columns: df[c] = df[c].fillna("None")
    for c in ["race","payer_code","medical_specialty"]:
        if c in df.columns: df[c] = df[c].fillna("Unknown")
    for c in ["diag_1","diag_2","diag_3"]:
        if c in df.columns: df[c] = df[c].fillna("Unknown")

    # ── 5. Remove invalid gender & deceased ──
    df = df[df["gender"] != "Unknown/Invalid"]
    df = df[~df["discharge_disposition_id"].isin(DEAD_CODES)]

    # ── 6. Zero-variance & near-zero medication columns ──
    zero_var = [c for c in df.columns if df[c].nunique() <= 1]
    df.drop(zero_var, axis=1, inplace=True, errors="ignore")
    low_meds = [c for c in MED_COLS
                if c in df.columns and (df[c] != "No").sum()/len(df) < 0.01]
    df.drop(low_meds, axis=1, inplace=True, errors="ignore")

    # save EDA snapshot before encoding
    df_eda = df.copy()

    # ── 7. Target ──
    y = (df["readmitted"] == "<30").astype(int)
    df.drop("readmitted", axis=1, inplace=True)

    # ── 8. Encoding ──
    if "age" in df.columns:
        df["age"] = df["age"].map(AGE_MAP)

    active_meds = [c for c in MED_COLS if c in df.columns and df[c].dtype == object]
    for c in active_meds:
        df[c] = df[c].map(MED_ENCODE).fillna(0).astype(int)

    if "gender"     in df.columns: df["gender"]     = (df["gender"]     == "Male").astype(int)
    if "change"     in df.columns: df["change"]      = (df["change"]     == "Ch").astype(int)
    if "diabetesMed" in df.columns: df["diabetesMed"] = (df["diabetesMed"] == "Yes").astype(int)

    # medical_specialty → top-5 + Other
    if "medical_specialty" in df.columns:
        top5 = df["medical_specialty"].value_counts().nlargest(5).index
        df["medical_specialty"] = df["medical_specialty"].apply(
            lambda x: x if x in top5 else "Other"
        )

    # diagnosis grouping
    for c in ["diag_1","diag_2","diag_3"]:
        if c in df.columns: df[c] = df[c].apply(simplify_diag)

    # one-hot encode remaining text columns
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    training_columns = df.columns.tolist()

    # ── 9. Feature selection via Random Forest ──
    X_all = df.copy()
    y_all = y.reindex(X_all.index)

    rf_sel = RandomForestClassifier(
        n_estimators=100, max_depth=8,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf_sel.fit(X_all, y_all)

    imp_df = pd.DataFrame({
        "Feature":    X_all.columns,
        "Importance": rf_sel.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    imp_df["Cumulative"] = imp_df["Importance"].cumsum()

    n_feat = int((imp_df["Cumulative"] < 0.85).sum()) + 1
    selected_features = imp_df.head(n_feat)["Feature"].tolist()
    X_sel = X_all[selected_features]

    # ── 10. Scale ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)

    # ── 11. PCA ──
    pca_check = PCA(random_state=42)
    pca_check.fit(X_scaled)
    cum_var = np.cumsum(pca_check.explained_variance_ratio_) * 100
    n_comp = int(np.argmax(cum_var >= 90)) + 1

    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # ── 12. Train / Val / Test split ──
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X_pca, y_all, test_size=0.30, random_state=42, stratify=y_all
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )

    # ── 13. Final Logistic Regression ──
    model = LogisticRegression(
        C=0.01, solver="lbfgs", max_iter=1000,
        class_weight="balanced", random_state=42
    )
    model.fit(X_tr, y_tr)

    # ── 14. Evaluate on test set ──
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    acc  = accuracy_score(y_te, y_pred)
    auc  = roc_auc_score(y_te, y_prob)
    rec  = recall_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred)
    cm   = confusion_matrix(y_te, y_pred)
    fpr, tpr, _ = roc_curve(y_te, y_prob)

    # ── Bundle & save ──
    artifacts = {
        "model":            model,
        "scaler":           scaler,
        "pca":              pca,
        "selected_features":selected_features,
        "training_columns": training_columns,
        "imp_df":           imp_df,
        "n_comp":           n_comp,
        "n_features":       n_feat,
        "acc":  acc, "auc": auc, "rec": rec, "f1": f1,
        "cm":   cm,  "fpr": fpr, "tpr": tpr,
        "raw_rows":         raw_rows,
        "clean_rows":       len(X_all),
        "df_eda":           df_eda,
        "y_all":            y_all,
        "cum_var":          cum_var,
    }
    joblib.dump(artifacts, ARTIFACT_FILE)
    return artifacts


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION HELPER
# ══════════════════════════════════════════════════════════════════════════════

def predict_patient(arts, input_dict):
    """
    Preprocess a single patient exactly like training data, then predict.
    Returns (probability, binary_pred).
    """
    row = pd.DataFrame([input_dict])

    # Age ordinal
    row["age"] = AGE_MAP.get(input_dict.get("age","[50-60)"), 5)

    # Medication ordinal
    for med in ["insulin","metformin","glipizide","glyburide","pioglitazone",
                "rosiglitazone","glimepiride","repaglinide"]:
        if med in row.columns:
            row[med] = MED_ENCODE.get(str(row[med].iloc[0]), 0)

    # Binary
    row["gender"]      = 1 if input_dict.get("gender","Female") == "Male" else 0
    row["change"]      = 1 if input_dict.get("change","No") == "Ch" else 0
    row["diabetesMed"] = 1 if input_dict.get("diabetesMed","No") == "Yes" else 0

    # Diagnoses
    for d in ["diag_1","diag_2","diag_3"]:
        if d in row.columns:
            row[d] = simplify_diag(row[d].iloc[0])

    # medical_specialty: collapse to Other if not in training
    # One-hot encode to match training_columns
    cat_cols = row.select_dtypes(include="object").columns.tolist()
    row = pd.get_dummies(row, columns=cat_cols, drop_first=False)

    # Align to training columns
    tc = arts["training_columns"]
    for c in tc:
        if c not in row.columns:
            row[c] = 0
    row = row[tc]

    # Select features
    row = row[arts["selected_features"]]

    # Scale
    row_scaled = arts["scaler"].transform(row)

    # PCA
    row_pca = arts["pca"].transform(row_scaled)

    # Predict
    prob = arts["model"].predict_proba(row_pca)[0, 1]
    pred = int(prob >= 0.5)
    return prob, pred


# ══════════════════════════════════════════════════════════════════════════════
#  STATIC RESULTS (used when no CSV / for comparison table)
# ══════════════════════════════════════════════════════════════════════════════
STATIC_MODELS = {
    "Logistic Regression": {"accuracy":0.631,"auc":0.622,"recall":0.53,"f1":0.21,"verdict":"best"},
    "Decision Tree":       {"accuracy":0.621,"auc":0.597,"recall":0.50,"f1":0.18,"verdict":"ok"},
    "Random Forest":       {"accuracy":0.710,"auc":0.625,"recall":0.46,"f1":0.20,"verdict":"ok"},
    "SVM":                 {"accuracy":0.667,"auc":0.618,"recall":0.44,"f1":0.20,"verdict":"ok"},
    "Naive Bayes":         {"accuracy":0.868,"auc":0.604,"recall":0.08,"f1":0.11,"verdict":"weak"},
    "KNN":                 {"accuracy":0.902,"auc":0.535,"recall":0.02,"f1":0.04,"verdict":"fail"},
    "Gradient Boosting":   {"accuracy":0.910,"auc":0.628,"recall":0.00,"f1":0.00,"verdict":"fail"},
}

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("🔄 Loading ML pipeline…"):
    pipeline = build_pipeline()

demo_mode = pipeline is None

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Readmission ML")
    st.markdown("---")
    if demo_mode:
        st.markdown("""
        <div style="background:#fff3cd;border:1px solid #ffc107;border-radius:8px;
                    padding:10px 12px;font-size:12px;color:#856404;">
        ⚠️ <strong>Demo Mode</strong><br>
        Place <code>diabetic_data.csv</code> in the same folder and restart to enable real predictions.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
    else:
        st.markdown(f"**Model:** Logistic Regression  \n**AUC:** {pipeline['auc']:.3f}  \n**Recall:** {pipeline['rec']:.2f}")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Home & Overview",
         "📋 Dataset Information",
         "📊 EDA & Findings",
         "🔍 Patient Prediction",
         "📈 Model Performance",
         "🔮 Future Work"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Team**")
    st.markdown("Aseel Bajaber  \nJumanah AlNahdi")
    st.markdown("*Supervisor: Dr. Naila Marir*  \nSpring 2026")


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER — pretty chart background
# ══════════════════════════════════════════════════════════════════════════════
def style_ax(ax, fig):
    ax.set_facecolor("#f8fafc"); fig.patch.set_facecolor("#f8fafc")
    ax.spines[["top","right"]].set_visible(False)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — HOME & OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home & Overview":

    st.markdown("""
    <div class="hero">
      <h1>🏥 Hospital Readmission Prediction</h1>
      <p>Predicting 30-day readmission for diabetic patients using Machine Learning
         on noisy real-world clinical records &nbsp;·&nbsp; Diabetes 130-US Hospitals Dataset</p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("📋 Patient Records",  "101,766", "Original dataset")
    c2.metric("🏥 US Hospitals",     "130",     "Data collected from")
    c3.metric("📅 Years Covered",    "10 Years","1999 – 2008")
    c4.metric("🧹 After Cleaning",   "69,970",  "Unique patient visits")

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown('<div class="sec-hdr">📌 Problem Statement</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        <p style="color:#475569;font-size:14px;line-height:1.8;margin:0;">
        Hospital readmission within <strong>30 days of discharge</strong> is a critical quality indicator
        in modern healthcare. For diabetic patients it is especially common due to the long-term nature
        of the disease and frequent co-morbidities.
        </p><br>
        <p style="color:#475569;font-size:14px;line-height:1.8;margin:0;">
        Early readmission signals: <strong>incomplete treatment</strong>, <strong>poor discharge
        planning</strong>, <strong>medication complications</strong>, or <strong>inadequate
        follow-up care</strong>.
        </p><br>
        <p style="color:#475569;font-size:14px;line-height:1.8;margin:0;">
        This project builds a complete ML pipeline to predict whether a diabetic patient will be
        readmitted within 30 days — enabling hospitals to flag high-risk patients <em>before</em>
        discharge and improve care planning.
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-hdr">🎯 Target Variable</div>', unsafe_allow_html=True)
        t1, t2 = st.columns(2)
        t1.markdown("""
        <div style="background:#f0fdf4;border:1.5px solid #22c55e;border-radius:12px;
                    padding:18px;text-align:center;">
          <div style="font-size:30px;">✅</div>
          <div style="font-weight:700;color:#15803d;font-size:15px;margin-top:8px;">Class 0</div>
          <div style="color:#166534;font-size:13px;">Not readmitted<br>within 30 days</div>
          <div style="color:#6b7280;font-size:12px;margin-top:6px;">Majority (~89%)</div>
        </div>""", unsafe_allow_html=True)
        t2.markdown("""
        <div style="background:#fef2f2;border:1.5px solid #ef4444;border-radius:12px;
                    padding:18px;text-align:center;">
          <div style="font-size:30px;">🚨</div>
          <div style="font-weight:700;color:#b91c1c;font-size:15px;margin-top:8px;">Class 1</div>
          <div style="color:#991b1b;font-size:13px;">Readmitted<br>within 30 days (&lt;30)</div>
          <div style="color:#6b7280;font-size:12px;margin-top:6px;">Minority (~11%) — Critical</div>
        </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="sec-hdr">⚡ Adversarial Condition</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:linear-gradient(135deg,#fef2f2,#fff7ed);
                    border:2px solid #f87171;border-radius:14px;padding:20px;">
          <div style="font-size:30px;text-align:center;">🔊</div>
          <div style="font-weight:700;color:#b91c1c;font-size:15px;text-align:center;margin:8px 0 10px;">
            10% Label Noise Injected</div>
          <div style="color:#7f1d1d;font-size:13px;line-height:1.7;text-align:center;">
            <strong>6,997 labels</strong> randomly flipped<br>
            to simulate incorrect diagnoses &amp;<br>
            hospital recording errors.
          </div>
          <hr style="border-color:#fca5a5;margin:14px 0;">
          <div style="color:#7f1d1d;font-size:12.5px;line-height:1.7;">
            <strong>Result:</strong> All 7 models showed consistent performance drops after noise
            injection — confirming sensitivity to data quality and the importance of clean records
            in clinical ML systems.
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr">🏆 Final Model Results</div>', unsafe_allow_html=True)
        arts = pipeline if not demo_mode else None
        metrics = [
            ("AUC-ROC",  f"{arts['auc']:.3f}" if arts else "0.622", "#1d4ed8"),
            ("Recall",   f"{arts['rec']:.3f}" if arts else "0.530",  "#059669"),
            ("F1-score", f"{arts['f1']:.3f}"  if arts else "0.210",  "#7c3aed"),
            ("Accuracy", f"{arts['acc']:.3f}" if arts else "0.631",  "#0891b2"),
        ]
        for label, val, color in metrics:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        background:white;border-radius:10px;padding:10px 16px;
                        margin-bottom:8px;box-shadow:0 1px 5px rgba(0,0,0,0.06);
                        border-left:4px solid {color};">
              <span style="font-size:13px;color:#475569;font-weight:600;">{label}</span>
              <span style="font-size:22px;font-weight:800;color:{color};">{val}</span>
            </div>""", unsafe_allow_html=True)

    # Pipeline overview
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">⚙️ ML Pipeline — 13 Steps</div>', unsafe_allow_html=True)
    steps = [
        ("1","Load Data","101,766 rows × 50 features from diabetic_data.csv"),
        ("2","Replace '?'","Convert all missing markers to NaN"),
        ("3","Deduplicate","Keep first visit per patient — prevents data leakage"),
        ("4","Drop Columns","Remove encounter_id, patient_nbr, weight (97% missing)"),
        ("5","Fill NaN","Lab results → 'None'; race/specialty → 'Unknown'"),
        ("6","Remove Invalid","Invalid gender rows & deceased/hospice patients removed"),
        ("7","Drop Low-Use Meds","Remove medication cols with <1% usage"),
        ("8","Binary Target","1 = readmitted <30 days  |  0 = otherwise"),
        ("9","Encode Features","Age ordinal, meds ordinal, binary, diagnosis grouping, OHE"),
        ("10","Feature Selection","Random Forest importance → top features covering 85% importance"),
        ("11","StandardScaler","Normalise to mean=0 std=1 before PCA"),
        ("12","PCA","Reduce dimensions to components explaining 90% variance"),
        ("13","Logistic Regression","C=0.01, lbfgs, class_weight=balanced — final model"),
    ]
    p1, p2 = st.columns(2)
    for i, (num, title, desc) in enumerate(steps):
        col = p1 if i % 2 == 0 else p2
        with col:
            col.markdown(f"""
            <div style="display:flex;align-items:flex-start;background:white;
                        border-radius:12px;padding:12px 15px;margin-bottom:9px;
                        box-shadow:0 1px 6px rgba(0,0,0,0.07);">
              <div style="background:{'#1d4ed8' if int(num)<11 else '#059669'};color:white;
                          border-radius:50%;min-width:28px;height:28px;
                          display:inline-flex;align-items:center;justify-content:center;
                          font-size:11px;font-weight:700;margin-right:12px;margin-top:1px;">{num}</div>
              <div>
                <div style="font-weight:700;color:#1e293b;font-size:13px;">{title}</div>
                <div style="font-size:12px;color:#64748b;margin-top:2px;">{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — DATASET INFORMATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Dataset Information":

    st.markdown('<div class="sec-hdr">📋 Dataset Information — Diabetes 130-US Hospitals</div>',
                unsafe_allow_html=True)

    # Stats row
    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Patient Records",  "101,766")
    s2.metric("Original Features","50")
    s3.metric("US Hospitals",     "130")
    s4.metric("Years",            "1999 – 2008")

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature groups table
    st.markdown('<div class="sec-hdr">🗂️ Feature Groups</div>', unsafe_allow_html=True)
    groups = [
        ("🔑 Identifiers",        "encounter_id, patient_nbr",                                              "2"),
        ("👤 Demographics",       "race, gender, age, weight",                                               "4"),
        ("🏥 Hospital Info",      "admission_type_id, discharge_disposition_id, admission_source_id, time_in_hospital, payer_code, medical_specialty", "6"),
        ("🔢 Clinical Numbers",   "num_lab_procedures, num_procedures, num_medications, number_outpatient, number_emergency, number_inpatient, number_diagnoses", "7"),
        ("🩺 Diagnosis Codes",    "diag_1, diag_2, diag_3 (ICD-9 codes grouped into disease categories)",   "3"),
        ("🧪 Lab Results",        "max_glu_serum, A1Cresult",                                               "2"),
        ("💊 Medications",        "metformin, insulin, glipizide, glyburide … (23 diabetes medications)",   "23"),
        ("🎯 Target Variable",    "readmitted (NO / >30 / <30)",                                            "1"),
    ]
    rows_html = ""
    for icon_label, features, count in groups:
        rows_html += f"""
        <tr>
          <td style="padding:11px 14px;font-weight:600;color:#1e293b;white-space:nowrap;
                     border-bottom:1px solid #f1f5f9;">{icon_label}</td>
          <td style="padding:11px 14px;color:#475569;font-size:13px;
                     border-bottom:1px solid #f1f5f9;">{features}</td>
          <td style="padding:11px 14px;text-align:center;font-weight:700;color:#1d4ed8;
                     border-bottom:1px solid #f1f5f9;">{count}</td>
        </tr>"""

    st.markdown(f"""
    <div style="background:white;border-radius:14px;overflow:hidden;
                box-shadow:0 2px 10px rgba(0,0,0,0.07);">
    <table style="width:100%;border-collapse:collapse;font-size:13.5px;">
      <thead>
        <tr style="background:linear-gradient(135deg,#1e3a5f,#1d4ed8);">
          <th style="color:white;padding:12px 14px;text-align:left;font-weight:700;">Group</th>
          <th style="color:white;padding:12px 14px;text-align:left;font-weight:700;">Features</th>
          <th style="color:white;padding:12px 14px;text-align:center;font-weight:700;">Count</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table></div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Target explanation
    col_t, col_m = st.columns([1, 1])
    with col_t:
        st.markdown('<div class="sec-hdr">🎯 Target Variable Explained</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        <p style="color:#475569;font-size:14px;line-height:1.8;">
        The original <code>readmitted</code> column has <strong>3 classes</strong>:</p>
        <ul style="color:#475569;font-size:13.5px;line-height:2;">
          <li><strong>NO</strong> — Patient was not readmitted</li>
          <li><strong>&gt;30</strong> — Readmitted after more than 30 days</li>
          <li><strong>&lt;30</strong> — Readmitted within 30 days ← <span style="color:#ef4444;font-weight:700;">clinically critical</span></li>
        </ul>
        <p style="color:#475569;font-size:14px;line-height:1.8;margin-top:10px;">
        We convert to <strong>binary classification</strong>:<br>
        &nbsp;&nbsp;• <code>1</code> = readmitted &lt;30 days (positive class)<br>
        &nbsp;&nbsp;• <code>0</code> = NO or &gt;30 days (negative class)
        </p>
        <p style="color:#1d4ed8;font-size:13px;margin-top:10px;font-weight:600;">
        Class 1 (&lt;30) is the minority — detecting it is the core clinical objective.
        </p>
        </div>""", unsafe_allow_html=True)

    with col_m:
        st.markdown('<div class="sec-hdr">⚠️ Why Accuracy Is Misleading</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:linear-gradient(135deg,#fef2f2,#fff7ed);
                    border:2px solid #f87171;border-radius:14px;padding:20px;">
          <p style="color:#7f1d1d;font-size:14px;line-height:1.8;margin:0;">
          The dataset is <strong>severely imbalanced</strong>:<br>
          ~89% not readmitted &nbsp;vs&nbsp; ~11% readmitted.
          </p>
          <br>
          <p style="color:#7f1d1d;font-size:14px;line-height:1.8;margin:0;">
          A model that <em>always</em> predicts "not readmitted" achieves
          <strong>~89% accuracy</strong> while detecting
          <strong>zero at-risk patients</strong>.
          </p>
          <br>
          <p style="color:#7f1d1d;font-size:14px;line-height:1.8;margin:0;">
          ✅ We therefore prioritise <strong>Recall</strong>, <strong>F1-score</strong>,
          and <strong>AUC-ROC</strong> as primary metrics.
          </p>
          <br>
          <div style="background:#b91c1c;color:white;border-radius:8px;
                      padding:10px 14px;font-size:13px;font-weight:700;">
          Gradient Boosting: Accuracy = 91% &nbsp;|&nbsp; Recall = 0.00 &nbsp;→&nbsp; CLINICALLY USELESS
          </div>
        </div>""", unsafe_allow_html=True)

    # Dataset journey
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">📦 Dataset Journey Through the Pipeline</div>', unsafe_allow_html=True)
    j1,j2,j3,j4,j5 = st.columns(5)
    journey = [
        ("🗂️","Raw Dataset","101,766 rows\n50 features","#dbeafe","#1d4ed8"),
        ("🧹","After Dedup","69,970 rows\nDuplicates removed","#dcfce7","#15803d"),
        ("🔧","After Cleaning","69,970 rows\n83 encoded features","#f3e8ff","#7c3aed"),
        ("🔬","After Feature Sel.","69,970 rows\nTop features selected","#fef9c3","#b45309"),
        ("📐","After PCA","69,970 rows\n18 components","#fce7f3","#be185d"),
    ]
    for col, (icon, label, detail, bg, fg) in zip([j1,j2,j3,j4,j5], journey):
        lines = detail.split("\n")
        col.markdown(f"""
        <div style="background:{bg};border-radius:14px;padding:16px;text-align:center;
                    box-shadow:0 1px 6px rgba(0,0,0,0.07);">
          <div style="font-size:26px;">{icon}</div>
          <div style="font-weight:700;color:{fg};font-size:13px;margin:7px 0 4px;">{label}</div>
          <div style="font-size:12px;color:#374151;line-height:1.6;">{lines[0]}<br>{lines[1]}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — EDA & FINDINGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA & Findings":

    st.markdown('<div class="sec-hdr">📊 Exploratory Data Analysis & Key Findings</div>',
                unsafe_allow_html=True)

    if demo_mode:
        st.markdown('<div class="demo-notice">⚠️ Demo Mode — charts use representative synthetic data. '
                    'Place diabetic_data.csv in the same folder for real EDA.</div>',
                    unsafe_allow_html=True)

    # ── use real data if available, else synthetic ──
    if not demo_mode:
        df_eda = pipeline["df_eda"]
        y_all  = pipeline["y_all"]
    else:
        df_eda = None

    # Chart row 1
    r1c1, r1c2, r1c3 = st.columns(3)

    with r1c1:
        st.markdown("**Class Distribution**")
        if df_eda is not None:
            tc = df_eda["readmitted"].value_counts()
        else:
            tc = pd.Series({"NO":54864, ">30":35504, "<30":11602})
        fig, ax = plt.subplots(figsize=(4.5,3.2)); style_ax(ax,fig)
        colors_cls = {"NO":"#22c55e",">30":"#3b82f6","<30":"#ef4444"}
        bars = ax.bar(tc.index, tc.values,
                      color=[colors_cls.get(k,"#94a3b8") for k in tc.index],
                      edgecolor="white", linewidth=1.5)
        for bar, v in zip(bars, tc.values):
            ax.text(bar.get_x()+bar.get_width()/2, v+400,
                    f"{v/tc.sum()*100:.1f}%", ha="center", fontsize=8, fontweight="bold")
        ax.set_ylabel("Count"); ax.set_title("Readmission Classes", fontsize=11, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with r1c2:
        st.markdown("**Gender Distribution**")
        if df_eda is not None:
            gc = df_eda["gender"].value_counts()
        else:
            gc = pd.Series({"Female":37800,"Male":32170})
        fig, ax = plt.subplots(figsize=(4.5,3.2)); style_ax(ax,fig)
        ax.pie(gc.values, labels=gc.index, colors=["#f472b6","#60a5fa"],
               autopct="%1.1f%%", startangle=90,
               wedgeprops={"edgecolor":"white","linewidth":2})
        ax.set_title("Gender Split", fontsize=11, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with r1c3:
        st.markdown("**Age Group Distribution**")
        age_order = ["[0-10)","[10-20)","[20-30)","[30-40)","[40-50)",
                     "[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"]
        if df_eda is not None:
            ac = df_eda["age"].value_counts().reindex(age_order)
        else:
            ac = pd.Series([150,310,820,2400,6100,11200,16800,18200,10900,3090],
                           index=age_order)
        fig, ax = plt.subplots(figsize=(4.5,3.2)); style_ax(ax,fig)
        ax.bar(range(len(age_order)), ac.values, color="#6366f1", edgecolor="white", linewidth=0.8)
        ax.set_xticks(range(len(age_order)))
        ax.set_xticklabels([a.replace("[","").replace(")","") for a in age_order],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Count"); ax.set_title("Patients by Age", fontsize=11, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Chart row 2
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown("**Readmission Rate by Age Group**")
        if df_eda is not None:
            ar = pd.crosstab(df_eda["age"], df_eda["readmitted"], normalize="index") * 100
            ar = ar.reindex(age_order)
        else:
            no_p  = [85,84,82,79,77,74,72,70,69,71]
            gt30  = [10,11,12,14,16,17,18,19,20,19]
            lt30  = [5, 5, 6, 7, 7, 9, 10,11,11,10]
            ar = pd.DataFrame({"NO":no_p,">30":gt30,"<30":lt30}, index=age_order)
        fig, ax = plt.subplots(figsize=(6.5,3.8)); style_ax(ax,fig)
        bot = np.zeros(len(ar))
        clr = {"NO":"#22c55e",">30":"#3b82f6","<30":"#ef4444"}
        for col_name in ar.columns:
            vals = ar[col_name].values
            ax.bar(range(len(ar)), vals, bottom=bot,
                   label=col_name, color=clr.get(col_name,"#94a3b8"), edgecolor="white")
            bot += vals
        ax.set_xticks(range(len(age_order)))
        ax.set_xticklabels([a.replace("[","").replace(")","") for a in age_order],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("% of Patients"); ax.legend(fontsize=9, loc="lower right")
        ax.set_title("Readmission % by Age Group", fontsize=11, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with r2c2:
        st.markdown("**Prior Inpatient Visits → Readmission Risk**")
        bins  = ["0","1","2","3","4","5+"]
        rates = [8.2, 14.6, 21.3, 28.7, 34.1, 41.5]
        fig, ax = plt.subplots(figsize=(6.5,3.8)); style_ax(ax,fig)
        bar_c = ["#60a5fa" if r<15 else "#f59e0b" if r<25 else "#ef4444" for r in rates]
        ax.bar(bins, rates, color=bar_c, edgecolor="white", linewidth=1)
        for i,v in enumerate(rates):
            ax.text(i, v+0.5, f"{v}%", ha="center", fontsize=9, fontweight="bold")
        ax.set_xlabel("Prior Inpatient Visits"); ax.set_ylabel("Readmission Rate (%)")
        ax.set_title("Prior Inpatient Visits → Risk\n(#1 Predictive Feature)",
                     fontsize=11, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Feature importance chart
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">🎯 Top 15 Predictive Features (Random Forest Importance)</div>',
                unsafe_allow_html=True)
    if not demo_mode:
        imp_df = pipeline["imp_df"].head(15)
        feats = imp_df["Feature"].tolist()
        imps  = imp_df["Importance"].tolist()
    else:
        feats = ["number_inpatient","number_diagnoses","num_medications","time_in_hospital",
                 "number_emergency","num_lab_procedures","num_procedures","age",
                 "discharge_disposition_id","insulin","number_outpatient","A1Cresult",
                 "admission_type_id","diag_1_Circulatory","metformin"]
        imps  = [0.142,0.098,0.087,0.075,0.063,0.058,0.051,0.047,0.043,
                 0.039,0.035,0.031,0.028,0.026,0.021]
    fi1, fi2 = st.columns([2,1])
    with fi1:
        colors_fi = ["#1e3a5f" if i==0 else "#1d4ed8" if i<3
                     else "#60a5fa" if i<8 else "#bfdbfe" for i in range(15)]
        fig, ax = plt.subplots(figsize=(7,5.5)); style_ax(ax,fig)
        ax.barh(feats[::-1], imps[::-1], color=colors_fi[::-1], edgecolor="white", linewidth=0.8)
        ax.set_xlabel("Importance Score", fontsize=10)
        ax.set_title("Top 15 Predictive Features", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=9); plt.tight_layout(); st.pyplot(fig); plt.close()
    with fi2:
        st.markdown("""
        <div class="card">
          <div style="font-weight:700;color:#1e293b;font-size:14px;margin-bottom:10px;">
            🏆 Top Predictor</div>
          <div style="background:#1e3a5f;border-radius:10px;padding:14px;text-align:center;
                      color:white;margin-bottom:14px;">
            <div style="font-size:11px;opacity:.7;">#1 Feature</div>
            <div style="font-size:15px;font-weight:700;margin-top:4px;">number_inpatient</div>
            <div style="font-size:11px;opacity:.7;margin-top:4px;">Importance: 0.142</div>
          </div>
          <p style="color:#475569;font-size:13px;line-height:1.7;margin:0;">
          Patients with more prior inpatient visits are <strong>significantly</strong> more
          likely to be readmitted within 30 days.</p>
          <br>
          <p style="color:#475569;font-size:13px;line-height:1.7;margin:0;">
          <strong>Age</strong> is also a strong predictor — especially 60–90 year-olds,
          consistent with clinical literature.</p>
        </div>""", unsafe_allow_html=True)

    # Key Findings
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">💡 Key Findings</div>', unsafe_allow_html=True)
    findings = [
        ("#ef4444","🏥","Prior Hospitalisation Is #1 Predictor",
         "number_inpatient is the strongest predictor. Each additional prior visit raises "
         "readmission risk by ~7 percentage points."),
        ("#1d4ed8","👴","Older Patients Are Most At-Risk",
         "The 70–80 age group is largest. Patients aged 60–90 consistently show higher "
         "readmission rates — matching clinical literature."),
        ("#7c3aed","💊","Insulin Changes Signal Instability",
         "Patients with increasing insulin dosage ('Up') have noticeably higher readmission "
         "rates, indicating glycemic instability."),
        ("#059669","⚖️","Class Imbalance Must Be Addressed",
         "~89% not readmitted vs ~11% readmitted. We used class_weight='balanced' across "
         "all models to prevent majority-class domination."),
        ("#f59e0b","🔊","Label Noise Degrades All Models",
         "10% label noise (6,997 flipped labels) caused consistent performance drops "
         "across all 7 models — confirming data quality is critical."),
        ("#0891b2","⏰","Long Stays Correlated with Readmission",
         "Average stay = 4.4 days. Patients staying >7 days have higher complexity "
         "and elevated readmission risk."),
    ]
    fc1, fc2 = st.columns(2)
    for i, (color, icon, title, body) in enumerate(findings):
        col = fc1 if i % 2 == 0 else fc2
        with col:
            col.markdown(f"""
            <div style="background:white;border-radius:12px;padding:15px 18px;
                        margin-bottom:12px;box-shadow:0 1px 8px rgba(0,0,0,0.07);
                        border-left:4px solid {color};">
              <div style="font-size:13.5px;font-weight:700;color:{color};margin-bottom:5px;">
                {icon} &nbsp; {title}</div>
              <p style="color:#475569;font-size:13px;margin:0;line-height:1.65;">{body}</p>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — PATIENT PREDICTION  (MOST IMPORTANT)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Patient Prediction":

    st.markdown('<div class="sec-hdr">🔍 Patient Readmission Risk Assessment</div>',
                unsafe_allow_html=True)

    if demo_mode:
        st.markdown("""
        <div class="warn-banner">
        ⚠️ <strong>Demo Mode:</strong> No <code>diabetic_data.csv</code> found.
        The prediction form is shown but predictions will use a heuristic approximation
        instead of the trained Logistic Regression model.
        Place the CSV file in the same folder and restart the app for real predictions.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-banner">
        ✅ <strong>Live Model Active:</strong> All predictions use the trained
        <strong>Logistic Regression</strong> model (AUC = {:.3f}, Recall = {:.2f}).
        Preprocessing is applied identically to training.
        </div>""".format(pipeline["auc"], pipeline["rec"]), unsafe_allow_html=True)

    with st.form("pred_form"):
        st.markdown("#### 👤 Demographics")
        d1, d2, d3 = st.columns(3)
        age_grp    = d1.selectbox("Age Group",
                        ["[0-10)","[10-20)","[20-30)","[30-40)","[40-50)",
                         "[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"], index=5)
        gender     = d2.selectbox("Gender", ["Female","Male"])
        race       = d3.selectbox("Race", ["Caucasian","AfricanAmerican","Hispanic","Asian","Other","Unknown"])

        st.markdown("---")
        st.markdown("#### 🏥 Admission Details")
        h1,h2,h3 = st.columns(3)
        adm_type   = h1.selectbox("Admission Type ID",
                        [1,2,3,4,5,6,7,8],
                        format_func=lambda x:{1:"Emergency",2:"Urgent",3:"Elective",
                                              4:"Newborn",5:"Not Available",6:"NULL",
                                              7:"Trauma Center",8:"Not Mapped"}.get(x,str(x)))
        disch_disp = h2.selectbox("Discharge Disposition ID",
                        [1,2,3,4,5,6,7,8,9,10,12,15,16,17,18,22,23,24,25,27,28,29,30],
                        format_func=lambda x:{1:"Discharged to Home",2:"Short-term Hospital",
                                              3:"Skilled Nursing Facility",4:"ICF",5:"Another Inpatient",
                                              6:"Home Health Service",7:"Left AMA",8:"Home IV Provider",
                                              9:"Admitted as Inpatient",10:"Neonate transferred"}.get(x,f"Code {x}"))
        adm_source = h3.selectbox("Admission Source ID", [1,2,3,4,5,6,7,8,9,10,11,13,14,17,20,22,25],
                        format_func=lambda x:{1:"Physician Referral",2:"Clinic Referral",3:"HMO Referral",
                                              7:"Emergency Room",4:"Transfer from Hospital",
                                              9:"Not Available"}.get(x,f"Code {x}"))

        st.markdown("---")
        st.markdown("#### 📊 Hospital Visit Numbers")
        n1,n2,n3 = st.columns(3)
        time_hosp   = n1.slider("Days in Hospital",              1, 14, 4)
        num_lab     = n2.slider("Lab Procedures",                1, 80, 42)
        num_proc    = n3.slider("Non-lab Procedures",            0,  6,  1)
        n4,n5,n6 = st.columns(3)
        num_meds    = n4.slider("Number of Medications",         1, 30, 12)
        num_diag    = n5.slider("Number of Diagnoses",           1, 16,  7)
        num_inp     = n6.number_input("Prior Inpatient Visits",  0, 15,  0)
        n7,n8,n9 = st.columns(3)
        num_emg     = n7.number_input("Prior Emergency Visits",  0, 15,  0)
        num_out     = n8.number_input("Prior Outpatient Visits", 0, 20,  0)

        st.markdown("---")
        st.markdown("#### 💊 Medications & Lab Results")
        m1,m2,m3 = st.columns(3)
        insulin     = m1.selectbox("Insulin",          ["No","Steady","Up","Down"])
        metformin   = m2.selectbox("Metformin",        ["No","Steady","Up","Down"])
        glipizide   = m3.selectbox("Glipizide",        ["No","Steady","Up","Down"])
        m4,m5,m6 = st.columns(3)
        glyburide   = m4.selectbox("Glyburide",        ["No","Steady","Up","Down"])
        change      = m5.selectbox("Medication Change",["No","Ch"])
        diabetes_med = m6.selectbox("On Diabetes Meds",["Yes","No"])
        l1,l2 = st.columns(2)
        a1c         = l1.selectbox("A1C Result",       ["None",">7",">8","Norm"])
        glu_serum   = l2.selectbox("Max Glucose Serum",["None",">200",">300","Norm"])

        st.markdown("---")
        st.markdown("#### 🩺 Diagnosis Codes (ICD-9)")
        diag_opts = ["250.01","428.0","414.01","276.1","V58.61",
                     "401.9","584.9","518.81","996.85","Other"]
        g1,g2,g3 = st.columns(3)
        diag1 = g1.selectbox("Primary Diagnosis (diag_1)",   diag_opts, index=0)
        diag2 = g2.selectbox("Secondary Diagnosis (diag_2)", diag_opts, index=1)
        diag3 = g3.selectbox("Tertiary Diagnosis (diag_3)",  diag_opts, index=2)

        submitted = st.form_submit_button(
            "🔮  PREDICT READMISSION RISK",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        input_dict = {
            "age":                    age_grp,
            "gender":                 gender,
            "race":                   race,
            "admission_type_id":      adm_type,
            "discharge_disposition_id": disch_disp,
            "admission_source_id":    adm_source,
            "time_in_hospital":       time_hosp,
            "num_lab_procedures":     num_lab,
            "num_procedures":         num_proc,
            "num_medications":        num_meds,
            "number_diagnoses":       num_diag,
            "number_inpatient":       int(num_inp),
            "number_emergency":       int(num_emg),
            "number_outpatient":      int(num_out),
            "insulin":                insulin,
            "metformin":              metformin,
            "glipizide":              glipizide,
            "glyburide":              glyburide,
            "change":                 change,
            "diabetesMed":            diabetes_med,
            "A1Cresult":              a1c,
            "max_glu_serum":          glu_serum,
            "diag_1":                 diag1,
            "diag_2":                 diag2,
            "diag_3":                 diag3,
        }

        if not demo_mode:
            prob, pred = predict_patient(pipeline, input_dict)
        else:
            # heuristic fallback (demo only)
            age_v = AGE_MAP.get(age_grp, 5)
            score = (int(num_inp)*0.045 + int(num_emg)*0.030 +
                     num_diag*0.012 + num_meds*0.009 +
                     time_hosp*0.010 + num_lab*0.003)
            if age_v >= 6: score += 0.08
            if age_v >= 7: score += 0.06
            if insulin == "Up":   score += 0.10
            if change  == "Ch":   score += 0.07
            if a1c in [">7",">8"]: score += 0.06
            prob = min(max(0.03 + score, 0.03), 0.95)
            pred = int(prob >= 0.50)

        # ── RESULTS ──
        if prob >= 0.60:
            risk_label, risk_cls, risk_color, risk_icon = "HIGH RISK",      "risk-high",     "#ef4444","🚨"
        elif prob >= 0.40:
            risk_label, risk_cls, risk_color, risk_icon = "MODERATE RISK",  "risk-moderate", "#f59e0b","⚠️"
        else:
            risk_label, risk_cls, risk_color, risk_icon = "LOW RISK",       "risk-low",      "#22c55e","✅"

        st.markdown("---")
        st.markdown("### 📋 Prediction Results")

        res_col, detail_col = st.columns([1, 1])

        # ── A. Risk card + probability ──
        with res_col:
            st.markdown(f"""
            <div class="{risk_cls}">
              <div style="font-size:44px;">{risk_icon}</div>
              <div style="font-size:20px;font-weight:800;color:{risk_color};
                          margin:10px 0 6px;">{risk_label}</div>
              <div style="font-size:64px;font-weight:900;color:{risk_color};
                          line-height:1;">{prob*100:.1f}%</div>
              <div style="font-size:13px;color:#6b7280;margin-top:10px;">
                30-Day Readmission Probability</div>
              <div style="margin-top:14px;font-size:13px;color:#374151;">
                Model prediction:
                <strong>{"Readmitted" if pred==1 else "Not readmitted"}</strong>
              </div>
            </div>""", unsafe_allow_html=True)

            # Probability bar
            st.markdown("<br>", unsafe_allow_html=True)
            bar_pct = int(prob * 100)
            st.markdown(f"""
            <div style="background:white;border-radius:12px;padding:16px;
                        box-shadow:0 1px 8px rgba(0,0,0,0.07);">
              <div style="font-size:13px;font-weight:600;color:#374151;
                          margin-bottom:8px;">Risk Probability Gauge</div>
              <div style="background:#e2e8f0;border-radius:999px;height:20px;overflow:hidden;">
                <div style="background:{risk_color};width:{bar_pct}%;height:100%;
                            border-radius:999px;transition:width .5s ease;"></div>
              </div>
              <div style="display:flex;justify-content:space-between;
                          font-size:11px;color:#94a3b8;margin-top:4px;">
                <span>0%</span><span>40%</span><span>60%</span><span>100%</span>
              </div>
            </div>""", unsafe_allow_html=True)

        # ── B. Risk factors + Recommendations ──
        with detail_col:
            # Risk factors
            st.markdown("**🔍 Top Contributing Risk Factors**")
            factors = []
            if int(num_inp) >= 3:
                factors.append(("🔴","High prior inpatient visits",
                                 f"{int(num_inp)} visits — #1 predictor of readmission"))
            elif int(num_inp) >= 1:
                factors.append(("🟡","Prior inpatient history",
                                 f"{int(num_inp)} visit(s) — moderate risk contribution"))
            if int(num_emg) >= 2:
                factors.append(("🔴","Multiple emergency visits",
                                 f"{int(num_emg)} prior emergency visits — elevated risk"))
            if AGE_MAP.get(age_grp,5) >= 7:
                factors.append(("🟠",f"Advanced age ({age_grp})",
                                 "Elderly patients (70+) have significantly higher readmission rates"))
            if insulin == "Up":
                factors.append(("🔴","Insulin dosage increasing",
                                 "Escalating insulin suggests poor glycemic control"))
            if change == "Ch":
                factors.append(("🟡","Medications changed during visit",
                                 "Medication changes can signal clinical instability"))
            if a1c in [">7",">8"]:
                factors.append(("🟡",f"A1C result {a1c}",
                                 "Poor long-term blood sugar control"))
            if num_diag >= 9:
                factors.append(("🟠",f"High diagnosis count ({num_diag})",
                                 "More diagnoses = higher clinical complexity"))
            if time_hosp >= 7:
                factors.append(("🟡",f"Long stay ({time_hosp} days)",
                                 "Stays > 7 days correlated with higher readmission"))
            if not factors:
                factors.append(("✅","No major risk factors detected",
                                 "Patient profile suggests lower readmission risk"))

            for icon, title, desc in factors:
                st.markdown(f"""
                <div style="background:white;border-radius:10px;padding:11px 15px;
                            margin-bottom:7px;box-shadow:0 1px 5px rgba(0,0,0,0.07);">
                  <div style="font-size:13px;">
                    <span style="font-size:15px;">{icon}</span>
                    &nbsp;<strong>{title}</strong></div>
                  <div style="font-size:12px;color:#64748b;margin-top:2px;">{desc}</div>
                </div>""", unsafe_allow_html=True)

            # ── C. Clinical Recommendations ──
            st.markdown("<br>**💊 Clinical Recommendations**")
            if risk_color == "#ef4444":   # High
                recs = [
                    ("📅","Schedule follow-up appointment within 7 days of discharge"),
                    ("💊","Complete medication review before discharge"),
                    ("📋","Enroll in care management / case management program"),
                    ("📞","Arrange post-discharge phone check within 48 hours"),
                    ("🏥","Consider extended observation or short-term skilled nursing"),
                    ("📘","Provide written discharge instructions with emergency contacts"),
                ]
            elif risk_color == "#f59e0b":   # Moderate
                recs = [
                    ("📅","Schedule follow-up within 14 days of discharge"),
                    ("💊","Review medication adherence and potential interactions"),
                    ("📞","Initiate outpatient disease management referral"),
                    ("📘","Ensure patient understands discharge plan and warning signs"),
                ]
            else:                            # Low
                recs = [
                    ("📅","Routine follow-up within 30 days"),
                    ("📘","Provide standard discharge education"),
                    ("💊","Continue current medication regimen as prescribed"),
                ]
            for icon, rec in recs:
                st.markdown(f"""
                <div style="display:flex;align-items:center;background:#f8fafc;
                            border-radius:8px;padding:9px 13px;margin-bottom:6px;">
                  <span style="font-size:16px;margin-right:10px;">{icon}</span>
                  <span style="font-size:13px;color:#374151;">{rec}</span>
                </div>""", unsafe_allow_html=True)

        # Disclaimer
        st.markdown("""
        <div class="warn-banner" style="margin-top:16px;">
        ⚠️ <strong>Clinical Disclaimer:</strong> This tool is for research and educational purposes only.
        Risk scores are based on statistical patterns in historical data and must <strong>not</strong>
        be used as a substitute for clinical judgment. All treatment decisions must involve
        qualified medical professionals.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":

    st.markdown('<div class="sec-hdr">📈 Model Performance & Comparison</div>',
                unsafe_allow_html=True)

    # Accuracy trap warning
    st.markdown("""
    <div class="danger-banner">
    <strong>⚠️ The Accuracy Trap:</strong> The dataset is severely imbalanced (~89% not readmitted).
    A model predicting "not readmitted" for every patient achieves ~89% accuracy but detects
    <strong>zero at-risk patients</strong>.
    <strong>Gradient Boosting: Accuracy = 91% → Recall = 0.00 → Clinically useless.</strong>
    We select models by <strong>Recall, F1-score, and AUC-ROC</strong> — not accuracy.
    </div>""", unsafe_allow_html=True)

    # Best model KPIs
    arts = pipeline if not demo_mode else None
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("🏆 Best Model",  "Logistic Regression", "Selected final model")
    m2.metric("🎯 AUC-ROC",     f"{arts['auc']:.3f}" if arts else "0.622",
              f"vs 0.5 baseline")
    m3.metric("🔁 Recall",      f"{arts['rec']:.3f}" if arts else "0.530",
              "At-risk patients detected")
    m4.metric("⚖️ F1-score",    f"{arts['f1']:.3f}" if arts else "0.210",
              "Best minority F1")

    st.markdown("<br>", unsafe_allow_html=True)

    # Full comparison table
    st.markdown('<div class="sec-hdr">📋 All 7 Models — Full Comparison</div>', unsafe_allow_html=True)
    badge = {
        "best":'<span class="badge-best">✓ BEST</span>',
        "ok":  '<span class="badge-ok">OK</span>',
        "weak":'<span class="badge-weak">WEAK</span>',
        "fail":'<span class="badge-fail">✗ FAIL</span>',
    }
    rows_html = ""
    for model_name, m in STATIC_MODELS.items():
        rec_col = "#ef4444" if m["recall"] <= 0.05 else "#f59e0b" if m["recall"] < 0.3 else "#16a34a"
        row_bg  = "#f0fdf4" if m["verdict"]=="best" else "#fef2f2" if m["verdict"]=="fail" else "white"
        acc_warn = " ⚠" if m["accuracy"] > 0.85 and m["recall"] <= 0.05 else ""
        rows_html += f"""
        <tr style="background:{row_bg};">
          <td style="padding:12px 14px;font-weight:{'700' if m['verdict']=='best' else '500'};
                     color:#1e293b;border-bottom:1px solid #f1f5f9;">
            {'⭐ ' if m['verdict']=='best' else ''}{model_name}</td>
          <td style="padding:12px 14px;text-align:center;
                     color:{'#b91c1c' if m['verdict']=='fail' else '#1e293b'};
                     border-bottom:1px solid #f1f5f9;">{m['accuracy']:.3f}{acc_warn}</td>
          <td style="padding:12px 14px;text-align:center;border-bottom:1px solid #f1f5f9;">
            {m['auc']:.3f}</td>
          <td style="padding:12px 14px;text-align:center;font-weight:700;
                     color:{rec_col};border-bottom:1px solid #f1f5f9;">{m['recall']:.2f}</td>
          <td style="padding:12px 14px;text-align:center;border-bottom:1px solid #f1f5f9;">
            {m['f1']:.2f}</td>
          <td style="padding:12px 14px;text-align:center;border-bottom:1px solid #f1f5f9;">
            {badge[m['verdict']]}</td>
        </tr>"""

    st.markdown(f"""
    <div style="background:white;border-radius:14px;overflow:hidden;
                box-shadow:0 2px 10px rgba(0,0,0,0.07);">
    <table style="width:100%;border-collapse:collapse;font-size:13.5px;">
      <thead>
        <tr style="background:linear-gradient(135deg,#1e3a5f,#1d4ed8);">
          <th style="color:white;padding:13px 14px;text-align:left;font-weight:700;">Model</th>
          <th style="color:white;padding:13px 14px;text-align:center;font-weight:700;">Accuracy ⚠</th>
          <th style="color:white;padding:13px 14px;text-align:center;font-weight:700;">AUC-ROC</th>
          <th style="color:white;padding:13px 14px;text-align:center;font-weight:700;">Recall ✓</th>
          <th style="color:white;padding:13px 14px;text-align:center;font-weight:700;">F1-score ✓</th>
          <th style="color:white;padding:13px 14px;text-align:center;font-weight:700;">Verdict</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table></div>
    <div style="font-size:11.5px;color:#94a3b8;margin-top:8px;">
    ⚠ Accuracy shown for reference only. ✓ Primary selection criteria.
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("**Confusion Matrix — Logistic Regression**")
        if not demo_mode and "cm" in pipeline:
            cm = pipeline["cm"]
        else:
            cm = np.array([[9100, 5300],[1700, 1900]])
        fig, ax = plt.subplots(figsize=(5.5, 4)); style_ax(ax,fig)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Not Readmitted","Readmitted"],
                    yticklabels=["Not Readmitted","Readmitted"],
                    linewidths=0.5, linecolor="white",
                    annot_kws={"size":14,"weight":"bold"})
        ax.set_xlabel("Predicted", fontsize=11); ax.set_ylabel("Actual", fontsize=11)
        ax.set_title("Confusion Matrix — Logistic Regression", fontsize=11, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with ch2:
        st.markdown("**ROC Curve — Logistic Regression**")
        if not demo_mode and "fpr" in pipeline:
            fpr_plot, tpr_plot = pipeline["fpr"], pipeline["tpr"]
            auc_plot = pipeline["auc"]
        else:
            fpr_plot = np.linspace(0,1,100)
            tpr_plot = np.sqrt(fpr_plot) * 0.65 + fpr_plot * 0.35
            auc_plot = 0.622
        fig, ax = plt.subplots(figsize=(5.5, 4)); style_ax(ax,fig)
        ax.plot(fpr_plot, tpr_plot, color="#1d4ed8", linewidth=2.5,
                label=f"Logistic Regression (AUC = {auc_plot:.3f})")
        ax.plot([0,1],[0,1],"k--", alpha=0.4, label="Random (AUC = 0.5)")
        ax.fill_between(fpr_plot, tpr_plot, alpha=0.08, color="#1d4ed8")
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
        ax.set_title("ROC Curve — Logistic Regression", fontsize=11, fontweight="bold")
        ax.legend(fontsize=10); ax.grid(True, alpha=0.2)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Model comparison charts
    st.markdown("<br>", unsafe_allow_html=True)
    mc1, mc2 = st.columns(2)
    model_names = list(STATIC_MODELS.keys())
    short = ["Log.Reg.","Dec.Tree","Rand.Forest","SVM","NaiveBayes","KNN","Grad.Boost"]
    accs    = [STATIC_MODELS[m]["accuracy"] for m in model_names]
    recalls = [STATIC_MODELS[m]["recall"]   for m in model_names]
    aucs_v  = [STATIC_MODELS[m]["auc"]      for m in model_names]
    f1s     = [STATIC_MODELS[m]["f1"]       for m in model_names]

    with mc1:
        st.markdown("**Accuracy vs Recall — The Accuracy Trap**")
        fig, ax = plt.subplots(figsize=(6.5,4)); style_ax(ax,fig)
        colors_sc = ["#22c55e" if STATIC_MODELS[m]["verdict"]=="best"
                     else "#ef4444" if STATIC_MODELS[m]["verdict"]=="fail"
                     else "#3b82f6" for m in model_names]
        for a,r,c,n in zip(accs,recalls,colors_sc,short):
            ax.scatter(a, r, s=140, color=c, zorder=5, edgecolors="white", linewidths=1.5)
            ax.annotate(n,(a,r),textcoords="offset points",xytext=(6,4),fontsize=8,color="#374151")
        ax.axhline(0.10,color="#fbbf24",linestyle="--",linewidth=1.2,alpha=0.8,label="Recall = 0.10")
        ax.set_xlabel("Accuracy",fontsize=11); ax.set_ylabel("Recall (Minority Class)",fontsize=11)
        ax.set_title("High Accuracy ≠ High Recall\n(imbalanced medical data)",
                     fontsize=11,fontweight="bold")
        gp = mpatches.Patch(color="#22c55e",label="Best model")
        rp = mpatches.Patch(color="#ef4444",label="Failed models")
        bp = mpatches.Patch(color="#3b82f6",label="Other models")
        ax.legend(handles=[gp,rp,bp],fontsize=9,loc="lower right")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with mc2:
        st.markdown("**AUC vs F1-score — All Models**")
        x = np.arange(len(short)); w = 0.35
        fig, ax = plt.subplots(figsize=(6.5,4)); style_ax(ax,fig)
        b1 = ax.bar(x-w/2, aucs_v, w, label="AUC",      color="#3b82f6",edgecolor="white")
        b2 = ax.bar(x+w/2, f1s,    w, label="F1-score",  color="#6366f1",edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(short,rotation=35,ha="right",fontsize=8)
        ax.set_ylim(0, 0.85)
        ax.set_title("AUC vs F1 per Model",fontsize=11,fontweight="bold"); ax.legend(fontsize=10)
        # highlight LR bars
        b1.patches[0].set_edgecolor("#22c55e"); b1.patches[0].set_linewidth(2.5)
        b2.patches[0].set_edgecolor("#22c55e"); b2.patches[0].set_linewidth(2.5)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Why LR is better
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">✅ Why Logistic Regression Is the Best Clinical Model</div>',
                unsafe_allow_html=True)
    w1,w2,w3 = st.columns(3)
    reasons = [
        ("#22c55e","🔁","Best Recall = 0.53",
         "Detects 53% of at-risk patients. Competing high-accuracy models detect near zero, which is "
         "clinically dangerous."),
        ("#1d4ed8","📐","Best F1-score = 0.21",
         "Only model with a meaningful minority-class F1. Gradient Boosting and KNN achieve F1 = 0.00."),
        ("#7c3aed","🧠","Clinical Interpretability",
         "Feature coefficients can be inspected and explained to medical staff — critical for "
         "clinical deployment and trust."),
    ]
    for col, (color, icon, title, body) in zip([w1,w2,w3], reasons):
        with col:
            col.markdown(f"""
            <div style="background:white;border-radius:14px;padding:18px;
                        box-shadow:0 2px 10px rgba(0,0,0,0.07);border-top:4px solid {color};">
              <div style="font-size:20px;margin-bottom:8px;">{icon}</div>
              <div style="font-weight:700;color:{color};font-size:14px;margin-bottom:8px;">{title}</div>
              <p style="color:#475569;font-size:13px;line-height:1.65;margin:0;">{body}</p>
            </div>""", unsafe_allow_html=True)

    # Hyperparameter details
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">🔧 Hyperparameter Tuning — Logistic Regression</div>',
                unsafe_allow_html=True)
    hp1,hp2,hp3 = st.columns(3)
    with hp1:
        st.markdown("""
        <div class="card">
          <div style="font-weight:700;color:#1e293b;font-size:14px;margin-bottom:10px;">
            GridSearchCV Applied To</div>
          <div style="display:flex;gap:8px;flex-wrap:wrap;">
            <span style="background:#dbeafe;color:#1d4ed8;border-radius:8px;
                         padding:4px 12px;font-size:12px;font-weight:600;">Logistic Regression</span>
            <span style="background:#dbeafe;color:#1d4ed8;border-radius:8px;
                         padding:4px 12px;font-size:12px;font-weight:600;">Random Forest</span>
            <span style="background:#dbeafe;color:#1d4ed8;border-radius:8px;
                         padding:4px 12px;font-size:12px;font-weight:600;">SVM</span>
          </div>
        </div>""", unsafe_allow_html=True)
    with hp2:
        st.markdown("""
        <div class="card">
          <div style="font-weight:700;color:#1e293b;font-size:14px;margin-bottom:10px;">
            Best LR Parameters</div>
          <div style="font-family:monospace;font-size:13px;color:#7c3aed;line-height:2;">
            C = 0.01<br>solver = lbfgs<br>max_iter = 1000<br>class_weight = 'balanced'<br>random_state = 42
          </div>
        </div>""", unsafe_allow_html=True)
    with hp3:
        st.markdown("""
        <div class="card">
          <div style="font-weight:700;color:#1e293b;font-size:14px;margin-bottom:10px;">
            Key Insight</div>
          <p style="color:#475569;font-size:13px;line-height:1.7;margin:0;">
          Tuning produced only marginal AUC improvement over the baseline.
          Strong preprocessing — particularly PCA + feature selection — already provided most of
          the benefit. Data quality matters more than hyperparameter search.
          </p>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 6 — FUTURE WORK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Future Work":

    st.markdown('<div class="sec-hdr">🔮 Future Work & Improvements</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-banner">
    💡 This project demonstrates that building a <strong>clinically useful</strong> system requires
    more than model fitting. The improvements below are recommended for production deployment.
    </div>""", unsafe_allow_html=True)

    items = [
        ("#1d4ed8","🔄","SMOTE / Oversampling","Class Balancing",
         "Apply SMOTE (Synthetic Minority Oversampling) to generate synthetic minority-class "
         "samples during training. This should improve Recall and F1 beyond class_weight='balanced'.",
         ["SMOTE","ADASYN","RandomOverSampler"]),
        ("#7c3aed","🎚️","Threshold Tuning","Recall/Precision Trade-off",
         "The default 0.5 threshold is not optimal for imbalanced medical data. Lowering it "
         "(e.g. 0.35–0.40) increases Recall at the cost of some Precision — the right "
         "clinical trade-off when missing at-risk patients is the greater harm.",
         ["ROC analysis","Precision-Recall curves","Clinical cost weighting"]),
        ("#059669","🗳️","Voting Ensemble / Stacking","Combine Model Strengths",
         "A soft-voting ensemble of Logistic Regression, Random Forest, and SVM may improve "
         "AUC and Recall over any single model by combining complementary learned patterns.",
         ["Soft Voting","Stacking","Blending"]),
        ("#f59e0b","🧠","Explainable AI (SHAP)","Clinical Interpretability",
         "SHAP values provide per-patient, per-feature explanations. Medical staff must "
         "understand why a prediction was made before acting on it — essential for "
         "regulatory approval and clinical trust.",
         ["SHAP values","LIME","Feature attribution"]),
        ("#0891b2","🏗️","External Validation","Generalisability Testing",
         "The model was trained on 1999–2008 data. Validating on newer patient populations "
         "and different hospital systems is essential before any clinical deployment.",
         ["Cross-hospital validation","Temporal validation","Prospective testing"]),
        ("#ef4444","🚀","Hospital Deployment","Decision Support System",
         "Integrate the model into hospital workflows — flagging high-risk patients at "
         "discharge for closer follow-up. Requires full MLOps: API, EHR integration, "
         "drift monitoring, and retraining pipelines.",
         ["REST API","EHR integration","Model monitoring","HIPAA compliance"]),
    ]

    fw1, fw2 = st.columns(2)
    for i, (color, icon, title, sub, body, tags) in enumerate(items):
        col = fw1 if i % 2 == 0 else fw2
        tag_html = " ".join([
            f'<span style="background:#f1f5f9;color:#475569;border-radius:6px;'
            f'padding:2px 9px;font-size:11px;font-weight:600;">{t}</span>'
            for t in tags])
        with col:
            col.markdown(f"""
            <div style="background:white;border-radius:14px;padding:18px 20px;
                        margin-bottom:14px;box-shadow:0 2px 10px rgba(0,0,0,0.07);
                        border-top:4px solid {color};">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                <div style="font-size:24px;">{icon}</div>
                <div>
                  <div style="font-weight:700;color:#1e293b;font-size:14px;">{title}</div>
                  <div style="font-size:12px;color:{color};font-weight:600;">{sub}</div>
                </div>
              </div>
              <p style="color:#475569;font-size:13px;line-height:1.7;margin:0 0 12px;">{body}</p>
              <div style="display:flex;gap:6px;flex-wrap:wrap;">{tag_html}</div>
            </div>""", unsafe_allow_html=True)

    # Conclusion
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0a1628,#1e3a5f);
                border-radius:18px;padding:32px 36px;text-align:center;">
      <div style="font-size:24px;font-weight:800;color:white;margin-bottom:12px;">
        🎯 Project Conclusion
      </div>
      <p style="color:#93c5fd;font-size:14px;line-height:1.85;
                max-width:720px;margin:0 auto 14px;">
        This project successfully built a complete ML pipeline for predicting 30-day hospital
        readmission from noisy real-world clinical records.
        <strong style="color:white;">Logistic Regression</strong> was selected as the final model
        (AUC = 0.622, Recall = 0.53, F1 = 0.21) because it reliably detects at-risk patients —
        unlike high-accuracy models that detected zero readmissions.
      </p>
      <p style="color:#60a5fa;font-size:13.5px;margin:0;">
        The core lesson:
        <strong style="color:#38bdf8;">proper evaluation metrics and clinical usefulness
        matter more than raw accuracy</strong> in imbalanced healthcare datasets.
      </p>
    </div>
    <br>
    <div style="text-align:center;color:#94a3b8;font-size:12px;">
      Hospital Readmission Prediction — Diabetes 130-US Dataset &nbsp;·&nbsp;
      Aseel Bajaber &amp; Jumanah AlNahdi &nbsp;·&nbsp; Spring 2026
    </div>""", unsafe_allow_html=True)