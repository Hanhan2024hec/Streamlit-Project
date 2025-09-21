import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay
)
from catboost import CatBoostClassifier


st.set_page_config(page_title="German Credit Risk ", page_icon= "🎯", layout="centered")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Explore", "Model & Predict",  "About"])

train = pd.read_csv("german_credit_train.csv").drop_duplicates()

def proportions_by(df: pd.DataFrame, by: str) -> pd.DataFrame:
    counts = df.groupby([by, "Risk"]).size().reset_index(name="count")
    totals = df.groupby(by).size().reset_index(name="total")
    out = counts.merge(totals, on=by)
    out["proportion"] = out["count"] / out["total"]
    return out

def render_confusion_matrix(y_true, y_pred, title: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    ConfusionMatrixDisplay(cm, display_labels=["No risk", "Risk"]).plot(ax=ax, colorbar=False)
    ax.set_title(title)
    st.pyplot(fig)

train['Risk'] = train['Risk'].map({'No Risk': 0, 'Risk': 1}).astype('Int64')
numeric_cols = ['LoanDuration', 'LoanAmount', 'Age',
                'InstallmentPercent', 'CurrentResidenceDuration']
category_cols = ['CheckingStatus', 'CreditHistory', 'LoanPurpose',
                 'ExistingSavings', 'EmploymentDuration', 'OthersOnLoan',
                 'Sex', 'OwnsProperty', 'InstallmentPlans', 'Housing',
                 'ExistingCreditsCount', 'Job', 'Dependents', 'Telephone']

X = train.drop(['Risk', 'ForeignWorker'], axis=1)
y = train['Risk']
for c in category_cols:
    if c in X.columns:
        X[c] = X[c].astype('string')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

cat_cols_in_X = [c for c in category_cols if c in X.columns]

# -------------------- PAGES --------------------
if page == "Overview":
    st.title("📊 German Credit Risk")
    st.markdown(
        """
        This app predicts whether a customer is **Risk** or **No Risk** for a lending system,
        based on features like **Gender**, **Age**, **Credit History**, **Employment Duration**, and more.
        
        **Model**: I used **CatBoostClassifier** for robust categorical handling and strong baseline performance.  
        **Evaluation**: Confusion matrix + precision/recall/F1, with an adjustable decision threshold.
        """)
    st.markdown("### Sample of the training data")
    st.dataframe(train.head(20), use_container_width=True)
    st.caption(f"Rows: {len(train):,}  •  Columns: {len(train.columns)}")

elif page == "Explore":
    st.title("🔎 Explore Risk Distributions")
    candidates = ["Sex", "CreditHistory", "EmploymentDuration"]
    by = st.selectbox("Choose a categorical feature", candidates, index=0)
    out = proportions_by(train, by)
    st.subheader(f"Aggregated proportions by {by}")
    st.dataframe(out, use_container_width=True)

    st.subheader(f"Bar chart — Proportion of Risk vs No Risk by {by}")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=out, x=by, y="proportion", hue="Risk", ax=ax)
    ax.set_ylabel("Proportion")
    ax.set_xlabel(by)
    ax.set_ylim(0, 1)
    st.pyplot(fig)


elif page == "Model & Predict":
    st.title("🤖 Train & Evaluate (CatBoost)")

    with st.sidebar:
        st.subheader("Model Controls")
        iterations = st.slider("Iterations", 100, 1000, 500)
        depth = st.select_slider("Depth", options=[4, 6, 8], value=8)
        lr = st.select_slider("Learning rate", options=[0.005, 0.01, 0.05], value=0.01)
        l2 = st.select_slider("L2 regularization", options=[1, 3, 5], value=3)
        threshold = st.slider("Decision threshold", 0.50, 0.95, 0.75)

    with st.spinner("Training CatBoost..."):
        model_cat = CatBoostClassifier(
        iterations=500,
        depth=depth,
        learning_rate=lr,
        l2_leaf_reg=l2,
        cat_features=cat_cols_in_X,
        verbose=False,
        random_seed=100
        )
        model_cat.fit(X_train, y_train)


    y_proba = model_cat.predict_proba(X_test)[:, 1]
    y_pred_custom = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred_custom)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_custom, average="binary", zero_division=0)
    # st.subheader("Confusion Matrix")
    # st.metric("Validation Accuracy", f"{acc:.3f}")

    st.subheader("✅ Key metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc:.3f}")
    m2.metric("Precision", f"{precision:.3f}")
    m3.metric("Recall", f"{recall:.3f}")
    m4.metric("F1-Score", f"{f1:.3f}")

    st.subheader("📊 Detailed classification report")
    st.code(classification_report(y_test, y_pred_custom, target_names=["No risk", "Risk"]), language="text")

    st.subheader("📑 Confusion matrix")
    render_confusion_matrix(y_test, y_pred_custom, f"Threshold = {threshold:.2f}")

    st.caption("Tip: Adjust the decision threshold to balance precision and recall.")

    st.markdown("---")

    st.subheader("🔮 Interactive Prediction")
    def opts(col):
        if col in train.columns:
            return sorted(train[col].astype("string").dropna().unique().tolist())
        return []
    
    num_defaults = {c: float(train[c].median()) for c in numeric_cols if c in train.columns}
    ncols = st.columns(2)
    num_values = {}
    for i, c in enumerate([c for c in numeric_cols if c in train.columns]):
        with ncols[i % 2]:
            num_values[c] = st.number_input(
                c, value=float(num_defaults[c]), step=1.0 if train[c].dtype.kind in "iu" else 0.1
            )
    
    extra_numeric = []
    for c in ["ExistingCreditsCount"]:
        if c in train.columns and c not in numeric_cols:
            extra_numeric.append(c)
            with ncols[(len(numeric_cols) + len(extra_numeric)) % 2]:
                num_values[c] = st.number_input(
                    c, value=float(train[c].median()), step=1.0
                )
    
    st.subheader("🧩 Categorical Features")
    cat_values = {}
    ccols = st.columns(3)
    present_cats = [c for c in category_cols if c in train.columns]
    for i, c in enumerate(present_cats):
        with ccols[i % 3]:
            choices = opts(c) or ["(missing)"]
            default_idx = 0
            cat_values[c] = st.selectbox(c, options=choices, index=default_idx, key=f"cat_{c}")
    
    row = {}

    for c in num_values:
        if c in X.columns:
            row[c] = num_values[c]

    for c in cat_values:
        if c in X.columns:
            row[c] = str(cat_values[c])

    for c in X.columns:
        if c not in row:
            if c in cat_cols_in_X:
                row[c] = str(train[c].astype("string").mode(dropna=True).iloc[0]) if c in train.columns else ""
            else:
            
                if c in train.columns and pd.api.types.is_numeric_dtype(train[c]):
                    row[c] = float(train[c].median())
                else:
                    row[c] = 0.0

    X_one = pd.DataFrame([row])[X.columns] 
    for c in cat_cols_in_X:
        if c in X_one.columns:
            X_one[c] = X_one[c].astype("string")


    proba = float(model_cat.predict_proba(X_one)[:, 1][0])
    pred = int(proba >= threshold)
    label = "Risk" if pred == 1 else "No Risk"


    col_l, col_r = st.columns([2,1])
    with col_l:
        st.markdown(f"## 🎯 Predicted Risk: **{label}**")
    with col_r:
        st.metric("Probability of Risk", f"{proba:.2%}", help="Model P(Risk | features)")
    st.caption(f"Decision threshold = {threshold:.2f} • Adjust it in the modeling section if exposed.")


elif page == "About":
    st.title("ℹ️ About this project")
    st.markdown(
        """
        **Goal**: Predict whether a customer is *Risk* or *No Risk* for a bank lending system  
        using features like **Gender**, **Age**, **Credit History**, **Employment Duration**, etc.

        **Why CatBoost?**  
        - Handles categorical variables natively  
        - Strong performance with minimal preprocessing  
 
        """
    )
    st.markdown("---")
    st.markdown("**Repo**: https://github.com/Hanhan2024hec/Streamlit-Project")
    st.markdown("**Docker Hub**: http://localhost:8600/")

