import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from catboost import CatBoostClassifier


st.set_page_config(page_title="German Credit Risk ", layout="centered")

st.title("German Credit Risk")
train = pd.read_csv("german_credit_train.csv").drop_duplicates()

st.header("Train Data")
st.write(train)

train_counts = (
    train.groupby(["Sex", "Risk"])
      .size()
      .reset_index(name="count")
    .merge(train.groupby("Sex").size().reset_index(name="total"), on="Sex")
)
train_counts["proportion"] = train_counts["count"] / train_counts["total"]

# by sex
st.header("Partition by Sex")
st.subheader("(1) Aggregated proportions by Sex")
st.dataframe(train_counts, use_container_width=True)

st.subheader("(2) Bar chart — Proportion of Risk vs No Risk by Sex")
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=train_counts, x="Sex", y="proportion", hue="Risk", ax=ax)
ax.set_title("Proportion of Risk vs No Risk by Sex")
ax.set_ylabel("Proportion")
ax.set_ylim(0, 1)
st.pyplot(fig)

# by credit history

train_counts_history = (
    train.groupby(["CreditHistory", "Risk"])
      .size()
      .reset_index(name="count")
    .merge(train.groupby("CreditHistory").size().reset_index(name="total"), on="CreditHistory")
)
train_counts_history["proportion"] = train_counts_history["count"] / train_counts_history["total"]

st.header("Partition by CreditHistory")
st.subheader("(1) Aggregated proportions by CreditHistory")
st.dataframe(train_counts_history, use_container_width=True)

st.subheader("(2) Bar chart — Proportion of Risk vs No Risk by CreditHistory")
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=train_counts_history, x="CreditHistory", y="proportion", hue="Risk", ax=ax)
ax.set_title("Proportion of Risk vs No Risk by CreditHistory")
ax.set_ylabel("Proportion")
ax.set_ylim(0, 1)
st.pyplot(fig)

# by EmploymentDuration

train_counts_duration = (
    train.groupby(["EmploymentDuration", "Risk"])
      .size()
      .reset_index(name="count")
    .merge(train.groupby("EmploymentDuration").size().reset_index(name="total"), on="EmploymentDuration")
)
train_counts_duration["proportion"] = train_counts_duration["count"] / train_counts_duration["total"]

st.header("Partition by EmploymentDuration")
st.subheader("(1) Aggregated proportions by EmploymentDuration")
st.dataframe(train_counts_duration, use_container_width=True)

st.subheader("(2) Bar chart — Proportion of Risk vs No Risk by EmploymentDuration")
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=train_counts_duration, x="EmploymentDuration", y="proportion", hue="Risk", ax=ax)
ax.set_title("Proportion of Risk vs No Risk by EmploymentDuration")
ax.set_ylabel("Proportion")
ax.set_ylim(0, 1)
st.pyplot(fig)


st.header("Data Modeling")
st.text('Loading data...')

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

model_cat = CatBoostClassifier(
    iterations=500,
    depth=8,
    learning_rate=0.01,
    l2_leaf_reg=3,
    cat_features=cat_cols_in_X,
    verbose=False,
    random_seed=100
)
model_cat.fit(X_train, y_train)

y_proba = model_cat.predict_proba(X_test)[:, 1]
threshold = 0.75  
y_pred_custom = (y_proba >= threshold).astype(int)

acc = accuracy_score(y_test, y_pred_custom)
# st.subheader("Confusion Matrix")
# st.metric("Validation Accuracy", f"{acc:.3f}")

st.write("Using the CatBoost model, we achieved the following precision and recall metrics:")
report = classification_report(y_test, y_pred_custom,
                               target_names=["No risk", "Risk"])
st.code(report, language="text")

cm = confusion_matrix(y_test, y_pred_custom)
fig_cm, ax_cm = plt.subplots(figsize=(3, 2))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No risk", "Risk"]).plot(ax=ax_cm)
ax_cm.set_title("Confusion Matrix (threshold = 0.75)")
st.pyplot(fig_cm)

