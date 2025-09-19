
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from catboost import CatBoostClassifier


df_test = pd.read_csv("german_credit_train.csv", nrows=50)
df_test.drop_duplicates(inplace = True)

df_test["Risk"] = df_test["Risk"].map({0: "No Risk", 1: "Risk"}).fillna(df_test["Risk"]).astype(str)

# partition by sex

def test_sex_proportions():
    counts = df_test.groupby(['Sex', "Risk"]).size().reset_index(name="count")
    totals = df_test.groupby('Sex').size().reset_index(name="total")
    sex_out = counts.merge(totals, on="Sex")
    sex_out["proportion"] = sex_out["count"] / sex_out["total"]

    assert sex_out["proportion"].between(0,1).all()

# partition by CreditHistory
def test_credit_proportions():
    credit_counts = df_test.groupby(['CreditHistory', 'Risk']).size().reset_index(name='credit_counts')
    total_credit_counts = df_test.groupby('CreditHistory').size().reset_index(name='total_credit_counts')
    credit_out = credit_counts.merge(total_credit_counts, on='CreditHistory')
    credit_out['proportion'] = (credit_out['credit_counts'] / credit_out['total_credit_counts'])
    
    assert credit_out["proportion"].between(0,1).all()

#partition by duration
def test_duration_proportions():
    duration_counts = df_test.groupby(['LoanPurpose', 'Risk']).size().reset_index(name='duration_counts')
    total_duration_counts = df_test.groupby('LoanPurpose').size().reset_index(name='total_duration_counts')
    duration_out = duration_counts.merge(total_duration_counts, on="LoanPurpose")
    duration_out['proportion'] = duration_out['duration_counts'] / duration_out['total_duration_counts']
    assert duration_out["proportion"].between(0,1).all()
    
# data modeling

def test_catboost_training():
    df_test['Risk'] = df_test['Risk'].map({'No Risk': 0, 'Risk': 1})
    X = df_test.drop(['Risk', 'ForeignWorker'], axis=1)
    y = df_test['Risk']
    
    numeric_cols = ['LoanDuration', 'LoanAmount', 'Age', 
                    'InstallmentPercent', 'CurrentResidenceDuration']
    category_cols = ['CheckingStatus', 'CreditHistory', 'LoanPurpose',
           'ExistingSavings', 'EmploymentDuration', 'OthersOnLoan', 
            'Sex', 'OwnsProperty', 'InstallmentPlans', 'Housing',
           'ExistingCreditsCount', 'Job', 'Dependents', 'Telephone']
    
    for c in category_cols:
        if c in X.columns:
            X[c] = X[c].astype('string')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                      test_size=0.2,
                                                      random_state=42,
                                                      stratify=y) 
    
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
    y_pred = (y_proba >= threshold).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["No risk", "Risk"])

    assert 0.0 <= acc <= 1.0
    assert isinstance(report, str) and "precision" in report

