import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap

from perform_operation.features import FEATURES, engineer_training_features

model = None
explainer = None


def train_model():
    global model, explainer

    print("Loading dataset from OpenML (UCI Credit Default)...")

    # -----------------------------
    # Load dataset from OpenML
    # -----------------------------
    data = fetch_openml(
        data_id=42477,   # UCI Credit Card Default dataset
        as_frame=True
    )

    df = data.frame

    #  OpenML target column is `y`
    y = df["y"].astype(int)

    # -----------------------------
    # Feature engineering
    # -----------------------------
    X = engineer_training_features(df)

    # -----------------------------
    # Train / Test split
    # -----------------------------
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # Train XGBoost PD model
    # -----------------------------
    model = XGBClassifier(
    n_estimators=300,              # more trees, better learning
    max_depth=4,                   # keep trees shallow
    learning_rate=0.03,            # slower, more stable PD
    
    subsample=0.8,                 # row sampling
    colsample_bytree=0.8,           # feature sampling
    
    min_child_weight=5,             # prevents noisy splits
    gamma=1.0,                      # split only if useful
    
    reg_alpha=0.5,                  # L1 regularization
    reg_lambda=1.0,                 # L2 regularization
    
    objective="binary:logistic",
    eval_metric="logloss",
    
    tree_method="hist",             # faster + stable
    random_state=42
)


    model.fit(X_train, y_train)

    # -----------------------------
    # SHAP Explainer
    # -----------------------------
    explainer = shap.TreeExplainer(model)

    print("Model training completed successfully.")


def predict_pd(input_df):
    return float(model.predict_proba(input_df)[0][1]) * 100



def explain(input_df):
    shap_values = explainer.shap_values(input_df)
    return [
    (feature, float(value))
    for feature, value in sorted(
        zip(FEATURES, shap_values[0]),
        key=lambda x: abs(x[1]),
        reverse=True
    )
]

