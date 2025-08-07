import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score


def evaluate_utility(real_data, synthetic_data, target_col, return_both=False):
    """
    Evaluates the utility of synthetic vs real data using quick ML proxy task.
    Handles categorical columns in X.
    """

    # Split features/target
    X_real = real_data.drop(columns=[target_col])
    y_real = real_data[target_col]

    X_synth = synthetic_data.drop(columns=[target_col])
    y_synth = synthetic_data[target_col]

    # ✅ Encode non-numeric features
    X_real = pd.get_dummies(X_real)
    X_synth = pd.get_dummies(X_synth)

    # Align columns (same one-hot structure)
    X_real, X_synth = X_real.align(X_synth, join="outer", axis=1, fill_value=0)

    # Split train/test
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_synth, y_synth, test_size=0.2, random_state=42)

    # Task type: classification vs regression
    if y_real.dtype == 'object' or y_real.nunique() <= 20:
        model_real = RandomForestClassifier(random_state=42).fit(Xr_train, yr_train)
        model_synth = RandomForestClassifier(random_state=42).fit(Xs_train, ys_train)
        metric = accuracy_score
    else:
        model_real = RandomForestRegressor(random_state=42).fit(Xr_train, yr_train)
        model_synth = RandomForestRegressor(random_state=42).fit(Xs_train, ys_train)
        metric = r2_score

    real_score = metric(yr_test, model_real.predict(Xr_test))
    synth_score = metric(ys_test, model_synth.predict(Xs_test))

    return (real_score, synth_score) if return_both else (real_score + synth_score) / 2
