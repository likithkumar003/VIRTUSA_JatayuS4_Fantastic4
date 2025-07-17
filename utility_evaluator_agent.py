from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def evaluate_utility(real_data, synthetic_data, target_col, return_both=False):
    print(f"\n🧪 Evaluating utility using target column: {target_col}")

    real_data = real_data.copy()
    synthetic_data = synthetic_data.copy()

    # Auto classification or regression
    is_classification = pd.api.types.is_integer_dtype(real_data[target_col]) or pd.api.types.is_bool_dtype(real_data[target_col])

    # Handle timestamps (if present)
    if "Last_Period_Date" in real_data.columns and "Next_Period_1" in real_data.columns:
        real_data["Next_Period_1"] = (pd.to_datetime(real_data["Next_Period_1"]) - pd.to_datetime(real_data["Last_Period_Date"])).dt.days
        synthetic_data["Next_Period_1"] = (pd.to_datetime(synthetic_data["Next_Period_1"]) - pd.to_datetime(synthetic_data["Last_Period_Date"])).dt.days

    drop_cols = [target_col, "User_ID", "Last_Period_Date", "Next_Period_1", "Next_Period_2", "Next_Period_3", "Ovulation_Start", "Ovulation_End"]
    drop_cols = [col for col in drop_cols if col in real_data.columns]

    # Preprocessing
    X_real = pd.get_dummies(real_data.drop(columns=drop_cols, errors="ignore"), drop_first=True)
    y_real = real_data[target_col]
    X_synth = pd.get_dummies(synthetic_data.drop(columns=drop_cols, errors="ignore"), drop_first=True)
    y_synth = synthetic_data[target_col]

    # Train-Test Split
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_real, y_real, test_size=0.3, random_state=42)
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_synth, y_synth, test_size=0.3, random_state=42)

    if is_classification:
        model_real = RandomForestClassifier().fit(Xr_train, yr_train)
        model_synth = RandomForestClassifier().fit(Xs_train, ys_train)
        real_score = accuracy_score(yr_test, model_real.predict(Xr_test))
        synth_score = accuracy_score(ys_test, model_synth.predict(Xs_test))
        print(f"[📈] Real Accuracy:     {real_score:.2f}")
        print(f"[📉] Synthetic Accuracy:{synth_score:.2f}")
    else:
        model_real = RandomForestRegressor().fit(Xr_train, yr_train)
        model_synth = RandomForestRegressor().fit(Xs_train, ys_train)
        real_score = r2_score(yr_test, model_real.predict(Xr_test))
        synth_score = r2_score(ys_test, model_synth.predict(Xs_test))
        print(f"[📈] Real R2 Score:     {real_score:.2f}")
        print(f"[📉] Synthetic R2 Score:{synth_score:.2f}")

   

    if return_both:
        return real_score, synth_score
    else:
        return synth_score
