"""
Train XGBoost models (Energy + Bill) on the building dataset.
Uses data_prep_unified for consistent feature engineering.
"""
import sys
import os
import xgboost as xgb
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import json

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data_prep_unified import load_and_prepare, FEATURES

def main():
    print("=" * 60)
    print("  XGBoost Model Training (Building Dataset)")
    print("=" * 60)

    # Load data
    X_train, X_test, y_train, y_test, feature_names, le = load_and_prepare()

    # ---- Hyperparameters ----
    params = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
    }

    # ---- Train Energy Model ----
    print("\n--- Training Energy Model (XGBoost) ---")
    energy_model = xgb.XGBRegressor(**params)
    energy_model.fit(
        X_train, y_train['Energy_Consumption_kWh'],
        eval_set=[(X_test, y_test['Energy_Consumption_kWh'])],
        verbose=50
    )

    y_pred_energy = energy_model.predict(X_test)
    r2_e = r2_score(y_test['Energy_Consumption_kWh'], y_pred_energy)
    rmse_e = np.sqrt(mean_squared_error(y_test['Energy_Consumption_kWh'], y_pred_energy))
    mae_e = mean_absolute_error(y_test['Energy_Consumption_kWh'], y_pred_energy)
    print(f"  Energy R2:   {r2_e:.4f}")
    print(f"  Energy RMSE: {rmse_e:.2f}")
    print(f"  Energy MAE:  {mae_e:.2f}")

    # ---- Train Bill Model ----
    print("\n--- Training Bill Model (XGBoost) ---")
    bill_model = xgb.XGBRegressor(**params)
    bill_model.fit(
        X_train, y_train['Electricity_Bill_Amount'],
        eval_set=[(X_test, y_test['Electricity_Bill_Amount'])],
        verbose=50
    )

    y_pred_bill = bill_model.predict(X_test)
    r2_b = r2_score(y_test['Electricity_Bill_Amount'], y_pred_bill)
    rmse_b = np.sqrt(mean_squared_error(y_test['Electricity_Bill_Amount'], y_pred_bill))
    mae_b = mean_absolute_error(y_test['Electricity_Bill_Amount'], y_pred_bill)
    print(f"  Bill R2:   {r2_b:.4f}")
    print(f"  Bill RMSE: {rmse_b:.2f}")
    print(f"  Bill MAE:  {mae_b:.2f}")

    # ---- Save Models ----
    joblib.dump(energy_model, "energy_model_xgb.pkl")
    joblib.dump(bill_model, "bill_model_xgb.pkl")
    print("\n[OK] Models saved: energy_model_xgb.pkl, bill_model_xgb.pkl")

    # ---- Feature Importance ----
    energy_imp = energy_model.feature_importances_.tolist()
    bill_imp = bill_model.feature_importances_.tolist()

    importance_data = {
        'features': feature_names,
        'energy_importance': energy_imp,
        'bill_importance': bill_imp,
    }
    with open("feature_importances_xgb.json", "w") as f:
        json.dump(importance_data, f, indent=2)
    print("[OK] Feature importances saved: feature_importances_xgb.json")

    # Save feature list for backend
    with open("model_features.json", "w") as f:
        json.dump(feature_names, f)
    print("[OK] Feature list saved: model_features.json")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Energy Model: R2={r2_e:.4f}, RMSE={rmse_e:.2f}")
    print(f"  Bill Model:   R2={r2_b:.4f}, RMSE={rmse_b:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
