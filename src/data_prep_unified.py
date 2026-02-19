"""
Unified Data Preparation Module
Loads the original building dataset (energy_dataset_india_2021_2025_advanced 1.xlsx)
and prepares features + targets for all model training scripts.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import joblib

# ============================================================
# FEATURE DEFINITIONS
# ============================================================

# ============================================================
# FEATURE DEFINITIONS
# ============================================================

# Original features
BASE_FEATURES = [
    # Time
    'Year', 'Month', 'Month_Sin', 'Month_Cos',
    'Day_of_Week', 'Is_Weekend', 'Is_Holiday',
    # Building
    'Building_ID_Encoded',
    'Building_Area_sqft', 'Number_of_Occupants',
    # Summer-specific (HIGH IMPACT on summer bills)
    'Is_Summer', 'Summer_Demand_Boost', 'Summer_Temp_Bill_Factor',
    # Weather
    'Temperature_Avg', 'Humidity', 'Rainfall_mm', 'Wind_Speed',
    'Cooling_Degree_Days', 'Heating_Degree_Days',
    # Temperature-based features (captures temp → power usage relationship)
    'Temperature_Anomaly', 'Temp_AC_Interaction', 'Temp_Peak_Load_Interaction', 'Temp_Multiplier',
    # Power / Electrical
    'Peak_Load_kW', 'Load_Factor', 'Power_Factor_Avg',
    'Voltage_Avg', 'Current_Avg',
    'Maximum_Demand_kW',
    # Operational
    'Interruption_Duration_Minutes',
    'Working_Hours', 'Occupancy_Rate',
    'AC_Usage_Hours', 'Equipment_Utilization_Percent',
    # Tariff / Cost
    'Tariff_per_kWh', 'Demand_Charge_per_kW', 'Fixed_Charge',
    # Lag / Rolling
    'Lag_1_Month_Consumption', 'Lag_3_Month_Consumption',
    'Rolling_3_Month_Avg', 'Previous_Month_Bill',
]

# Advanced engineered features - REMOVED

# Combined feature list
FEATURES = BASE_FEATURES

TARGETS = ['Energy_Consumption_kWh', 'Electricity_Bill_Amount']

# Season encoding map
SEASON_MAP = {'Winter': 0, 'Summer': 1, 'Monsoon': 2, 'Autumn': 3, 'Spring': 4}


def load_and_prepare(test_size=0.2, random_state=42):
    """
    Load the building energy dataset and prepare features + targets.

    Returns:
        X_train, X_test, y_train, y_test, feature_names, label_encoder
    """
    from sklearn.model_selection import train_test_split

    # Resolve path
    data_path = "Data/energy_dataset_india_2021_2025_advanced 1.xlsx"
    if not os.path.exists(data_path):
        data_path = "../Data/energy_dataset_india_2021_2025_advanced 1.xlsx"
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), "..", "Data",
                                 "energy_dataset_india_2021_2025_advanced 1.xlsx")

    print(f"Loading dataset from {data_path}...")
    df = pd.read_excel(data_path)
    print(f"  Raw rows: {len(df)}, columns: {len(df.columns)}")

    # ---- Encode Building ID ----
    le = LabelEncoder()
    df['Building_ID_Encoded'] = le.fit_transform(df['Building_ID'].astype(str))

    # Save encoder for prediction
    encoder_path = os.path.join(os.path.dirname(data_path), "..", "building_encoder.pkl")
    joblib.dump(le, encoder_path)
    print(f"  Building encoder saved ({len(le.classes_)} classes: {list(le.classes_)})")

    # ---- Ensure Month_Sin / Month_Cos exist ----
    if 'Month_Sin' not in df.columns:
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    if 'Month_Cos' not in df.columns:
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    # ---- Create Temperature-Based Features ----
    temp_mean = df['Temperature_Avg'].mean()
    df['Temperature_Anomaly'] = df['Temperature_Avg'] - temp_mean
    df['Temp_AC_Interaction'] = df['Temperature_Avg'] * df['AC_Usage_Hours']
    df['Temp_Peak_Load_Interaction'] = df['Temperature_Avg'] * df['Peak_Load_kW']
    df['Temp_Multiplier'] = 1 + (df['Temperature_Avg'] - 30) / 10
    df['Temp_Multiplier'] = df['Temp_Multiplier'].clip(lower=0.8)
    print("  Created temperature-based features to capture temperature -> consumption relationship")

    # ---- Create Summer-Specific Features ----
    # April (4), May (5), June (6) are summer months in India with high AC usage
    df['Is_Summer'] = df['Month'].isin([4, 5, 6]).astype(int)
    df['Summer_Demand_Boost'] = df['Is_Summer'] * df['Maximum_Demand_kW'] * 0.5  # 50% boost in peak demand
    df['Summer_Temp_Bill_Factor'] = df['Is_Summer'] * (df['Temperature_Avg'] - 30) * df['Maximum_Demand_kW']  # Temp * Demand in summer
    print("  Created summer-specific features (Is_Summer, Summer_Demand_Boost, Summer_Temp_Bill_Factor)")

    # ---- Handle Missing Values ----
    # Fill numeric NaN with column median
    for col in FEATURES + TARGETS:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Drop rows where targets are still NaN
    df = df.dropna(subset=TARGETS)

    # ---- Verify all features exist and select available ones ----
    available_features = [c for c in FEATURES if c in df.columns]
    missing_cols = [c for c in FEATURES if c not in df.columns]
    
    if missing_cols:
        print(f"\n  WARNING: {len(missing_cols)} features not available:")
        for col in missing_cols[:10]:  # Show first 10
            print(f"    - {col}")
        if len(missing_cols) > 10:
            print(f"    ... and {len(missing_cols) - 10} more")
        print(f"  Using {len(available_features)} available features")
    
    # For any missing numeric columns, fill with 0
    for col in available_features:
        if col not in df.columns:
            df[col] = 0

    # ---- Extract X, y ----
    X = df[available_features].copy()
    y = df[TARGETS].copy()

    print(f"  Final dataset: {len(X)} rows, {len(available_features)} features")
    print(f"  Targets: {TARGETS}")

    # ---- Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    return X_train, X_test, y_train, y_test, available_features, le
