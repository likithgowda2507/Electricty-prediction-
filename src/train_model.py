import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# 1. Load Data
excel_path = "Data/energy_dataset_india_2021_2025_advanced 1.xlsx"
if not os.path.exists(excel_path):
     # Fallback if running from src
    excel_path = "../Data/energy_dataset_india_2021_2025_advanced 1.xlsx"

print(f"Loading data from {excel_path}...")
df = pd.read_excel(excel_path)

# 2. Preprocessing
# Encode Building_ID
le = LabelEncoder()
df['Building_ID_Encoded'] = le.fit_transform(df['Building_ID'].astype(str))

# Save LabelEncoder for prediction
joblib.dump(le, "building_encoder.pkl")
print("Building Encoder saved.")

# Features Selection
features = [
    'Year', 'Month', 
    'Temperature_Avg', 'Rainfall_mm', 
    'Interruption_Duration_Minutes',
    'Lag_1_Month_Consumption', 
    'Previous_Month_Bill',
    'Building_ID_Encoded'
]

targets = ['Energy_Consumption_kWh', 'Electricity_Bill_Amount']

# Drop rows with missing values in selected columns
df_clean = df.dropna(subset=features + targets)

X = df_clean[features]
y = df_clean[targets]

# 3. Data Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Models
# We can use MultiOutputRegressor or just two separate models. 
# Separate models often give better control if targets have different scales/dynamics.
# Data shows Bill is roughly linear to Energy but with tariffs. 

print("Training Energy Model...")
energy_model = RandomForestRegressor(n_estimators=100, random_state=42)
energy_model.fit(X_train, y_train['Energy_Consumption_kWh'])

print("Training Bill Model...")
bill_model = RandomForestRegressor(n_estimators=100, random_state=42)
bill_model.fit(X_train, y_train['Electricity_Bill_Amount'])

# 5. Evaluate
y_pred_energy = energy_model.predict(X_test)
y_pred_bill = bill_model.predict(X_test)

r2_energy = r2_score(y_test['Energy_Consumption_kWh'], y_pred_energy)
r2_bill = r2_score(y_test['Electricity_Bill_Amount'], y_pred_bill)

print(f"Energy Model R2 Score: {r2_energy:.4f}")
print(f"Bill Model R2 Score: {r2_bill:.4f}")

# 6. Save Models
joblib.dump(energy_model, "energy_model_advanced.pkl")
joblib.dump(bill_model, "bill_model_advanced.pkl")
print("Models saved: energy_model_advanced.pkl, bill_model_advanced.pkl")
