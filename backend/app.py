from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import joblib
import os
import json
import numpy as np
import sys

# Add src to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "../src"))

try:
    from data_prep_unified import FEATURES, load_and_prepare
    print(f"[OK] Imported FEATURES from data_prep_unified: {len(FEATURES) if FEATURES else 0} features")
except ImportError as e:
    print(f"[WARN] Could not import from data_prep_unified: {e}")
    FEATURES = []

# Also try to load from model_features.json to ensure consistency with trained model
try:
    import json
    with open(os.path.join(MODEL_DIR, "model_features.json")) as f:
        model_features_data = json.load(f)
        if isinstance(model_features_data, dict):
            FEATURES = model_features_data.get('features', FEATURES)
        elif isinstance(model_features_data, list):
            FEATURES = model_features_data
    print(f"[OK] Loaded FEATURES from model_features.json: {len(FEATURES)} features")
except Exception as e:
    print(f"[WARN] Could not load from model_features.json: {e}")

app = Flask(__name__, static_folder="../frontend/static", template_folder="../frontend")

# Paths
MODEL_DIR = os.path.join(BASE_DIR, "..")
DATA_PATH = os.path.join(BASE_DIR, "../Data/energy_dataset_india_2021_2025_advanced 1.xlsx")
ENERGY_MODEL_PATH = os.path.join(MODEL_DIR, "energy_model_xgb.pkl")
BILL_MODEL_PATH = os.path.join(MODEL_DIR, "bill_model_xgb.pkl")
IMPORTANCE_PATH = os.path.join(MODEL_DIR, "feature_importances_xgb.json")
ENCODER_PATH = os.path.join(MODEL_DIR, "building_encoder.pkl")

# Global state
energy_model = None
bill_model = None
df = None
building_encoder = None
importance_data = None


def load_models():
    global energy_model, bill_model, importance_data, building_encoder
    
    try:
        print(f"Loading models from {ENERGY_MODEL_PATH} and {BILL_MODEL_PATH}...")
        energy_model = joblib.load(ENERGY_MODEL_PATH)
        bill_model = joblib.load(BILL_MODEL_PATH)
        print("[OK] Models loaded.")
    except Exception as e:
        print(f"[ERROR] Loading models: {e}")

    try:
        building_encoder = joblib.load(ENCODER_PATH)
        print(f"[OK] Building encoder loaded.")
    except Exception as e:
        print(f"[WARN] No encoder found: {e}")

    try:
        with open(IMPORTANCE_PATH) as f:
            importance_data = json.load(f)
        print("[OK] Feature importance loaded.")
    except Exception as e:
        print(f"[WARN] Feature importance not found: {e}")


def load_dataset():
    global df
    try:
        print(f"Loading dataset from {DATA_PATH}...")
        df = pd.read_excel(DATA_PATH)
        df['Date'] = pd.to_datetime(df['Date'])

        # Ensure Building_ID_Encoded exists
        if building_encoder is not None:
            df['Building_ID_Encoded'] = building_encoder.transform(df['Building_ID'].astype(str))
        else:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df['Building_ID_Encoded'] = le.fit_transform(df['Building_ID'].astype(str))

        # Extract features and target
        features_available = len([c for c in FEATURES if c in df.columns])
        print(f"[OK] Dataset loaded: {len(df)} rows, features: {features_available}/{len(FEATURES)}")
    except Exception as e:
        print(f"[ERROR] Loading dataset: {e}")
    except Exception as e:
        print(f"[ERROR] Loading dataset: {e}")
        import traceback
        traceback.print_exc()


load_models()
load_dataset()


# ============================================================
# HELPER
# ============================================================

def get_historical_weather(month):
    """Get avg weather for a month from historical data."""
    if df is None:
        return 25, 60, 5
    m = df[df['Month'] == month]
    return (
        float(m['Temperature_Avg'].mean()) if not m.empty else 25,
        float(m['Humidity'].mean()) if not m.empty else 60,
        float(m['Rainfall_mm'].mean()) if not m.empty else 5
    )


def get_baseline_usage(building_id, month):
    """Get avg historical consumption for a building + month."""
    if df is None:
        return 100, 5000
    mask = (df['Building_ID'] == building_id) & (df['Month'] == month)
    subset = df[mask]
    if subset.empty:
        return 100, 5000
    return (
        float(subset['Energy_Consumption_kWh'].mean()),
        float(subset['Electricity_Bill_Amount'].mean())
    )


def build_input_row(building_id, year, month, temp=None, humidity=None, rainfall=None):
    """Build a single prediction input row using historical baselines."""
    if df is None:
        return None

    # Historical baseline for this building + month
    mask = (df['Building_ID'] == building_id) & (df['Month'] == month)
    baseline = df[mask]

    if baseline.empty:
        # Try just the building
        baseline = df[df['Building_ID'] == building_id]
    if baseline.empty:
        return None

    row = {}
    
    # Build row for each feature
    for feat in FEATURES:
        if feat == 'Year':
            row[feat] = year
        elif feat == 'Month':
            row[feat] = month
        elif feat == 'Month_Sin':
            row[feat] = np.sin(2 * np.pi * month / 12)
        elif feat == 'Month_Cos':
            row[feat] = np.cos(2 * np.pi * month / 12)
        elif feat == 'Temperature_Avg':
            row[feat] = temp if temp is not None else (float(baseline[feat].median()) if feat in baseline.columns and pd.notna(baseline[feat].median()) else 25)
        elif feat == 'Humidity':
            row[feat] = humidity if humidity is not None else (float(baseline[feat].median()) if feat in baseline.columns and pd.notna(baseline[feat].median()) else 60)
        elif feat == 'Rainfall_mm':
            row[feat] = rainfall if rainfall is not None else (float(baseline[feat].median()) if feat in baseline.columns and pd.notna(baseline[feat].median()) else 5)
        elif feat == 'Building_ID_Encoded':
            if building_encoder is not None and building_id in building_encoder.classes_:
                row[feat] = int(building_encoder.transform([building_id])[0])
            else:
                row[feat] = 0
        else:
            # Use median of historical data for this building+month
            if feat in baseline.columns:
                val = baseline[feat].median()
                row[feat] = float(val) if pd.notna(val) else 0
            else:
                # Fill missing features with 0
                row[feat] = 0

    # Create DataFrame with all features in correct order
    input_df = pd.DataFrame([row])
    
    # Reorder columns to match FEATURES exactly
    return input_df[FEATURES]


# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')


# ---- HISTORY / DASHBOARD DATA ----
@app.route('/api/history')
def get_history():
    if df is None:
        return jsonify({'error': 'Dataset not loaded'}), 500

    # Filter valid buildings (B1, B2, B3)
    valid = df[df['Building_ID'].isin(['B1', 'B2', 'B3'])]

    # KPI
    total_consumption = float(valid['Energy_Consumption_kWh'].sum())
    avg_pf = float(valid['Power_Factor_Avg'].mean())
    max_demand = float(valid['Maximum_Demand_kW'].max())
    total_interruption = float(valid['Interruption_Duration_Minutes'].sum())

    # Monthly agg
    monthly = valid.groupby('Month').agg(
        consumption=('Energy_Consumption_kWh', 'sum'),
        bill_avg=('Electricity_Bill_Amount', 'mean')
    ).reset_index().sort_values('Month')

    # Building totals
    buildings_data = valid.groupby('Building_ID').agg(
        Energy_Consumption_kWh=('Energy_Consumption_kWh', 'sum'),
        Electricity_Bill_Amount=('Electricity_Bill_Amount', 'sum')
    ).reset_index().to_dict('records')

    # Yearly trend
    yearly = valid.groupby('Year').agg(
        consumption=('Energy_Consumption_kWh', 'sum'),
        bill=('Electricity_Bill_Amount', 'sum')
    ).reset_index().sort_values('Year')

    yearly_data = {
        'years': yearly['Year'].tolist(),
        'consumption': yearly['consumption'].tolist(),
        'bill': yearly['bill'].tolist()
    }

    # Building monthly
    building_monthly = {}
    for bid in ['B1', 'B2', 'B3']:
        bdf = valid[valid['Building_ID'] == bid]
        bm = bdf.groupby('Month').agg(
            consumption=('Energy_Consumption_kWh', 'sum'),
            bill=('Electricity_Bill_Amount', 'sum'),
            voltage=('Voltage_Avg', 'mean'),
            current=('Current_Avg', 'mean'),
            pf=('Power_Factor_Avg', 'mean'),
            demand=('Maximum_Demand_kW', 'max'),
            peak_load=('Peak_Load_kW', 'mean'),
            interruptions=('Interruption_Duration_Minutes', 'sum'),
        ).reset_index().sort_values('Month')

        building_monthly[bid] = {
            'months': bm['Month'].tolist(),
            'consumption': bm['consumption'].tolist(),
            'bill': bm['bill'].tolist(),
            'voltage': [round(v, 1) for v in bm['voltage'].tolist()],
            'current': [round(v, 2) for v in bm['current'].tolist()],
            'pf': [round(v, 3) for v in bm['pf'].tolist()],
            'demand': [round(v, 1) for v in bm['demand'].tolist()],
            'peak_load': [round(v, 1) for v in bm['peak_load'].tolist()],
            'interruptions': bm['interruptions'].tolist()
        }

    # Temp vs Energy
    temp_energy = valid.groupby('Month').agg(
        temp=('Temperature_Avg', 'mean'),
        energy=('Energy_Consumption_kWh', 'mean')
    ).reset_index().sort_values('Month')

    # PF by building
    pf_building = valid.groupby('Building_ID').agg(
        Power_Factor_Avg=('Power_Factor_Avg', 'mean')
    ).reset_index().to_dict('records')

    return jsonify({
        'kpi': {
            'total_consumption': total_consumption,
            'avg_pf': round(avg_pf, 2),
            'max_demand': max_demand,
            'interruption': total_interruption
        },
        'months': monthly['Month'].tolist(),
        'consumption': monthly['consumption'].tolist(),
        'monthly_bill_avg': monthly['bill_avg'].tolist(),
        'buildings': buildings_data,
        'yearly': yearly_data,
        'building_monthly': building_monthly,
        'temp_energy': {
            'months': temp_energy['Month'].tolist(),
            'temp': temp_energy['temp'].tolist(),
            'energy': temp_energy['energy'].tolist()
        },
        'pf_building': pf_building
    })


# ---- IMPACT ANALYSIS ----
@app.route('/api/impact-analysis')
def impact_analysis():
    if df is None:
        return jsonify({'error': 'Dataset not loaded'}), 500

    valid = df[df['Building_ID'].isin(['B1', 'B2', 'B3'])]

    # Temperature impact
    bins = [0, 20, 25, 30, 35, 50]
    labels_temp = ['<20C', '20-25C', '25-30C', '30-35C', '>35C']
    valid = valid.copy()
    valid['temp_bin'] = pd.cut(valid['Temperature_Avg'], bins=bins, labels=labels_temp)
    weather = valid.groupby('temp_bin', observed=True).agg(
        consumption=('Energy_Consumption_kWh', 'mean')
    ).reset_index()

    # Interruption impact
    int_bins = [0, 30, 120, 500, 10000]
    int_labels = ['None/Low (<30m)', 'Medium (30-120m)', 'High (2-8h)', 'Severe (>8h)']
    valid['int_bin'] = pd.cut(valid['Interruption_Duration_Minutes'], bins=int_bins, labels=int_labels)
    interruption = valid.groupby('int_bin', observed=True).agg(
        consumption=('Energy_Consumption_kWh', 'mean'),
        bill=('Electricity_Bill_Amount', 'mean')
    ).reset_index()

    # Seasonal
    seasonal = valid.groupby('Season').agg(
        consumption=('Energy_Consumption_kWh', 'mean'),
        bill=('Electricity_Bill_Amount', 'mean')
    ).reset_index()
    season_order = ['Summer', 'Monsoon', 'Winter']
    seasonal['sort'] = seasonal['Season'].map({s: i for i, s in enumerate(season_order)})
    seasonal = seasonal.sort_values('sort').dropna(subset=['sort'])

    # Weekday / Weekend
    weekday = valid.groupby('Is_Weekend').agg(
        consumption=('Energy_Consumption_kWh', 'mean'),
        bill=('Electricity_Bill_Amount', 'mean')
    ).reset_index()

    # Rainfall impact
    rain_bins = [0, 5, 50, 200, 1000]
    rain_labels = ['Dry (<5mm)', 'Light (5-50mm)', 'Moderate (50-200mm)', 'Heavy (>200mm)']
    valid['rain_bin'] = pd.cut(valid['Rainfall_mm'], bins=rain_bins, labels=rain_labels)
    rainfall = valid.groupby('rain_bin', observed=True).agg(
        consumption=('Energy_Consumption_kWh', 'mean')
    ).reset_index()

    return jsonify({
        'weather': {
            'labels': weather['temp_bin'].tolist(),
            'consumption': [round(v, 2) for v in weather['consumption'].tolist()]
        },
        'interruptions': {
            'labels': interruption['int_bin'].tolist(),
            'consumption': [round(v, 2) for v in interruption['consumption'].tolist()],
            'bill': [round(v, 2) for v in interruption['bill'].tolist()]
        },
        'seasonal': {
            'labels': seasonal['Season'].tolist(),
            'consumption': [round(v, 2) for v in seasonal['consumption'].tolist()],
            'bill': [round(v, 2) for v in seasonal['bill'].tolist()]
        },
        'weekday_weekend': {
            'labels': ['Weekday', 'Weekend'],
            'consumption': [round(v, 2) for v in weekday['consumption'].tolist()],
            'bill': [round(v, 2) for v in weekday['bill'].tolist()]
        },
        'rainfall': {
            'labels': rainfall['rain_bin'].tolist(),
            'consumption': [round(v, 2) for v in rainfall['consumption'].tolist()]
        }
    })


# ---- FEATURE IMPORTANCE ----
@app.route('/api/feature-importance')
def feature_importance():
    if importance_data:
        return jsonify(importance_data)
    return jsonify({
        'features': FEATURES,
        'energy_importance': [1.0 / max(len(FEATURES), 1)] * len(FEATURES)
    })


# ---- YEARLY FORECAST ----
@app.route('/api/yearly-forecast', methods=['POST'])
def yearly_forecast():
    if not energy_model or not bill_model:
        return jsonify({'error': 'Models not loaded'}), 500

    data = request.json
    year = int(data.get('year', 2026))
    building_id = data.get('building_id', 'B1')

    results = []

    for month in range(1, 13):
        try:
            temp, humidity, rainfall = get_historical_weather(month)

            X_input = build_input_row(building_id, year, month, temp, humidity, rainfall)
            if X_input is None:
                continue

            energy_pred = float(energy_model.predict(X_input)[0])
            bill_pred = float(bill_model.predict(X_input)[0])

            # Previous year baseline
            prev_energy, prev_bill = get_baseline_usage(building_id, month)

            # Tariff rate from historical data
            mask = (df['Building_ID'] == building_id) & (df['Month'] == month)
            tariff = float(df[mask]['Tariff_per_kWh'].mean()) if not df[mask].empty else 7.0

            results.append({
                'month': month,
                'energy_pred': round(energy_pred, 2),
                'bill_pred': round(bill_pred, 2),
                'prev_energy': round(prev_energy, 2),
                'prev_bill': round(prev_bill, 2),
                'temp': round(temp, 1),
                'tariff_rate': round(tariff, 2)
            })
        except Exception as e:
            print(f"Error month {month}: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_energy = sum(r['energy_pred'] for r in results)
    total_bill = sum(r['bill_pred'] for r in results)

    return jsonify({
        'year': year,
        'building': building_id,
        'monthly': results,
        'total_energy': round(total_energy, 2),
        'total_bill': round(total_bill, 2)
    })


# ---- BUSINESS QA ----
@app.route('/api/business-qa', methods=['POST'])
def business_qa():
    data = request.json
    month = int(data.get('month', 1))
    year_a = int(data.get('year_a', 2025))
    year_b = int(data.get('year_b', 2026))
    building_id = data.get('building_id', 'B1')

    MONTH_NAMES = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Year A actuals
    mask_a = (df['Building_ID'] == building_id) & (df['Month'] == month) & (df['Year'] == year_a)
    df_a = df[mask_a]

    if df_a.empty:
        actual_bill_a = 0
        actual_energy_a = 0
        temp_a = 25
    else:
        actual_bill_a = float(df_a['Electricity_Bill_Amount'].mean())
        actual_energy_a = float(df_a['Energy_Consumption_kWh'].mean())
        temp_a = float(df_a['Temperature_Avg'].mean())

    # Year B prediction
    temp_b, humidity_b, rainfall_b = get_historical_weather(month)
    X_input = build_input_row(building_id, year_b, month, temp_b, humidity_b, rainfall_b)

    if X_input is not None:
        pred_bill_b = float(bill_model.predict(X_input)[0])
        pred_energy_b = float(energy_model.predict(X_input)[0])
    else:
        pred_bill_b = 0
        pred_energy_b = 0

    pct_change = 0
    if actual_bill_a > 0:
        pct_change = ((pred_bill_b - actual_bill_a) / actual_bill_a) * 100

    verdict = "APPROXIMATELY THE SAME"
    if pct_change > 5:
        verdict = "HIGHER"
    elif pct_change < -5:
        verdict = "LOWER"

    factors = []

    if temp_b > 30:
        factors.append({
            'factor': 'High Temperature',
            'detail': f'Expected {temp_b:.1f}C increases cooling demand',
            'impact': 'negative'
        })
    elif temp_b < 20:
        factors.append({
            'factor': 'Low Temperature',
            'detail': f'Expected {temp_b:.1f}C reduces cooling needs',
            'impact': 'positive'
        })

    if rainfall_b > 100:
        factors.append({
            'factor': 'High Rainfall',
            'detail': f'{rainfall_b:.0f}mm expected -- humidity increases load',
            'impact': 'negative'
        })

    # Tariff factor
    mask_tariff = (df['Building_ID'] == building_id) & (df['Month'] == month)
    avg_tariff = float(df[mask_tariff]['Tariff_per_kWh'].mean()) if not df[mask_tariff].empty else 7.0
    if avg_tariff > 8:
        factors.append({
            'factor': 'High Tariff',
            'detail': f'Avg tariff Rs.{avg_tariff:.2f}/kWh',
            'impact': 'negative'
        })

    if not factors:
        factors.append({
            'factor': 'Stable Conditions',
            'detail': 'No extreme weather or tariff changes expected',
            'impact': 'neutral'
        })

    return jsonify({
        'building': building_id,
        'month': month,
        'month_name': MONTH_NAMES[month],
        'year_a': year_a,
        'year_b': year_b,
        'actual_bill_a': round(actual_bill_a, 2),
        'actual_energy_a': round(actual_energy_a, 2),
        'predicted_energy_b': round(pred_energy_b, 2),
        'predicted_bill_b': round(pred_bill_b, 2),
        'pct_change': round(pct_change, 1),
        'verdict': verdict,
        'factors': factors,
        'assumed_conditions': {
            'temp': round(temp_b, 1),
            'tariff': round(avg_tariff, 2)
        }
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
