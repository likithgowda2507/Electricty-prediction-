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

# Define MODEL_DIR first
MODEL_DIR = os.path.join(BASE_DIR, "..")

try:
    from data_prep_unified import FEATURES, load_and_prepare
    print(f"[OK] Imported FEATURES from data_prep_unified: {len(FEATURES) if FEATURES else 0} features")
except ImportError as e:
    print(f"[WARN] Could not import from data_prep_unified: {e}")
    FEATURES = []

# Also try to load from model_features.json to ensure consistency with trained model
try:
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
        elif feat == 'Is_Summer':
            # Summer is April-June (months 4, 5, 6)
            row[feat] = 1 if month in [4, 5, 6] else 0
        elif feat == 'Summer_Demand_Boost':
            # 50% boost to peak demand during summer
            is_summer = 1 if month in [4, 5, 6] else 0
            peak_demand = float(baseline['Maximum_Demand_kW'].median()) if 'Maximum_Demand_kW' in baseline.columns else 50
            row[feat] = is_summer * peak_demand * 0.5
        elif feat == 'Summer_Temp_Bill_Factor':
            # Temperature × Demand interaction specifically in summer
            is_summer = 1 if month in [4, 5, 6] else 0
            temp_val = temp if temp is not None else 25
            peak_demand = float(baseline['Maximum_Demand_kW'].median()) if 'Maximum_Demand_kW' in baseline.columns else 50
            row[feat] = is_summer * (temp_val - 30) * peak_demand
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
            
            # Get tariff info for display
            mask = (df['Building_ID'] == building_id) & (df['Month'] == month)
            tariff = float(df[mask]['Tariff_per_kWh'].mean()) if not df[mask].empty else 7.0

            # Previous year baseline
            prev_energy, prev_bill = get_baseline_usage(building_id, month)

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

    # === PRIMARY FACTOR: Year-over-Year Bill/Energy Trend ===
    # Thresholds match the verdict (±5%) so they never contradict
    energy_change_pct = 0
    if actual_energy_a > 0:
        energy_change_pct = ((pred_energy_b - actual_energy_a) / actual_energy_a) * 100

    if pct_change > 5:
        factors.append({
            'factor': 'Year-over-Year Bill Increase',
            'detail': f'Predicted bill ₹{pred_bill_b:,.0f} vs {year_a} actual ₹{actual_bill_a:,.0f} — up {pct_change:+.1f}% | Energy {energy_change_pct:+.1f}%',
            'impact': 'negative'
        })
    elif pct_change < -5:
        factors.append({
            'factor': 'Year-over-Year Bill Decrease',
            'detail': f'Predicted bill ₹{pred_bill_b:,.0f} vs {year_a} actual ₹{actual_bill_a:,.0f} — down {pct_change:+.1f}% | Energy {energy_change_pct:+.1f}%',
            'impact': 'positive'
        })
    else:
        factors.append({
            'factor': 'Stable Year-over-Year Bill',
            'detail': f'Predicted bill ₹{pred_bill_b:,.0f} vs {year_a} actual ₹{actual_bill_a:,.0f} — change {pct_change:+.1f}%',
            'impact': 'neutral'
        })

    # === Historical Consumption Momentum ===
    mask_bm = (df['Building_ID'] == building_id) & (df['Month'] == month)
    bm_data = df[mask_bm]
    overall_bld = df[df['Building_ID'] == building_id]

    if not bm_data.empty:
        # Check if consumption has been rising year-over-year
        yearly_energy = df[(df['Building_ID'] == building_id) & (df['Month'] == month)].groupby('Year')['Energy_Consumption_kWh'].mean()
        if len(yearly_energy) >= 2:
            sorted_years = yearly_energy.sort_index()
            recent_val = float(sorted_years.iloc[-1])
            older_val = float(sorted_years.iloc[-2])
            if older_val > 0:
                hist_growth = ((recent_val - older_val) / older_val) * 100
                if hist_growth > 3:
                    factors.append({
                        'factor': 'Rising Consumption Trend',
                        'detail': f'Energy grew {hist_growth:+.1f}% between {int(sorted_years.index[-2])}–{int(sorted_years.index[-1])} for this month',
                        'impact': 'negative'
                    })
                elif hist_growth < -3:
                    factors.append({
                        'factor': 'Declining Consumption Trend',
                        'detail': f'Energy fell {hist_growth:+.1f}% between {int(sorted_years.index[-2])}–{int(sorted_years.index[-1])} for this month',
                        'impact': 'positive'
                    })

    # === Lag Consumption / Rolling Average ===
    if not bm_data.empty and 'Lag_1_Month_Consumption' in df.columns:
        lag1 = float(bm_data['Lag_1_Month_Consumption'].mean())
        avg_consumption = float(overall_bld['Energy_Consumption_kWh'].mean())
        if avg_consumption > 0:
            lag_pct = ((lag1 - avg_consumption) / avg_consumption) * 100
            if lag_pct > 10:
                factors.append({
                    'factor': 'High Previous-Month Consumption',
                    'detail': f'Prior month consumption {lag1:.0f} kWh vs building avg {avg_consumption:.0f} kWh — momentum carries forward',
                    'impact': 'negative'
                })
            elif lag_pct < -10:
                factors.append({
                    'factor': 'Low Previous-Month Consumption',
                    'detail': f'Prior month consumption {lag1:.0f} kWh vs building avg {avg_consumption:.0f} kWh — lower baseline',
                    'impact': 'positive'
                })

    # === Peak Load / Maximum Demand ===
    if not bm_data.empty and 'Peak_Load_kW' in df.columns:
        peak_month = float(bm_data['Peak_Load_kW'].mean())
        peak_overall = float(overall_bld['Peak_Load_kW'].mean())
        if peak_overall > 0:
            peak_pct = ((peak_month - peak_overall) / peak_overall) * 100
            if peak_pct > 10:
                factors.append({
                    'factor': 'High Peak Load',
                    'detail': f'Peak demand {peak_month:.1f} kW vs avg {peak_overall:.1f} kW — demand charges increase bill',
                    'impact': 'negative'
                })
            elif peak_pct < -10:
                factors.append({
                    'factor': 'Low Peak Load',
                    'detail': f'Peak demand {peak_month:.1f} kW vs avg {peak_overall:.1f} kW — lower demand charges',
                    'impact': 'positive'
                })

    # === Temperature ===
    if temp_b > 30:
        factors.append({
            'factor': 'High Temperature',
            'detail': f'Expected {temp_b:.1f}°C increases cooling demand',
            'impact': 'negative'
        })
    elif temp_b < 20:
        factors.append({
            'factor': 'Low Temperature',
            'detail': f'Expected {temp_b:.1f}°C reduces cooling needs',
            'impact': 'positive'
        })

    # === Tariff ===
    avg_tariff = float(df[mask_bm]['Tariff_per_kWh'].mean()) if not df[mask_bm].empty else 7.0
    overall_tariff = float(overall_bld['Tariff_per_kWh'].mean())
    if overall_tariff > 0:
        tariff_diff = ((avg_tariff - overall_tariff) / overall_tariff) * 100
        if tariff_diff > 2:
            factors.append({
                'factor': 'Higher Tariff Rate',
                'detail': f'₹{avg_tariff:.2f}/kWh vs annual avg ₹{overall_tariff:.2f}/kWh — cost per unit is higher',
                'impact': 'negative'
            })
        elif tariff_diff < -2:
            factors.append({
                'factor': 'Lower Tariff Rate',
                'detail': f'₹{avg_tariff:.2f}/kWh vs annual avg ₹{overall_tariff:.2f}/kWh — cost per unit is lower',
                'impact': 'positive'
            })

    # === Equipment Efficiency ===
    if not bm_data.empty and 'Equipment_Utilization_Percent' in df.columns:
        equip_month = float(bm_data['Equipment_Utilization_Percent'].mean())
        equip_overall = float(overall_bld['Equipment_Utilization_Percent'].mean())

        if equip_month > equip_overall * 1.15:
            factors.append({
                'factor': 'High Equipment Utilization',
                'detail': f'Equipment at {equip_month:.1f}% vs avg {equip_overall:.1f}% — heavier machinery load',
                'impact': 'negative'
            })
        elif equip_month < equip_overall * 0.85:
            factors.append({
                'factor': 'Low Equipment Utilization',
                'detail': f'Equipment at {equip_month:.1f}% vs avg {equip_overall:.1f}% — lighter load saves energy',
                'impact': 'positive'
            })

    # === Building Occupancy ===
    if not bm_data.empty and 'Occupancy_Rate' in df.columns:
        occ_month = float(bm_data['Occupancy_Rate'].mean())
        occ_overall = float(overall_bld['Occupancy_Rate'].mean())

        if occ_month > occ_overall * 1.1:
            factors.append({
                'factor': 'Higher Occupancy',
                'detail': f'Occupancy at {occ_month:.1f}% vs avg {occ_overall:.1f}% — more people = more energy',
                'impact': 'negative'
            })
        elif occ_month < occ_overall * 0.9:
            factors.append({
                'factor': 'Lower Occupancy',
                'detail': f'Occupancy at {occ_month:.1f}% vs avg {occ_overall:.1f}% — fewer occupants save energy',
                'impact': 'positive'
            })

    # === Seasonal Variation ===
    season_map = {1: 'Winter', 2: 'Winter', 3: 'Summer', 4: 'Summer',
                  5: 'Summer', 6: 'Monsoon', 7: 'Monsoon', 8: 'Monsoon',
                  9: 'Monsoon', 10: 'Winter', 11: 'Winter', 12: 'Winter'}
    current_season = season_map.get(month, 'Winter')

    if 'Season' in df.columns:
        seasonal_avg = df[df['Building_ID'] == building_id].groupby('Season')['Energy_Consumption_kWh'].mean()
        overall_energy_avg = float(overall_bld['Energy_Consumption_kWh'].mean())

        if current_season in seasonal_avg.index:
            season_energy = float(seasonal_avg[current_season])
            pct_diff_season = ((season_energy - overall_energy_avg) / overall_energy_avg) * 100 if overall_energy_avg > 0 else 0

            if current_season == 'Summer':
                factors.append({
                    'factor': 'Summer Season Effect',
                    'detail': f'Summer months see {pct_diff_season:+.1f}% energy vs yearly avg — cooling demand peaks',
                    'impact': 'negative' if pct_diff_season > 5 else 'neutral'
                })
            elif current_season == 'Monsoon':
                factors.append({
                    'factor': 'Monsoon Season Effect',
                    'detail': f'Monsoon months see {pct_diff_season:+.1f}% energy vs yearly avg — humidity drives HVAC load',
                    'impact': 'negative' if pct_diff_season > 5 else 'neutral'
                })
            elif current_season == 'Winter':
                factors.append({
                    'factor': 'Winter Season Effect',
                    'detail': f'Winter months see {pct_diff_season:+.1f}% energy vs yearly avg — lower cooling needs',
                    'impact': 'positive' if pct_diff_season < -5 else 'neutral'
                })

    # === Weather (Humidity, CDD, Heat Index) ===
    if not bm_data.empty:
        if 'Humidity' in df.columns:
            humidity_month = float(bm_data['Humidity'].mean())
            if humidity_month > 75:
                factors.append({
                    'factor': 'High Humidity',
                    'detail': f'Humidity at {humidity_month:.0f}% — HVAC works harder for dehumidification',
                    'impact': 'negative'
                })

        if 'Cooling_Degree_Days' in df.columns:
            cdd_month = float(bm_data['Cooling_Degree_Days'].mean())
            cdd_overall = float(overall_bld['Cooling_Degree_Days'].mean())
            if cdd_month > cdd_overall * 1.3 and cdd_month > 3:
                factors.append({
                    'factor': 'High Cooling Demand',
                    'detail': f'Cooling degree days {cdd_month:.1f} vs avg {cdd_overall:.1f} — significant cooling energy required',
                    'impact': 'negative'
                })

        if 'Heat_Index' in df.columns:
            heat_idx = float(bm_data['Heat_Index'].mean())
            if heat_idx > 35:
                factors.append({
                    'factor': 'Extreme Heat Index',
                    'detail': f'Heat index of {heat_idx:.1f}°C — combined heat & humidity strain on cooling',
                    'impact': 'negative'
                })

    # === FILTER: Only show factors that align with the prediction direction ===
    # When bill is HIGHER → show negative factors (they explain the increase)
    # When bill is LOWER  → show positive factors (they explain the decrease)
    # Always keep the primary Year-over-Year factor and neutral factors
    primary_keywords = ['Year-over-Year']

    if verdict == 'HIGHER':
        factors = [f for f in factors
                   if f['impact'] in ('negative', 'neutral')
                   or any(kw in f['factor'] for kw in primary_keywords)]
    elif verdict == 'LOWER':
        factors = [f for f in factors
                   if f['impact'] in ('positive', 'neutral')
                   or any(kw in f['factor'] for kw in primary_keywords)]

    # Ensure we always have at least 2 factors for context
    if len(factors) < 2:
        # Add the energy trend as a supporting factor
        if pred_energy_b > actual_energy_a:
            factors.append({
                'factor': 'Higher Energy Demand',
                'detail': f'Predicted {pred_energy_b:.0f} kWh vs {year_a} actual {actual_energy_a:.0f} kWh — model expects increased usage',
                'impact': 'negative'
            })
        elif pred_energy_b < actual_energy_a:
            factors.append({
                'factor': 'Lower Energy Demand',
                'detail': f'Predicted {pred_energy_b:.0f} kWh vs {year_a} actual {actual_energy_a:.0f} kWh — model expects reduced usage',
                'impact': 'positive'
            })

        # Add tariff context
        factors.append({
            'factor': 'Tariff & Demand Charges',
            'detail': f'Avg tariff ₹{avg_tariff:.2f}/kWh combined with demand charges shape the final bill',
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


# ---- SOLAR ANALYSIS ----
@app.route('/api/solar-analysis', methods=['POST'])
def solar_analysis():
    """Calculate monthly solar power generation potential and savings."""
    if df is None:
        return jsonify({'error': 'Dataset not loaded'}), 500

    data = request.json
    building_id = data.get('building_id', 'B1')
    year = int(data.get('year', 2026))
    panel_capacity_kw = float(data.get('panel_capacity_kw', 0))  # 0 = auto-calculate

    # Building info
    bld = df[df['Building_ID'] == building_id]
    if bld.empty:
        return jsonify({'error': 'Building not found'}), 404

    building_area = float(bld['Building_Area_sqft'].mean())
    building_type = str(bld['Building_Type'].iloc[0]) if 'Building_Type' in bld.columns else 'Office'

    # Solar panel sizing: ~10% of building rooftop usable, 1 kW per 100 sqft
    usable_roof_sqft = building_area * 0.30  # 30% rooftop available for panels
    auto_capacity_kw = usable_roof_sqft / 100  # 1 kW per 100 sqft of panels

    if panel_capacity_kw <= 0:
        panel_capacity_kw = auto_capacity_kw

    # Average Peak Sun Hours by month for India (typical values)
    # Based on MNRE data for central/south India
    peak_sun_hours = {
        1: 5.0, 2: 5.5, 3: 6.0, 4: 6.5, 5: 6.8, 6: 5.5,
        7: 4.5, 8: 4.5, 9: 5.0, 10: 5.5, 11: 5.0, 12: 4.8
    }

    # Days per month
    days_in_month = {
        1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
        7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
    }

    monthly_results = []
    total_generation = 0
    total_savings = 0
    total_consumption = 0

    for month in range(1, 13):
        mask = (df['Building_ID'] == building_id) & (df['Month'] == month)
        month_data = df[mask]

        if month_data.empty:
            continue

        # Weather conditions for this month
        cloud_cover = float(month_data['Cloud_Cover'].mean()) if 'Cloud_Cover' in df.columns else 50
        temp_avg = float(month_data['Temperature_Avg'].mean())
        humidity = float(month_data['Humidity'].mean()) if 'Humidity' in df.columns else 60
        rainfall = float(month_data['Rainfall_mm'].mean()) if 'Rainfall_mm' in df.columns else 5

        # Average consumption and bill for this building + month
        avg_consumption = float(month_data['Energy_Consumption_kWh'].mean())
        avg_bill = float(month_data['Electricity_Bill_Amount'].mean())
        avg_tariff = float(month_data['Tariff_per_kWh'].mean()) if 'Tariff_per_kWh' in df.columns else 7.0

        # Solar generation calculation
        # 1. Cloud cover derating factor (0% cloud = 100% generation, 100% cloud = ~20%)
        cloud_factor = 1.0 - (cloud_cover / 100.0) * 0.8

        # 2. Temperature derating (-0.4% per degree above 25°C for silicon panels)
        temp_factor = 1.0
        if temp_avg > 25:
            temp_factor = 1.0 - 0.004 * (temp_avg - 25)
        temp_factor = max(temp_factor, 0.7)  # Floor at 70%

        # 3. System efficiency (inverter, wiring, dust, degradation)
        system_efficiency = 0.80

        # 4. Daily generation = capacity × peak sun hours × cloud factor × temp factor × efficiency
        daily_generation = panel_capacity_kw * peak_sun_hours[month] * cloud_factor * temp_factor * system_efficiency

        # 5. Monthly generation
        monthly_generation = daily_generation * days_in_month[month]

        # 6. Actual usable solar (can't exceed consumption)
        self_consumed = min(monthly_generation, avg_consumption)

        # 7. Excess exported to grid (net metering at ~50% tariff)
        excess = max(0, monthly_generation - avg_consumption)
        export_credit = excess * avg_tariff * 0.5

        # 8. Savings
        direct_savings = self_consumed * avg_tariff
        total_monthly_savings = direct_savings + export_credit

        # Solar offset percentage
        solar_offset_pct = (monthly_generation / avg_consumption * 100) if avg_consumption > 0 else 0

        # CO2 savings (India grid emission factor: ~0.82 kg CO2/kWh)
        co2_saved_kg = monthly_generation * 0.82

        monthly_results.append({
            'month': month,
            'days': days_in_month[month],
            'solar_generation_kwh': round(monthly_generation, 1),
            'self_consumed_kwh': round(self_consumed, 1),
            'excess_exported_kwh': round(excess, 1),
            'avg_consumption_kwh': round(avg_consumption, 1),
            'net_consumption_kwh': round(max(0, avg_consumption - self_consumed), 1),
            'avg_bill': round(avg_bill, 0),
            'direct_savings': round(direct_savings, 0),
            'export_credit': round(export_credit, 0),
            'total_savings': round(total_monthly_savings, 0),
            'reduced_bill': round(avg_bill - total_monthly_savings, 0),
            'solar_offset_pct': round(solar_offset_pct, 1),
            'co2_saved_kg': round(co2_saved_kg, 1),
            'cloud_cover': round(cloud_cover, 1),
            'temp_avg': round(temp_avg, 1),
            'peak_sun_hrs': peak_sun_hours[month]
        })

        total_generation += monthly_generation
        total_savings += total_monthly_savings
        total_consumption += avg_consumption

    # Annual summary
    total_original_bill = sum(r['avg_bill'] for r in monthly_results)
    total_reduced_bill = sum(r['reduced_bill'] for r in monthly_results)
    annual_co2_saved = sum(r['co2_saved_kg'] for r in monthly_results)

    # ROI calculation
    # Average cost of solar in India: Rs.45,000-55,000 per kW installed
    installation_cost_per_kw = 50000
    total_installation_cost = panel_capacity_kw * installation_cost_per_kw
    payback_years = total_installation_cost / total_savings if total_savings > 0 else 99

    # Best and worst months
    best_month = max(monthly_results, key=lambda x: x['solar_generation_kwh']) if monthly_results else None
    worst_month = min(monthly_results, key=lambda x: x['solar_generation_kwh']) if monthly_results else None

    return jsonify({
        'building': building_id,
        'building_type': building_type,
        'building_area_sqft': round(building_area, 0),
        'panel_capacity_kw': round(panel_capacity_kw, 1),
        'year': year,
        'monthly': monthly_results,
        'summary': {
            'total_generation_kwh': round(total_generation, 0),
            'total_consumption_kwh': round(total_consumption, 0),
            'total_savings_rs': round(total_savings, 0),
            'original_annual_bill': round(total_original_bill, 0),
            'reduced_annual_bill': round(total_reduced_bill, 0),
            'bill_reduction_pct': round((total_savings / total_original_bill * 100) if total_original_bill > 0 else 0, 1),
            'annual_co2_saved_kg': round(annual_co2_saved, 0),
            'annual_co2_saved_tons': round(annual_co2_saved / 1000, 1),
            'installation_cost': round(total_installation_cost, 0),
            'payback_years': round(payback_years, 1),
            'overall_solar_offset_pct': round((total_generation / total_consumption * 100) if total_consumption > 0 else 0, 1),
            'best_month': best_month['month'] if best_month else None,
            'worst_month': worst_month['month'] if worst_month else None,
            'trees_equivalent': round(annual_co2_saved / 21, 0)  # 1 tree absorbs ~21 kg CO2/year
        }
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
