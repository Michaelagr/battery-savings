import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import locale
from datetime import datetime, timedelta
import io

# Set locale for German formatting
locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')

# Page configuration
st.set_page_config(
    page_title="Energie-Kosten-Analyse",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .scenario-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .cost-positive {
        color: #28a745;
        font-weight: bold;
    }
    .cost-negative {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">⚡ Energie-Kosten-Analyse für Industrieunternehmen</h1>', unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Konfiguration")
    
    # File upload
    with st.expander("📁 Daten-Upload", expanded=True):
        load_profile_file = st.file_uploader(
            "Lastprofil (15-Minuten-Intervalle, kW)",
            type=['csv', 'xlsx'],
            help="Upload der Lastprofil-Daten für ein ganzes Jahr in 15-Minuten-Intervallen"
        )
        
        pv_data_file = st.file_uploader(
            "PV-Daten (CSV mit timestamp und yearly_production_fraction)",
            type=['csv'],
            help="Upload der PV-Daten mit Wahrscheinlichkeiten für Solarwerte. Erwartetes Format: timestamp (dd.mm.yy HH:MM) und yearly_production_fraction (0-1)"
        )
        
        spot_price_file = st.file_uploader(
            "Spot-Preis-Daten (Excel mit timestamp und spotPrice)",
            type=['xlsx', 'xls'],
            help="Upload der Spot-Preis-Daten. Erwartetes Format: timestamp (ISO) und spotPrice (€/MWh)"
        )
    
    # PV Configuration
    st.subheader("☀️ PV-Konfiguration")
    kwp_value = st.number_input(
        "PV-Leistung (kWp)",
        min_value=0.0,
        max_value=10000.0,
        value=400.0,
        step=50.0,
        help="Skalierungsfaktor für das PV-Lastprofil. Bei Upload von PV-Daten wird dieser Wert zum Skalieren der Wahrscheinlichkeiten verwendet."
    )
    
    # Battery Configuration
    st.subheader("🔋 Batterie-Konfiguration")
    battery_power = st.number_input(
        "Batterie-Leistung (kW)",
        min_value=0.0,
        max_value=10000.0,
        value=100.0,
        step=10.0
    )
    battery_capacity = st.number_input(
        "Batterie-Kapazität (kWh)",
        min_value=0.0,
        max_value=200.0,
        value=200.0,
        step=10.0
    )
    
    # Cost Parameters
    st.subheader("💰 Kosten-Parameter")
    spot_price_base = st.number_input(
        "Basis-Strompreis (€/MWh)",
        min_value=0.0,
        max_value=1000.0,
        value=80.0,
        step=5.0
    )
    grid_fee = st.number_input(
        "Netzentgelt (€/MWh)",
        min_value=0.0,
        max_value=200.0,
        value=25.0,
        step=1.0
    )
    demand_charge_high = st.number_input(
        "Leistungspreis >2500h (€/kW/Jahr)",
        min_value=0.0,
        max_value=500.0,
        value=200.0,
        step=5.0
    )
    demand_charge_low = st.number_input(
        "Leistungspreis <2500h (€/kW/Jahr)",
        min_value=0.0,
        max_value=500.0,
        value=20.0,
        step=5.0
    )
    eeg_levy = st.number_input(
        "EEG-Umlage (€/MWh)",
        min_value=0.0,
        max_value=100.0,
        value=6.5,
        step=0.1
    )
    vat_rate = st.number_input(
        "Mehrwertsteuer (%)",
        min_value=0.0,
        max_value=25.0,
        value=19.0,
        step=0.5
    )

# Sample data generation functions
def generate_sample_load_profile():
    """Generate a sample load profile for demonstration"""
    # Create a year of 15-minute intervals
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31, 23, 45)
    
    # Generate timestamps
    timestamps = pd.date_range(start=start_date, end=end_date, freq='15T')
    
    # Create realistic load profile with seasonal and daily patterns
    np.random.seed(42)  # For reproducible results
    
    # Base load with seasonal variation
    base_load = 50 + 20 * np.sin(2 * np.pi * np.arange(len(timestamps)) / (24 * 4 * 365))
    
    # Daily pattern (higher during day, lower at night)
    hour_of_day = timestamps.hour + timestamps.minute / 60
    daily_pattern = 1 + 0.3 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    
    # Weekly pattern (lower on weekends)
    day_of_week = timestamps.dayofweek
    weekly_pattern = np.where(day_of_week < 5, 1.0, 0.7)
    
    # Add some randomness
    noise = np.random.normal(0, 50, len(timestamps))
    
    # Combine all patterns
    load = base_load * daily_pattern * weekly_pattern + noise
    load = np.maximum(load, 100)  # Minimum load
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'load_kw': load
    })

def generate_pv_profile():
    """Generate a probabilistic PV profile"""
    # Create a year of 15-minute intervals
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31, 23, 45)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='15T')
    
    # Solar irradiance model
    latitude = 51.1657  # Germany average
    declination = 23.45 * np.sin(2 * np.pi * (timestamps.dayofyear - 80) / 365)
    
    # Hour angle
    hour_angle = 15 * (timestamps.hour + timestamps.minute / 60 - 12)
    
    # Solar altitude
    solar_altitude = np.arcsin(
        np.sin(np.radians(latitude)) * np.sin(np.radians(declination)) +
        np.cos(np.radians(latitude)) * np.cos(np.radians(declination)) * np.cos(np.radians(hour_angle))
    )
    
    # Solar irradiance (simplified)
    irradiance = np.maximum(0, 1000 * np.sin(np.degrees(solar_altitude)) * 0.8)
    
    # Add some cloudiness and variability
    cloud_factor = 0.3 + 0.7 * np.random.beta(2, 2, len(timestamps))
    irradiance = irradiance * cloud_factor
    
    # Convert to power (assuming 15% efficiency)
    pv_power = irradiance * 0.15 / 1000  # kW per m²
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'pv_power_kw_per_kwp': pv_power
    })

def process_pv_data(pv_file, pv_total):
    """Process PV data using the user's approach with scaling"""
    MAGIC_YEARLY_PV_MULTIPLIER = 1000
    INTERVAL_HOURS = 0.25
    
    df_pv = pd.read_csv(pv_file)
    
    # Dataframe to store PV distribution - handle timezone-aware datetime objects
    df_pv["timestamp"] = pd.to_datetime(df_pv["timestamp"], format="%d.%m.%y %H:%M", utc=True).dt.tz_convert('UTC').dt.tz_localize(None)
    df_pv["yearly_production_kw"] = df_pv["yearly_production_fraction"].astype(float).to_numpy().clip(min=0) * pv_total * MAGIC_YEARLY_PV_MULTIPLIER / INTERVAL_HOURS  # result in kW
    df_pv["yearly_production_kwh"] = df_pv["yearly_production_fraction"].astype(float).to_numpy().clip(min=0) * pv_total * MAGIC_YEARLY_PV_MULTIPLIER # result in kwh
    
    # Convert to the format expected by the dashboard
    df_pv["pv_power_kw_per_kwp"] = df_pv["yearly_production_kw"] / pv_total
    
    return df_pv[["timestamp", "pv_power_kw_per_kwp"]]

def process_spot_price_data(spot_price_file):
    """Process spot price data from Excel file"""
    df_spot = pd.read_excel(spot_price_file)
    
    # Debug: Show column names and data types
    st.sidebar.write("Available columns:", df_spot.columns.tolist())
    st.sidebar.write("Data types 👍:", df_spot.dtypes.to_dict())
    
    # Handle timestamp column - handle both uploaded files and local files
    try:
        # Always convert to datetime first, regardless of current type - handle timezone-aware datetime objects
        df_spot["timestamp"] = pd.to_datetime(df_spot["timestamp"], utc=True)
        
        # Then remove timezone if present
        if hasattr(df_spot["timestamp"], 'dt') and df_spot["timestamp"].dt.tz is not None:
            df_spot["timestamp"] = df_spot["timestamp"].dt.tz_convert('UTC').dt.tz_localize(None)
            
    except Exception as e:
        st.error(f"Error converting timestamp: {str(e)}")
        st.write("Sample timestamp values:", df_spot["timestamp"].head().tolist())
        raise
    
    # Rename spotPrice column to match expected format
    if "spotPrice" in df_spot.columns:
        df_spot = df_spot.rename(columns={"spotPrice": "spot_price_eur_mwh"})
    
    # Ensure we have the required columns
    required_cols = ["timestamp", "spot_price_eur_mwh"]
    if not all(col in df_spot.columns for col in required_cols):
        st.error(f"Missing required columns. Available: {df_spot.columns.tolist()}")
        st.write("First few rows:", df_spot.head())
        raise ValueError("Missing required columns")
    
    return df_spot[["timestamp", "spot_price_eur_mwh"]]
    

def generate_spot_prices():
    """Generate realistic spot prices for 2024"""
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31, 23, 45)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='15T')
    
    np.random.seed(42)
    
    # Base price with seasonal variation
    base_price = 80 + 20 * np.sin(2 * np.pi * np.arange(len(timestamps)) / (24 * 4 * 365))
    
    # Daily pattern (higher during peak hours)
    hour_of_day = timestamps.hour + timestamps.minute / 60
    daily_pattern = 1 + 0.2 * np.sin(2 * np.pi * (hour_of_day - 12) / 24)
    
    # Add volatility
    volatility = np.random.normal(0, 10, len(timestamps))
    
    # Combine and ensure positive prices
    prices = base_price * daily_pattern + volatility
    prices = np.maximum(prices, 20)  # Minimum price
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'spot_price_eur_mwh': prices
    })

# Data processing functions
def calculate_demand_charge(load_profile, demand_charge_high, demand_charge_low):
    """Calculate demand charge based on Vollaststunden"""
    max_load = load_profile['load_kw'].max()
    total_energy = load_profile['load_kw'].sum() * 0.25  # 15-minute intervals
    vollaststunden = total_energy / max_load if max_load > 0 else 0
    
    if vollaststunden > 2500:
        return max_load * demand_charge_high / 1000  # Convert to €/year
    else:
        return max_load * demand_charge_low / 1000

def calculate_energy_cost(load_profile, spot_prices, grid_fee, eeg_levy, vat_rate):
    """Calculate total energy cost"""
    # Ensure both dataframes have the same timestamp type
    load_profile = load_profile.copy()
    spot_prices = spot_prices.copy()
    
    # Convert timestamps to the same type - handle timezone-aware datetime objects
    load_profile['timestamp'] = pd.to_datetime(load_profile['timestamp'], utc=True).dt.tz_convert('UTC').dt.tz_localize(None)
    spot_prices['timestamp'] = pd.to_datetime(spot_prices['timestamp'], utc=True).dt.tz_convert('UTC').dt.tz_localize(None)
    
    # Merge load profile with spot prices
    merged = pd.merge(load_profile, spot_prices, on='timestamp', how='left')
    
    # Calculate energy cost components
    energy_cost = (merged['load_kw'] * merged['spot_price_eur_mwh'] * 0.25 / 1000).sum()
    grid_cost = (merged['load_kw'] * grid_fee * 0.25 / 1000).sum()
    eeg_cost = (merged['load_kw'] * eeg_levy * 0.25 / 1000).sum()
    
    # Add VAT
    subtotal = energy_cost + grid_cost + eeg_cost
    vat_amount = subtotal * vat_rate / 100
    total_cost = subtotal + vat_amount
    
    return {
        'energy_cost': energy_cost,
        'grid_cost': grid_cost,
        'eeg_cost': eeg_cost,
        'vat_amount': vat_amount,
        'total_cost': total_cost
    }

def simple_battery_optimization(load_profile, pv_profile, battery_power, battery_capacity):
    """Simple battery optimization for self-consumption"""
    # Scale PV profile
    pv_load = pv_profile['pv_power_kw_per_kwp'] * kwp_value
    
    # Calculate net load
    net_load = load_profile['load_kw'] - pv_load
    
    # Simple battery optimization
    battery_soc = np.zeros(len(net_load))
    battery_charge = np.zeros(len(net_load))
    battery_discharge = np.zeros(len(net_load))
    
    for i in range(len(net_load)):
        if net_load[i] < 0:  # PV surplus
            # Charge battery if possible
            charge_power = min(-net_load[i], battery_power, battery_capacity - battery_soc[i-1] if i > 0 else battery_capacity)
            battery_charge[i] = charge_power
            battery_soc[i] = battery_soc[i-1] + charge_power if i > 0 else charge_power
        else:  # Grid consumption
            # Discharge battery if possible
            discharge_power = min(net_load[i], battery_power, battery_soc[i-1] if i > 0 else 0)
            battery_discharge[i] = discharge_power
            battery_soc[i] = battery_soc[i-1] - discharge_power if i > 0 else 0
        
        # Ensure SOC stays within bounds
        battery_soc[i] = np.clip(battery_soc[i], 0, battery_capacity)
    
    # Calculate final grid load
    final_grid_load = net_load + battery_charge - battery_discharge
    
    return {
        'net_load': net_load,
        'battery_soc': battery_soc,
        'battery_charge': battery_charge,
        'battery_discharge': battery_discharge,
        'final_grid_load': final_grid_load,
        'pv_surplus': np.maximum(-net_load, 0),
        'grid_consumption': np.maximum(net_load, 0)
    }

def smart_battery_optimization(load_profile, pv_profile, battery_power, battery_capacity, spot_prices):
    """Smart battery optimization with peak shaving and load shifting"""
    # Scale PV profile
    pv_load = pv_profile['pv_power_kw_per_kwp'] * kwp_value
    
    # Calculate net load
    net_load = load_profile['load_kw'] - pv_load
    
    # Ensure consistent timestamp types for merging - handle timezone-aware datetime objects
    load_timestamps = pd.to_datetime(load_profile['timestamp'], utc=True).dt.tz_convert('UTC').dt.tz_localize(None)
    spot_timestamps = pd.to_datetime(spot_prices['timestamp'], utc=True).dt.tz_convert('UTC').dt.tz_localize(None)
    
    # Merge with spot prices for optimization
    merged = pd.merge(
        pd.DataFrame({
            'timestamp': load_timestamps,
            'net_load': net_load,
            'original_load': load_profile['load_kw']
        }),
        spot_prices.assign(timestamp=spot_timestamps),
        on='timestamp',
        how='left'
    )
    
    # Initialize battery variables
    battery_soc = np.zeros(len(merged))
    battery_charge = np.zeros(len(merged))
    battery_discharge = np.zeros(len(merged))
    
    # Phase 1: Peak shaving
    # Find peak periods and use battery to reduce them
    sorted_indices = np.argsort(merged['net_load'])[::-1]  # Sort by load descending
    
    for idx in sorted_indices:
        if merged.iloc[idx]['net_load'] > 0:  # Only for positive loads
            # Calculate how much we can reduce this peak
            available_soc = battery_soc[idx-1] if idx > 0 else 0
            discharge_power = min(
                merged.iloc[idx]['net_load'],
                battery_power,
                available_soc
            )
            
            if discharge_power > 0:
                battery_discharge[idx] = discharge_power
                battery_soc[idx] = available_soc - discharge_power
                merged.iloc[idx, merged.columns.get_loc('net_load')] -= discharge_power
    
    # Phase 2: Load shifting based on price
    # Sort by price for optimal charging/discharging
    price_sorted_indices = np.argsort(merged['spot_price_eur_mwh'])
    
    for idx in price_sorted_indices:
        current_soc = battery_soc[idx-1] if idx > 0 else 0
        
        if merged.iloc[idx]['net_load'] < 0:  # PV surplus, charge if price is low
            if merged.iloc[idx]['spot_price_eur_mwh'] < merged['spot_price_eur_mwh'].quantile(0.3):
                charge_power = min(
                    -merged.iloc[idx]['net_load'],
                    battery_power,
                    battery_capacity - current_soc
                )
                battery_charge[idx] = charge_power
                battery_soc[idx] = current_soc + charge_power
        else:  # Grid consumption, discharge if price is high
            if merged.iloc[idx]['spot_price_eur_mwh'] > merged['spot_price_eur_mwh'].quantile(0.7):
                discharge_power = min(
                    merged.iloc[idx]['net_load'],
                    battery_power,
                    current_soc
                )
                battery_discharge[idx] = discharge_power
                battery_soc[idx] = current_soc - discharge_power
                merged.iloc[idx, merged.columns.get_loc('net_load')] -= discharge_power
        
        # Ensure SOC stays within bounds
        battery_soc[idx] = np.clip(battery_soc[idx], 0, battery_capacity)
    
    # Calculate final grid load
    final_grid_load = merged['net_load'] + battery_charge - battery_discharge
    
    return {
        'net_load': merged['net_load'],
        'battery_soc': battery_soc,
        'battery_charge': battery_charge,
        'battery_discharge': battery_discharge,
        'final_grid_load': final_grid_load,
        'pv_surplus': np.maximum(-merged['net_load'], 0),
        'grid_consumption': np.maximum(merged['net_load'], 0),
        'original_load': merged['original_load']
    }

# Main dashboard logic
if load_profile_file is not None:
    # Load user data
    try:
        if load_profile_file.name.endswith('.csv'):
            load_profile = pd.read_csv(load_profile_file)
        else:
            load_profile = pd.read_excel(load_profile_file)
        
        # Accept alternative column names for timestamp and load_kw
        timestamp_cols = ['timestamp', 'time', 'date', 'day']
        load_cols = ['load_kw', 'load', 'value', 'value_kw', 'last']

        found_timestamp_col = next((col for col in timestamp_cols if col in load_profile.columns), None)
        found_load_col = next((col for col in load_cols if col in load_profile.columns), None)

        if found_timestamp_col is None or found_load_col is None:
            st.write("Gefunden: " + found_timestamp_col)
            st.write("Gefunden: " +     found_load_col)
            st.error("Die Datei muss eine Zeitspalte ('timestamp', 'time', 'date', 'day') und eine Lastspalte ('load_kw', 'load', 'value', 'last') enthalten.")
            st.stop()

        # Rename columns to standard names for further processing
        load_profile = load_profile.rename(columns={found_timestamp_col: 'timestamp', found_load_col: 'load_kw'})
        # Convert timestamp column and ensure consistent datetime type - handle timezone-aware datetime objects
        load_profile['timestamp'] = pd.to_datetime(load_profile['timestamp'], utc=True).dt.tz_convert('UTC').dt.tz_localize(None)
        
    except Exception as e:
        st.error(f"Fehler beim Laden der Datei: {str(e)}")
        st.stop()
else:
    # Use sample data
    st.info("📊 Verwende Beispieldaten für die Demonstration. Laden Sie Ihre eigenen Daten hoch, um echte Analysen durchzuführen.")
    load_profile = generate_sample_load_profile()

# Generate supporting data
if pv_data_file is not None:
    try:
        pv_profile_generated = process_pv_data(pv_data_file, kwp_value)
        st.sidebar.success("✅ PV-Daten erfolgreich geladen und verarbeitet")
    except Exception as e:
        st.error(f"Fehler beim Laden der PV-Daten: {str(e)}")
        st.info("Verwende generierte PV-Daten als Fallback - bitte überprüfen!")
        pv_profile_generated = generate_pv_profile()
else:
    try:
        #pv_file = C:\Users\mgr\OneDrive\Dokumente\battery-project\battery-savings\input\solar_data_de_small.csv
        pv_profile_generated = process_pv_data("input/solar_data_de_small.csv", kwp_value)
        st.sidebar.success("✅ PV-Daten erfolgreich geladen")
    except Exception as e:
        st.error(f"Fehler beim Laden der PV-Daten: {str(e)}")
        st.info("Verwende generierte PV-Daten als Fallback - bitte überprüfen!")
        pv_profile_generated = generate_pv_profile()

# Load spot price data
if spot_price_file is not None:
    try:
        spot_prices = process_spot_price_data(spot_price_file)
        st.sidebar.success("✅ Spot-Preis-Daten erfolgreich geladen und verarbeitet")
        st.sidebar.info(f"Spot-Preis-Daten: {len(spot_prices)} Zeilen, Zeitraum: {spot_prices['timestamp'].min()} bis {spot_prices['timestamp'].max()}")
    except Exception as e:
        st.error(f"Fehler beim Laden der Spot-Preis-Daten: {str(e)}")
        st.info("Verwende generierte Spot-Preis-Daten als Fallback")
        spot_prices = generate_spot_prices()
else:
    # Try to load default spot price file
    default_spot_file = "input/spot_data_2024.xlsx"
    try:
        import os
        if os.path.exists(default_spot_file):
            spot_prices = process_spot_price_data(default_spot_file)
            #st.success("✅ Standard Spot-Preis-Daten geladen")
            #st.info(f"Spot-Preis-Daten: {len(spot_prices)} Zeilen, Zeitraum: {spot_prices['timestamp'].min()} bis {spot_prices['timestamp'].max()}")
        else:
            spot_prices = generate_spot_prices()
            st.info("📊 Verwende generierte Spot-Preis-Daten. Laden Sie Ihre eigenen Daten hoch für echte Analysen.")
    except Exception as e:
        spot_prices = generate_spot_prices()
        st.info("📊 Verwende generierte Spot-Preis-Daten. Laden Sie Ihre eigenen Daten hoch für echte Analysen.")

# Calculate scenarios
st.header("📈 Szenario-Analyse")

# Scenario 1: Baseline (no PV, no battery)
st.subheader("🎯 Szenario 1: Baseline (ohne PV und Batterie)")
baseline_demand_charge = calculate_demand_charge(load_profile, demand_charge_high, demand_charge_low)
baseline_energy_costs = calculate_energy_cost(load_profile, spot_prices, grid_fee, eeg_levy, vat_rate)
baseline_total_cost = baseline_energy_costs['total_cost'] + baseline_demand_charge

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Gesamtkosten", f"{baseline_total_cost:,.0f} €/Jahr")
with col2:
    st.metric("Energiekosten", f"{baseline_energy_costs['total_cost']:,.0f} €/Jahr")
with col3:
    st.metric("Leistungskosten", f"{baseline_demand_charge:,.0f} €/Jahr")
with col4:
    st.metric("Max. Last", f"{load_profile['load_kw'].max():.1f} kW")

# Scenario 2: PV only
st.subheader("☀️ Szenario 2: PV-Anlage")
pv_load = pv_profile_generated['pv_power_kw_per_kwp'] * kwp_value
net_load_pv = load_profile['load_kw'] - pv_load
consumed_load_pv = np.maximum(net_load_pv, 0)  # No negative grid load
surplus_load_pv = np.minimum(net_load_pv, 0) # negative grid load

# Create net load profile for PV scenario
net_load_profile_pv = pd.DataFrame({
    'timestamp': load_profile['timestamp'],
    'load_kw': net_load_pv
})

pv_demand_charge = calculate_demand_charge(net_load_profile_pv, demand_charge_high, demand_charge_low)
pv_energy_costs = calculate_energy_cost(net_load_profile_pv, spot_prices, grid_fee, eeg_levy, vat_rate)
pv_total_cost = pv_energy_costs['total_cost'] + pv_demand_charge



# Calculate PV metrics
pv_surplus = np.maximum(-(load_profile['load_kw'] - pv_load), 0)
total_pv_generation = pv_load.sum() * 0.25 / 1000  # MWh
total_pv_surplus = pv_surplus.sum() * 0.25 / 1000  # MWh
self_consumption_rate = (total_pv_generation - total_pv_surplus) / total_pv_generation * 100 if total_pv_generation > 0 else 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Gesamtkosten", f"{pv_total_cost:,.0f} €/Jahr", 
              delta=f"{baseline_total_cost - pv_total_cost:,.0f} €/Jahr")
with col2:
    st.metric("PV-Eigenverbrauch", f"{self_consumption_rate:.1f}%")
with col3:
    st.metric("PV-Überschuss", f"{total_pv_surplus:.1f} MWh/Jahr")
with col4:
    st.metric("Max. Netto-Last", f"{net_load_pv.max():.1f} kW")

# Scenario 3: PV + Simple Battery
st.subheader("🔋 Szenario 3: PV + Einfache Batterie-Optimierung")
simple_battery_results = simple_battery_optimization(load_profile, pv_profile_generated, battery_power, battery_capacity)

simple_battery_profile = pd.DataFrame({
    'timestamp': load_profile['timestamp'],
    'load_kw': simple_battery_results['final_grid_load']
})

simple_battery_demand_charge = calculate_demand_charge(simple_battery_profile, demand_charge_high, demand_charge_low)
simple_battery_energy_costs = calculate_energy_cost(simple_battery_profile, spot_prices, grid_fee, eeg_levy, vat_rate)
simple_battery_total_cost = simple_battery_energy_costs['total_cost'] + simple_battery_demand_charge

# Calculate battery metrics
battery_cycles = simple_battery_results['battery_charge'].sum() * 0.25 / battery_capacity
battery_efficiency = 0.9  # Assume 90% round-trip efficiency

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Gesamtkosten", f"{simple_battery_total_cost:,.0f} €/Jahr",
              delta=f"{baseline_total_cost - simple_battery_total_cost:,.0f} €/Jahr")
with col2:
    st.metric("Batterie-Zyklen", f"{battery_cycles:.0f}/Jahr")
with col3:
    st.metric("Eigenverbrauch", f"{(self_consumption_rate + 10):.1f}%")  # Approximate improvement
with col4:
    st.metric("Max. Netto-Last", f"{simple_battery_results['final_grid_load'].max():.1f} kW")

# Scenario 4: Smart Battery
st.subheader("🧠 Szenario 4: Intelligente Batterie-Optimierung")
smart_battery_results = smart_battery_optimization(load_profile, pv_profile_generated, battery_power, battery_capacity, spot_prices)

smart_battery_profile = pd.DataFrame({
    'timestamp': load_profile['timestamp'],
    'load_kw': smart_battery_results['final_grid_load']
})

smart_battery_demand_charge = calculate_demand_charge(smart_battery_profile, demand_charge_high, demand_charge_low)
smart_battery_energy_costs = calculate_energy_cost(smart_battery_profile, spot_prices, grid_fee, eeg_levy, vat_rate)
smart_battery_total_cost = smart_battery_energy_costs['total_cost'] + smart_battery_demand_charge

# Calculate smart battery metrics
smart_battery_cycles = smart_battery_results['battery_charge'].sum() * 0.25 / battery_capacity
peak_reduction = (load_profile['load_kw'].max() - smart_battery_results['final_grid_load'].max()) / load_profile['load_kw'].max() * 100

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Gesamtkosten", f"{smart_battery_total_cost:,.0f} €/Jahr",
              delta=f"{baseline_total_cost - smart_battery_total_cost:,.0f} €/Jahr")
with col2:
    st.metric("Peak-Reduktion", f"{peak_reduction:.1f}%")
with col3:
    st.metric("Batterie-Zyklen", f"{smart_battery_cycles:.0f}/Jahr")
with col4:
    st.metric("Max. Netto-Last", f"{smart_battery_results['final_grid_load'].max():.1f} kW")

# Comparison chart
st.header("📊 Kostenvergleich aller Szenarien")

scenarios_data = {
    'Szenario': ['Baseline', 'PV', 'PV + Batterie', 'Smart Batterie'],
    'Gesamtkosten (€/Jahr)': [
        baseline_total_cost,
        pv_total_cost,
        simple_battery_total_cost,
        smart_battery_total_cost
    ],
    'Energiekosten (€/Jahr)': [
        baseline_energy_costs['total_cost'],
        pv_energy_costs['total_cost'],
        simple_battery_energy_costs['total_cost'],
        smart_battery_energy_costs['total_cost']
    ],
    'Leistungspreis (€/Jahr)': [
        baseline_demand_charge,
        pv_demand_charge,
        simple_battery_demand_charge,
        smart_battery_demand_charge
    ]
}

scenarios_df = pd.DataFrame(scenarios_data)

# Create comparison chart
fig_comparison = go.Figure()

fig_comparison.add_trace(go.Bar(
    name='Energiekosten',
    x=scenarios_df['Szenario'],
    y=scenarios_df['Energiekosten (€/Jahr)'],
    marker_color='#1f77b4'
))

fig_comparison.add_trace(go.Bar(
    name='Leistungspreis',
    x=scenarios_df['Szenario'],
    y=scenarios_df['Leistungspreis (€/Jahr)'],
    marker_color='#ff7f0e'
))

fig_comparison.update_layout(
    title='Kostenvergleich aller Szenarien',
    xaxis_title='Szenario',
    yaxis_title='Kosten (€/Jahr)',
    barmode='stack',
    height=500
)

st.plotly_chart(fig_comparison, use_container_width=True)

# Detailed analysis
st.header("🔍 Detaillierte Analyse")

# Load profile visualization
tab1, tab2, tab3, tab4 = st.tabs(["Lastprofil", "PV-Profil", "Batterie-Analyse", "Kostendetails"])

with tab1:
    st.subheader("Jahreslastprofil")
    
    # Visualization options for this tab only
    use_daily_averages = st.checkbox(
        "Tägliche Durchschnittswerte verwenden", 
        value=True,
        help="Aktiviert: Zeigt tägliche Durchschnittswerte für bessere Übersicht. Deaktiviert: Zeigt alle 15-Minuten-Daten.",
        key="load_profile_checkbox"
    )
    
    # Use switch to determine data granularity
    if use_daily_averages:
        load_data = load_profile.set_index('timestamp')['load_kw'].resample('D').mean()
        title = 'Tägliche Durchschnittslast'
        y_label = 'Durchschnittslast (kW)'
    else:
        load_data = load_profile.set_index('timestamp')['load_kw']
        title = '15-Minuten-Lastprofil'
        y_label = 'Last (kW)'
    
    fig_load = px.line(
        x=load_data.index,
        y=load_data.values,
        title=title,
        labels={'x': 'Datum', 'y': y_label}
    )
    st.plotly_chart(fig_load, use_container_width=True)
    
    # Show statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Maximale Last", f"{load_profile['load_kw'].max():.1f} kW")
    with col2:
        st.metric("Durchschnittslast", f"{load_profile['load_kw'].mean():.1f} kW")
    with col3:
        st.metric("Jahresverbrauch", f"{load_profile['load_kw'].sum() * 0.25 / 1000:.1f} MWh")

with tab2:
    st.subheader("PV-Profil und Eigenverbrauch")
    
    # Visualization options for this tab only
    use_daily_averages = st.checkbox(
        "Tägliche Durchschnittswerte verwenden", 
        value=True,
        help="Aktiviert: Zeigt tägliche Durchschnittswerte für bessere Übersicht. Deaktiviert: Zeigt alle 15-Minuten-Daten.",
        key="pv_profile_checkbox"
    )
    
    # Show PV generation vs consumption
    if use_daily_averages:
        pv_data = pv_profile_generated.set_index('timestamp')['pv_power_kw_per_kwp'].resample('D').mean() * kwp_value
        consumption_data = load_profile.set_index('timestamp')['load_kw'].resample('D').mean()
        title = 'PV-Generierung vs. Verbrauch (Tagesdurchschnitt)'
    else:
        pv_data = pv_profile_generated.set_index('timestamp')['pv_power_kw_per_kwp'] * kwp_value
        consumption_data = load_profile.set_index('timestamp')['load_kw']
        title = 'PV-Generierung vs. Verbrauch (15-Minuten-Intervalle)'
    
    fig_pv = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_pv.add_trace(
        go.Scatter(x=pv_data.index, y=pv_data.values, name="PV-Generierung", line=dict(color='orange')),
        secondary_y=False
    )
    
    fig_pv.add_trace(
        go.Scatter(x=consumption_data.index, y=consumption_data.values, name="Verbrauch", line=dict(color='blue')),
        secondary_y=True
    )
    
    fig_pv.update_layout(title=title)
    fig_pv.update_xaxes(title_text='Datum')
    fig_pv.update_yaxes(title_text='PV-Leistung (kW)', secondary_y=False)
    fig_pv.update_yaxes(title_text='Verbrauch (kW)', secondary_y=True)
    
    st.plotly_chart(fig_pv, use_container_width=True)

with tab3:
    st.subheader("Batterie-Performance (Smart Battery)")
    
    # Visualization options for this tab only
    use_daily_averages = st.checkbox(
        "Tägliche Durchschnittswerte verwenden", 
        value=True,
        help="Aktiviert: Zeigt tägliche Durchschnittswerte für bessere Übersicht. Deaktiviert: Zeigt alle 15-Minuten-Daten.",
        key="battery_profile_checkbox"
    )
    
    # Show battery SOC over time
    fig_battery = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Batterie-Ladezustand', 'Batterie-Leistung'),
        vertical_spacing=0.1
    )
    
    # Use switch to determine data granularity and time period
    if use_daily_averages:
        # Use more data points for daily averages
        data_points = min(30 * 96, len(smart_battery_results['battery_soc']))  # 30 days
        title_suffix = " (30 Tage)"
    else:
        # Use a week of data for detailed view
        data_points = min(7 * 96, len(smart_battery_results['battery_soc']))  # 7 days
        title_suffix = " (1 Woche)"
    
    battery_soc_data = smart_battery_results['battery_soc'][:data_points]
    battery_charge_data = smart_battery_results['battery_charge'][:data_points]
    battery_discharge_data = smart_battery_results['battery_discharge'][:data_points]
    timestamps = load_profile['timestamp'][:data_points]
    
    # Resample if using daily averages
    if use_daily_averages:
        # Create a temporary dataframe for resampling
        temp_df = pd.DataFrame({
            'timestamp': timestamps,
            'soc': battery_soc_data,
            'charge': battery_charge_data,
            'discharge': battery_discharge_data
        }).set_index('timestamp')
        
        resampled_df = temp_df.resample('D').mean()
        battery_soc_data = resampled_df['soc'].values
        battery_charge_data = resampled_df['charge'].values
        battery_discharge_data = resampled_df['discharge'].values
        timestamps = resampled_df.index
    
    fig_battery.add_trace(
        go.Scatter(x=timestamps, y=battery_soc_data, name="SOC", line=dict(color='green')),
        row=1, col=1
    )
    
    fig_battery.add_trace(
        go.Scatter(x=timestamps, y=battery_charge_data, 
                  name="Laden", line=dict(color='blue')),
        row=2, col=1
    )
    
    fig_battery.add_trace(
        go.Scatter(x=timestamps, y=battery_discharge_data, 
                  name="Entladen", line=dict(color='red')),
        row=2, col=1
    )
    
    fig_battery.update_layout(height=600, title_text=f"Batterie-Performance{title_suffix}")
    st.plotly_chart(fig_battery, use_container_width=True)

with tab4:
    st.subheader("Detaillierte Kostenaufschlüsselung")
    
    # Create detailed cost breakdown
    cost_breakdown = pd.DataFrame({
        'Szenario': ['Baseline', 'PV', 'PV + Batterie', 'Smart Batterie'],
        'Energiekosten (€/Jahr)': [
            baseline_energy_costs['energy_cost'],
            pv_energy_costs['energy_cost'],
            simple_battery_energy_costs['energy_cost'],
            smart_battery_energy_costs['energy_cost']
        ],
        'Netzentgelt (€/Jahr)': [
            baseline_energy_costs['grid_cost'],
            pv_energy_costs['grid_cost'],
            simple_battery_energy_costs['grid_cost'],
            smart_battery_energy_costs['grid_cost']
        ],
        'EEG-Umlage (€/Jahr)': [
            baseline_energy_costs['eeg_cost'],
            pv_energy_costs['eeg_cost'],
            simple_battery_energy_costs['eeg_cost'],
            smart_battery_energy_costs['eeg_cost']
        ],
        'MwSt. (€/Jahr)': [
            baseline_energy_costs['vat_amount'],
            pv_energy_costs['vat_amount'],
            simple_battery_energy_costs['vat_amount'],
            smart_battery_energy_costs['vat_amount']
        ],
        'Leistungspreis (€/Jahr)': [
            baseline_demand_charge,
            pv_demand_charge,
            simple_battery_demand_charge,
            smart_battery_demand_charge
        ]
    })
    
    st.dataframe(cost_breakdown, use_container_width=True)
    
    # Savings analysis
    st.subheader("Einsparungsanalyse")
    
    savings_data = {
        'Szenario': ['PV', 'PV + Batterie', 'Smart Batterie'],
        'Einsparung vs. Baseline (€/Jahr)': [
            baseline_total_cost - pv_total_cost,
            baseline_total_cost - simple_battery_total_cost,
            baseline_total_cost - smart_battery_total_cost
        ],
        'Einsparung (%)': [
            (baseline_total_cost - pv_total_cost) / baseline_total_cost * 100,
            (baseline_total_cost - simple_battery_total_cost) / baseline_total_cost * 100,
            (baseline_total_cost - smart_battery_total_cost) / baseline_total_cost * 100
        ]
    }
    
    savings_df = pd.DataFrame(savings_data)
    st.dataframe(savings_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>⚡ ecoplanet Energie-Kosten-Analyse Dashboard | Entwickelt für deutsche Industrieunternehmen</p>
    <p>Hinweis: Diese Analyse basiert auf den eingegebenen Parametern und Annahmen. Für präzise Geschäftsentscheidungen sollten detaillierte Studien durchgeführt werden.</p>
</div>
""", unsafe_allow_html=True)
