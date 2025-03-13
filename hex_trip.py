import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import pickle

# --------------------------------------------------------------
# 1. Load Model and Feature Data
# --------------------------------------------------------------
# Load the XGBoost model.
with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

# Load the feature DataFrame used for training.
X_hex = pd.read_csv("X_hex.csv", index_col=0, parse_dates=True)
model_features = X_hex.columns.tolist()

# Load target column names.
Y_hex = pd.read_csv("Y_hex.csv")
target_columns = Y_hex.columns.tolist()

# --------------------------------------------------------------
# 2. Baseline Weather Values
# --------------------------------------------------------------
BASE_TEMP = 15.0      # °C, defined as float
BASE_HUMIDITY = 70    # %, defined as integer for slider with step=1
BASE_WIND = 5.0       # m/s, defined as float
BASE_RAIN = 0.0       # mm/h, defined as float
BASE_CLOUDS = 50      # %, defined as integer for slider with step=1

# --------------------------------------------------------------
# 3. Build Scenario Predictions
# --------------------------------------------------------------
def build_scenario_predictions(hour, day_of_week, temp, humidity, wind, rain, clouds, selected_team=None):
    """
    Creates a one-row scenario DataFrame using median values,
    then overrides hour, weekday, weather columns, team flags, etc.
    Finally, returns the predictions melted into long format.
    """
    # 1) Create a single-row from median values.
    median_values = {col: X_hex[col].median() for col in model_features}
    df_scenario = pd.DataFrame([median_values], columns=model_features)

    # 2) Override columns if they exist.
    if "hour" in df_scenario.columns:
        df_scenario["hour"] = hour
    if "day_of_week" in df_scenario.columns:
        df_scenario["day_of_week"] = day_of_week
    if "temp" in df_scenario.columns:
        df_scenario["temp"] = temp
    if "humidity" in df_scenario.columns:
        df_scenario["humidity"] = humidity
    if "wind_speed" in df_scenario.columns:
        df_scenario["wind_speed"] = wind
    if "rain_1h" in df_scenario.columns:
        df_scenario["rain_1h"] = rain
    if "clouds_all" in df_scenario.columns:
        df_scenario["clouds_all"] = clouds

    # 3) Override baseline if needed.
    if "baseline" in df_scenario.columns:
        df_scenario["baseline"] = 1000

    # 4) Set all team columns to 0 if they exist.
    if "Team_ChicagoBulls" in df_scenario.columns:
        df_scenario["Team_ChicagoBulls"] = 0
    if "Team_FireFC" in df_scenario.columns:
        df_scenario["Team_FireFC"] = 0
    if "Team_StarsFC" in df_scenario.columns:
        df_scenario["Team_StarsFC"] = 0

    # If a team is selected, set that team flag to 1.
    if selected_team == "ChicagoBulls" and "Team_ChicagoBulls" in df_scenario.columns:
        df_scenario["Team_ChicagoBulls"] = 1
    elif selected_team == "FireFC" and "Team_FireFC" in df_scenario.columns:
        df_scenario["Team_FireFC"] = 1
    elif selected_team == "StarsFC" and "Team_StarsFC" in df_scenario.columns:
        df_scenario["Team_StarsFC"] = 1

    # 5) Ensure all expected columns exist.
    missing_cols = set(model_features) - set(df_scenario.columns)
    for col in missing_cols:
        df_scenario[col] = 0
    df_scenario = df_scenario[model_features]

    # 6) Predict using the model.
    pred_array = xgb_model.predict(df_scenario)
    pred_df = pd.DataFrame(pred_array, columns=target_columns, index=[0])

    # 7) Melt wide -> long.
    pred_long = pred_df.melt(var_name="hex_id", value_name="pred_trip")
    return pred_long

# --------------------------------------------------------------
# 4. Build Pydeck Deck for the Hourly Scenario
# --------------------------------------------------------------
def compute_rgba(count, cmin, cmax):
    """Map 'count' to an RGBA color from green to red."""
    if cmax == cmin:
        return (128, 128, 128, 255)
    ratio = (count - cmin) / (cmax - cmin)
    r = int(255 * ratio)
    g = int(255 * (1 - ratio))
    return (r, g, 0, 255)

def build_deck_for_hour(selected_hour, day_of_week, temp, humidity, wind, rain, clouds):
    """
    Generate a pydeck.Deck object using predictions from the scenario.
    """
    pred_long = build_scenario_predictions(
        selected_hour, day_of_week, temp, humidity, wind, rain, clouds, selected_team=None
    )

    if pred_long.empty:
        view_state = pdk.ViewState(latitude=41.8781, longitude=-87.6298, zoom=10, pitch=45)
        return pdk.Deck(layers=[], initial_view_state=view_state)

    cmin, cmax = pred_long['pred_trip'].min(), pred_long['pred_trip'].max()

    def assign_color(row):
        r, g, b, a = compute_rgba(row['pred_trip'], cmin, cmax)
        return pd.Series({"colorR": r, "colorG": g, "colorB": b, "colorA": a})

    color_df = pred_long.apply(assign_color, axis=1)
    pred_long = pd.concat([pred_long, color_df], axis=1)
    data = pred_long.to_dict(orient="records")

    hex_layer = pdk.Layer(
        "H3HexagonLayer",
        data=data,
        get_hexagon="hex_id",
        get_elevation="pred_trip",
        elevation_scale=100,  # Amplify hex heights
        extruded=True,
        coverage=1,
        pickable=True,
        get_fill_color=["colorR", "colorG", "colorB", "colorA"]
    )

    view_state = pdk.ViewState(latitude=41.8781, longitude=-87.6298, zoom=10, pitch=45)
    deck = pdk.Deck(
        layers=[hex_layer],
        initial_view_state=view_state,
        map_provider="carto",
        map_style="light",
        tooltip={"text": "Hex {hex_id}\nPredicted Trips: {pred_trip}"}
    )
    return deck

# --------------------------------------------------------------
# 5. Streamlit App Setup
# --------------------------------------------------------------
st.title("Scooter Trips: Hourly 3D Hex Map Scenario Predictions")

# Use the sidebar for input parameters.
st.sidebar.header("Input Parameters")
hour = st.sidebar.slider("Hour", min_value=0, max_value=23, value=0, step=1)
weekday = st.sidebar.slider("Weekday (0=Mon, 6=Sun)", min_value=0, max_value=6, value=0, step=1)
temp = st.sidebar.slider("Temp (°C)", min_value=-10.0, max_value=40.0, value=BASE_TEMP, step=0.5)
humidity = st.sidebar.slider("Humidity (%)", min_value=0, max_value=100, value=BASE_HUMIDITY, step=1)
wind = st.sidebar.slider("Wind (m/s)", min_value=0.0, max_value=20.0, value=BASE_WIND, step=0.5)
rain = st.sidebar.slider("Rain (mm/h)", min_value=0.0, max_value=10.0, value=BASE_RAIN, step=0.1)
clouds = st.sidebar.slider("Clouds (%)", min_value=0, max_value=100, value=BASE_CLOUDS, step=1)

# Build the pydeck chart using the input parameters.
deck = build_deck_for_hour(hour, weekday, temp, humidity, wind, rain, clouds)

# Display the deck in the main app.
st.pydeck_chart(deck)