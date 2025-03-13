import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import pickle

# --------------------------------------------------------------
# 1. Load Model and Demand Data
# --------------------------------------------------------------
with open("demand_model.pkl", "rb") as f:
    demand_model = pickle.load(f)

# X_demand: Feature DataFrame used for training.
X_demand = pd.read_csv("X_demand.csv", index_col=0, parse_dates=True)
model_features = X_demand.columns.tolist()

# Y_demand: Has the target columns (hex IDs) for wide -> long meltdown.
Y_demand = pd.read_csv("Y_demand.csv")
target_columns = Y_demand.columns.tolist()

# --------------------------------------------------------------
# 2. Baseline Weather Values
# --------------------------------------------------------------
BASE_TEMP = 15.0      # °C
BASE_HUMIDITY = 70    # %
BASE_WIND = 5.0       # m/s
BASE_RAIN = 0.0       # mm/h
BASE_CLOUDS = 50      # %

# --------------------------------------------------------------
# 3. Build Scenario Predictions (Demand)
# --------------------------------------------------------------
def build_scenario_predictions(time_bin, weekday, temp, humidity, wind, rain, clouds, selected_team):
    # 1) Create a single-row from median values.
    median_values = {col: X_demand[col].median() for col in model_features}
    df_scenario = pd.DataFrame([median_values], columns=model_features)

    # 2) Override hour, weather, etc. if they exist.
    if "hour" in df_scenario.columns:
        df_scenario["hour"] = time_bin
    if "day_of_week" in df_scenario.columns:
        df_scenario["day_of_week"] = weekday
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

    # If a team is selected, set that one to 1.
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
    pred_array = demand_model.predict(df_scenario)
    pred_df = pd.DataFrame(pred_array, columns=target_columns, index=[0])

    # 7) Melt wide -> long.
    pred_long = pred_df.melt(var_name="hex_id", value_name="pred_demand")
    return pred_long

# --------------------------------------------------------------
# 4. Color Function & Deck Construction
# --------------------------------------------------------------
def compute_rgba(value, min_val, max_val):
    """
    Color scale:
      - Negative values: green -> red
      - Zero: green
      - Positive values: green -> blue
    """
    red   = np.array([255,   0,   0, 250])
    green = np.array([  0, 255,   0, 250])
    blue  = np.array([  0,   0, 255, 250])

    if np.isclose(value, 0.0):
        return tuple(green)

    if value < 0:
        # Ratio in [-6, 0]
        ratio = np.clip(value / -6, 0, 1)
        color = green + ratio * (red - green)
        return tuple(color.astype(int))
    else:
        # Ratio in [0, 6]
        ratio = np.clip(value / 6, 0, 1)
        color = green + ratio * (blue - green)
        return tuple(color.astype(int))

def build_deck_for_time_bin(time_bin, weekday, temp, humidity, wind, rain, clouds, selected_team):
    """
    1) Predict demand from medians + overrides.
    2) Use the sign for color (negative=red, positive=blue, zero=green).
    3) Use the absolute value of demand for extrusion height.
    """
    pred_long = build_scenario_predictions(time_bin, weekday, temp, humidity, wind, rain, clouds, selected_team)
    if pred_long.empty:
        vs = pdk.ViewState(latitude=41.8781, longitude=-87.6298, zoom=10, pitch=45)
        return pdk.Deck(layers=[], initial_view_state=vs, tooltip={"text": "No data"})

    # Use predicted demand directly.
    pred_long['adj_demand'] = pred_long['pred_demand']

    # For color scaling.
    cmin, cmax = pred_long['adj_demand'].min(), pred_long['adj_demand'].max()

    # 1) Assign color based on sign.
    color_df = pred_long.apply(
        lambda row: compute_rgba(row['adj_demand'], cmin, cmax),
        axis=1, result_type='expand'
    )
    pred_long[['colorR','colorG','colorB','colorA']] = color_df

    # 2) Use the absolute value for extrusion height.
    pred_long['elev'] = pred_long['adj_demand'].abs()

    data = pred_long.to_dict(orient="records")

    hex_layer = pdk.Layer(
        "H3HexagonLayer",
        data=data,
        get_hexagon="hex_id",
        get_elevation="elev",  # Use absolute demand for height.
        elevation_scale=500,
        extruded=True,
        coverage=1,
        pickable=True,
        get_fill_color=["colorR","colorG","colorB","colorA"]
    )

    view_state = pdk.ViewState(latitude=41.8781, longitude=-87.6298, zoom=10, pitch=45)
    deck = pdk.Deck(
        layers=[hex_layer],
        initial_view_state=view_state,
        map_provider="carto",
        map_style="light",
        tooltip={"text": "Hex {hex_id}\nPredicted Demand: {adj_demand}"}
    )
    return deck

# --------------------------------------------------------------
# 5. Streamlit App Setup with Sidebar Controls
# --------------------------------------------------------------

st.title("Demand Scenario Deck")

# Define constant options.
TIME_BINS = [2.5, 8.5, 14.5, 20.5]
TEAM_OPTIONS = ["No Team", "ChicagoBulls", "FireFC", "StarsFC"]

# Sidebar controls.
st.sidebar.header("Input Parameters")
time_bin = st.sidebar.selectbox("Time Bin", options=TIME_BINS, index=0)
weekday = st.sidebar.slider("Weekday", min_value=0, max_value=6, value=0, step=1)
temp = st.sidebar.slider("Temp (°C)", min_value=-10.0, max_value=40.0, value=BASE_TEMP, step=0.5)
humidity = st.sidebar.slider("Humidity (%)", min_value=0, max_value=100, value=BASE_HUMIDITY, step=1)
wind = st.sidebar.slider("Wind (m/s)", min_value=0.0, max_value=20.0, value=BASE_WIND, step=0.5)
rain = st.sidebar.slider("Rain (mm/h)", min_value=0.0, max_value=10.0, value=BASE_RAIN, step=0.1)
clouds = st.sidebar.slider("Clouds (%)", min_value=0, max_value=100, value=BASE_CLOUDS, step=1)
team = st.sidebar.selectbox("Team", options=TEAM_OPTIONS, index=0)
predict_button = st.sidebar.button("Predict")

# Display the pydeck chart.
if predict_button:
    deck = build_deck_for_time_bin(time_bin, weekday, temp, humidity, wind, rain, clouds, team)
    st.pydeck_chart(deck)
else:
    # Display a default deck.
    default_deck = build_deck_for_time_bin(
        TIME_BINS[0],
        weekday=0,
        temp=BASE_TEMP,
        humidity=BASE_HUMIDITY,
        wind=BASE_WIND,
        rain=BASE_RAIN,
        clouds=BASE_CLOUDS,
        selected_team="No Team"
    )
    st.pydeck_chart(default_deck)
