import pandas as pd
import streamlit as st
import pydeck as pdk
import plotly.express as px
import numpy as np

# --------------------------------
# Reduce font sizes to avoid cut-off
# --------------------------------
st.set_page_config(layout="wide")
st.markdown("""
<style>
/* Reduce metric label font size */
[data-testid="stMetricLabel"] {
    font-size: 0.8rem;
}
/* Reduce metric value font size */
[data-testid="stMetricValue"] {
    font-size: 1rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 1. Load and Preprocess Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("hex_toolpin.csv")
    # Convert "year_month" to datetime (day=1)
    df['year_month'] = df['year_month'].astype(str).str.strip()
    df['year_month'] = pd.to_datetime(df['year_month'], format="%Y-%m")
    return df

data = load_data()

# -------------------------------
# 2. Dynamic Color for Trip Count
# -------------------------------
def compute_rgba(count, cmin, cmax):
    # If all counts identical, return neutral gray
    if cmax == cmin:
        return (128, 128, 128, 255)
    ratio = (count - cmin) / (cmax - cmin)
    # Low => green, High => red
    r = int(255 * ratio)
    g = int(255 * (1 - ratio))
    b = 0
    a = 250
    return (r, g, b, a)

# -------------------------------
# 3. Build Map Layer for a Given Month
# -------------------------------
def build_deck_for_month(ym, highlight_hex=None):
    df_month = data[data['year_month'] == ym].copy()
    if not df_month.empty:
        cmin = df_month["trip_count"].min()
        cmax = df_month["trip_count"].max()
    else:
        cmin, cmax = 0, 1
    
    # Compute fill color columns
    color_cols = df_month.apply(lambda row: pd.Series(compute_rgba(row["trip_count"], cmin, cmax)), axis=1)
    color_cols.columns = ["colorR", "colorG", "colorB", "colorA"]
    df_month = pd.concat([df_month, color_cols], axis=1)
    
    # Highlight selected hex (bright blue)
    if highlight_hex is not None:
        df_month.loc[df_month['hex_id'] == highlight_hex, ["colorR","colorG","colorB","colorA"]] = [0, 0, 255, 255]
    
    # Create H3 layer
    hex_layer = pdk.Layer(
        "H3HexagonLayer",
        data=df_month,
        get_hexagon="hex_id",
        get_elevation="trip_count",
        elevation_scale=0.5,
        extruded=True,
        coverage=1,
        pickable=True,
        get_fill_color=["colorR", "colorG", "colorB", "colorA"],
        opacity=1.0
    )
    view_state = pdk.ViewState(latitude=41.8781, longitude=-87.6298, zoom=10, pitch=45)
    deck = pdk.Deck(
        layers=[hex_layer],
        initial_view_state=view_state,
        map_provider="carto",
        map_style="light"
    )
    return deck, df_month

# -------------------------------
# 4. Sidebar Controls
# -------------------------------
st.sidebar.title("Controls")

available_months = sorted(data['year_month'].dt.strftime('%Y-%m').unique())
selected_month_str = st.sidebar.selectbox("Select Month", available_months)
selected_month = pd.to_datetime(selected_month_str, format="%Y-%m")

# Build deck with no highlight first, so we can get the monthly data
_, df_month = build_deck_for_month(selected_month)

if df_month.empty:
    st.sidebar.info("No hexagon data available for the selected month.")
    st.error("No hexagon data available for the selected month.")
else:
    # Sort hexes by trip_count (descending)
    df_month_sorted = df_month.sort_values('trip_count', ascending=False)
    hex_ids = df_month_sorted["hex_id"].unique()
    slider_index = st.sidebar.slider("Select Hex (by height ranking)", 
                                     min_value=0, max_value=len(hex_ids)-1, 
                                     value=0, step=1)
    selected_hex = hex_ids[slider_index]

    # -------------------------------
    # 5. Map + Info Side-by-Side
    # -------------------------------
    deck, df_month = build_deck_for_month(selected_month, highlight_hex=selected_hex)
    col_map, col_info = st.columns([3, 2])
    
    with col_map:
        st.pydeck_chart(deck)
    
    with col_info:
        st.subheader("Hexagon Details")
        # Instead of raw hex_id, show a label: "Hexagon #X of Y"
        hex_label = f"Hexagon #{slider_index+1} of {len(hex_ids)}"
        st.markdown(f"**Selected {hex_label}**")
        
        hex_data = df_month[df_month["hex_id"] == selected_hex].iloc[0]
        
        # Shorter metric labels to avoid cutting off
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trips", f"{int(hex_data['trip_count'])}")
        c2.metric("Avg Dist", f"{hex_data['avg_distance']:.0f}m")
        c3.metric("Avg Dur", f"{hex_data['avg_duration']:.0f}s")
        c4.metric("Net Acc", f"{int(hex_data['net_accumulation'])}")
        
        # ------------------------------------------
        # QUICK WORKAROUND: Re-label columns in donut chart
        # local_trips => "Incoming"
        # incoming_trips => "Local"
        # outgoing_trips => "Outgoing"
        # ------------------------------------------
        breakdown_df = pd.DataFrame({
            "Trip Type": ["Incoming", "Outgoing", "Local"],
            "Count": [
                hex_data["local_trips"],      # Show local_trips as "Incoming"
                hex_data["outgoing_trips"],   # Outgoing stays the same
                hex_data["incoming_trips"]    # Show incoming_trips as "Local"
            ]
        })
        
        fig = px.pie(
            breakdown_df, 
            names="Trip Type", 
            values="Count", 
            hole=0.5, 
            title="Trip Breakdown"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly summary
        st.subheader("Monthly Summary")
        total_trips = df_month["trip_count"].sum()
        net_total = (df_month["incoming_trips"] - df_month["outgoing_trips"]).sum()
        avg_trips = df_month["trip_count"].mean()
        c_s1, c_s2, c_s3 = st.columns(3)
        c_s1.metric("Trips", f"{int(total_trips)}")
        c_s2.metric("Net Acc", f"{int(net_total)}")
        c_s3.metric("Avg/Hex", f"{avg_trips:.1f}")
        
        # Add a note if net_total is not zero
        if net_total != 0:
            st.warning("Net Accumulation ≠ 0. Data may not be fully closed.")
