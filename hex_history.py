
import pandas as pd
import pydeck as pdk
import streamlit as st
import panel as pn
from shapely import wkt
from shapely.geometry import mapping
import h3
import gdown

# Cache Data
@st.cache_data
def load_and_preprocess_data():
    # Load CSVs (Assuming you've uploaded them manually)
    lime_df = pd.read_csv('lime_data.csv')
    lime_df['Start Time'] = pd.to_datetime(lime_df['Start Time'], errors='coerce')

    # Group by date and location and count trips
    spt_daily_trips = lime_df.groupby(
        [lime_df['Start Time'].dt.date, 'Start Centroid Location']
    ).agg(trip_count=('Trip ID', 'count')).reset_index()
    spt_daily_trips.rename(columns={'Start Time': 'date'}, inplace=True)

    # Extract lat/lon from WKT
    spt_daily_trips['lat'] = spt_daily_trips['Start Centroid Location'].apply(lambda pt: wkt.loads(pt).y)
    spt_daily_trips['lon'] = spt_daily_trips['Start Centroid Location'].apply(lambda pt: wkt.loads(pt).x)

    # H3 Conversion
    resolution = 7
    trips_h3 = []
    for idx, row in spt_daily_trips.iterrows():
        lon = row["lon"]
        lat = row["lat"]
        hex_id = h3.latlng_to_cell(lat, lon, resolution)
        trips_h3.append({
            "hex_id": hex_id,
            "trip_count": row["trip_count"],
            "date": row["date"]
        })
    
    df_hex = pd.DataFrame(trips_h3)
    hex_trips = df_hex.groupby(["hex_id", "date"])["trip_count"].sum().reset_index()
    hex_trips.columns = ["hex_id", "date", "trip_count"]

    # Get the map data of Chicago
    url = 'https://drive.google.com/uc?id=1KrUyLtr8_KN_ZXJIjxVMS_yANg2yNYz8'
    output = 'CommAreas_20250226.csv'
    gdown.download(url, output, quiet=False)

    # Load the community areas into a dataframe
    comm_areas = pd.read_csv(output)
    
    return hex_trips, comm_areas

# Cache the processed data
hex_trips, comm_areas = load_and_preprocess_data()

# Cache monthly data processing
@st.cache_data
def get_monthly_groups():
    hex_trips["date"] = pd.to_datetime(hex_trips["date"], errors="coerce")
    hex_trips["year_month"] = hex_trips["date"].dt.to_period("M").astype(str)
    monthly_data = hex_trips.groupby(["hex_id", "year_month"], as_index=False)["trip_count"].sum()
    return {ym: df.drop(columns="year_month") for ym, df in monthly_data.groupby("year_month")}

monthly_groups = get_monthly_groups()
available_months = sorted(monthly_groups.keys())

# Simplify geometries if necessary for performance
@st.cache_data
def process_geojson(comm_areas):
    features = []
    for _, row in comm_areas.iterrows():
        shape = wkt.loads(row["the_geom"])
        simplified_shape = shape.simplify(0.01)  # Simplify with tolerance
        geom_geojson = mapping(simplified_shape)
        feature = {"type": "Feature", "properties": {}, "geometry": geom_geojson}
        features.append(feature)

    return {"type": "FeatureCollection", "features": features}

chicago_fc = process_geojson(comm_areas)

# Function to compute RGBA color
def compute_rgba(count, cmin, cmax):
    if cmax == cmin:
        return (128, 128, 128, 255)
    ratio = (count - cmin) / (cmax - cmin)
    return (int(255 * ratio), int(255 * (1 - ratio)), 0, 255)

# Function to build Pydeck map
def build_deck_for_month(ym_str):
    df = monthly_groups.get(ym_str, pd.DataFrame({"hex_id": [], "trip_count": []}))
    cmin, cmax = df["trip_count"].min(), df["trip_count"].max() if not df.empty else (0, 1)

    df = df.copy()
    df[["colorR", "colorG", "colorB", "colorA"]] = df.apply(lambda row: compute_rgba(row["trip_count"], cmin, cmax), axis=1, result_type="expand")

    hex_layer = pdk.Layer(
        "H3HexagonLayer",
        data=df.to_dict(orient="records"),
        get_hexagon="hex_id",
        get_elevation="trip_count",
        elevation_scale=0.5,
        extruded=True,
        coverage=1,
        pickable=True,
        auto_highlight=False,
        get_fill_color=["colorR", "colorG", "colorB", "colorA"],
        opacity=0.8
    )

    boundary_layer = pdk.Layer(
        "GeoJsonLayer",
        data=chicago_fc,
        stroked=True,
        filled=True,
        get_line_color=[80, 80, 80],
        get_fill_color=[200, 200, 200],
        opacity=1.0
    )

    view_state = pdk.ViewState(latitude=41.8781, longitude=-87.6298, zoom=10, pitch=45)
    return pdk.Deck(layers=[boundary_layer, hex_layer], initial_view_state=view_state, map_provider="carto", map_style="light")

# Display Interactive Plot
selected_month = st.selectbox("Select Month", available_months)
st.pydeck_chart(build_deck_for_month(selected_month))