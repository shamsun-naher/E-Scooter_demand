import streamlit as st
from keplergl import KeplerGl
from streamlit_keplergl import keplergl_static
import pandas as pd

# --------------------------------------------------------------
# 1. Load and Prepare Data
# --------------------------------------------------------------
df = pd.read_csv("lime_maphex.csv")
df = df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], errors="ignore")

# --------------------------------------------------------------
# 2. Create a Kepler.gl Map and Add Data
# --------------------------------------------------------------
map_1 = KeplerGl(height=800, width=1200)
map_1.add_data(data=df, name="Trip Data")

# --------------------------------------------------------------
# 3. Define Configuration with One ArcLayer using Trip Count
#    for both stroke thickness and color, plus a Hex Layer for context.
# --------------------------------------------------------------
config = {
    "version": "v1",
    "config": {
        "visState": {
            "layers": [
                # Single ArcLayer using tripcounts for both color and stroke thickness.
                {
                    "id": "arc_layer_tripcounts",
                    "type": "arc",
                    "config": {
                        "dataId": "Trip Data",
                        "label": "Trips by Count",
                        "columns": {
                            "lat0": "Start Centroid Latitude",
                            "lng0": "Start Centroid Longitude",
                            "lat1": "End Centroid Latitude",
                            "lng1": "End Centroid Longitude"
                        },
                        "isVisible": True,
                        "visConfig": {
                            "opacity": 0.8,
                            "thickness": 2,
                            # Increase sizeRange to exaggerate stroke differences
                            "sizeRange": [0, 50],
                            # Define a sequential color range mapping low to high trip counts.
                            "colorRange": {
                                "name": "Global Sequential",
                                "type": "sequential",
                                "category": "Uber",
                                "colors": ["#87CEFA", "#FF4500"]  # Light blue to orange-red
                            },
                            "targetColor": None
                        },
                        "hidden": False,
                        "textLabel": []
                    },
                    "visualChannels": {
                        "weightField": {"name": "tripcounts", "type": "integer"},
                        "weightScale": "log",
                        "colorField": {"name": "tripcounts", "type": "integer"},
                        "colorScale": "log"
                    }
                },
                # Hex Layer for context (using native H3 recognition from start_hex)
                {
                    "id": "hex_layer",
                    "type": "hexagonId",
                    "config": {
                        "dataId": "Trip Data",
                        "label": "Hexagons",
                        "columns": {
                            "hex_id": "start_hex"
                        },
                        "isVisible": True,
                        "visConfig": {
                            "opacity": 0.1,
                            "strokeColor": [200, 200, 200],
                            "strokeWidth": 1,
                            "fillColor": [150, 150, 150]
                        },
                        "hidden": False,
                        "textLabel": []
                    },
                    "visualChannels": {}
                }
            ],
            "interactionConfig": {
                "tooltip": {
                    "fieldsToShow": {
                        "Trip Data": ["tripcounts", "start_hex", "end_hex"]
                    },
                    "enabled": True
                },
                "brush": {"enabled": True},
                "geocoder": {"enabled": False},
                "coordinate": {"enabled": False}
            }
        },
        "mapState": {
            "bearing": 0,
            "dragRotate": True,
            "latitude": 41.8781,
            "longitude": -87.6298,
            "pitch": 0,
            "zoom": 9
        }
    }
}

# Apply the configuration.
map_1.config = config

# --------------------------------------------------------------
# 4. Render the Map in Streamlit
# --------------------------------------------------------------
st.title("Trip Data with Trip Count-Based Color and Stroke")
keplergl_static(map_1)
