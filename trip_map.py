import streamlit as st
from keplergl import KeplerGl
from streamlit_keplergl import keplergl_static
import pandas as pd

# Load the data from CSV
grouped_df = pd.read_csv("lime_maphex.csv")
grouped_df = grouped_df.drop(columns="Unnamed: 0")  # Dropping unwanted column

# Create a Kepler.gl map
map_1 = KeplerGl(height=800, width=1200)

# Add data to the map
map_1.add_data(data=grouped_df, name="Trip Data")

# Define the configuration for the map
config = {
    'version': 'v1',
    'config': {
        'mapState': {
            'bearing': 0,
            'dragRotate': True,
            'latitude': 41.8781,  # Set to the center latitude of your data
            'longitude': -87.6298,  # Set to the center longitude of your data
            'pitch': 0,
            'zoom': 9  # Adjust zoom level as needed
        }
    }
}

# Apply the configuration to the map
map_1.config = config

# Streamlit app setup
st.title('Trip Data Visualization')

# Render the Kepler.gl map in Streamlit
keplergl_static(map_1)