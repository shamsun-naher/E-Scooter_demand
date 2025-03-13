import streamlit as st


# --- PAGE SETUP ---
about_page = st.Page(
    "e-scooter.py",
    title="E-scooter",
    icon=":material/web:",
    default=True,
)
project_3_page = st.Page(
    "dashboard.py",
    title="Historical dashboard",
    icon=":material/history:",
)
project_4_page = st.Page(
    "temp.py",
    title="Temporal Scenario Deck",
    icon=":material/trending_up:",
)
project_5_page = st.Page(
    "hex_history.py",
    title="Hex plot",
    icon=":material/thumb_up:",
)
project_6_page = st.Page(
    "hex_trip.py",
    title="Hex plot trip",
    icon=":material/hexagon:",
)
project_7_page = st.Page(
    "hex_demand.py",
    title="Hex plot demand",
    icon=":material/monitoring:",
)
project_8_page = st.Page(
    "trip_map.py",
    title="Trip Arc map",
    icon=":material/explore:",
)




# --- NAVIGATION SETUP [WITHOUT SECTIONS] ---
# pg = st.navigation(pages=[about_page, project_1_page, project_2_page])

# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "Info": [about_page],
        "Projects": [project_3_page, project_4_page, project_5_page, project_6_page, project_7_page, project_8_page],# project_6_page, project_7_page],
    }
)


# --- SHARED ON ALL PAGES ---
#st.logo("assets/codingisfun_logo.png")
#st.sidebar.markdown("Made with ❤️ by [Sven](https://youtube.com/@codingisfun)")


# --- RUN NAVIGATION ---
pg.run()