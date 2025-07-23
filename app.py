import streamlit as st

home_page = st.Page("1.home.py", title="Home", icon="🏠")
info_page = st.Page("2.info.py", title="Customer Profile", icon="🪪")
dashboard_page = st.Page("3.dashboard.py", title="Dashboard", icon="📊")
playground_page = st.Page("4.playground.py", title="Playground", icon="🚀")

pg = st.navigation([home_page, info_page, dashboard_page, playground_page])
st.set_page_config(page_title="Economist App", page_icon="📕", layout="wide")

pg.run()