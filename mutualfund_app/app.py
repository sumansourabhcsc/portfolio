import streamlit as st
from utils.auth import global_auth

st.set_page_config(page_title="Mutual Fund Dashboard", layout="wide")

# Authentication
global_auth()

st.title("📈 Mutual Fund Dashboard")
st.write("Use the sidebar to navigate between Overview, Monthly Analytics, and Rebalance.")
