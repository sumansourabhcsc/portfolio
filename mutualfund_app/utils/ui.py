import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


def render_header():
    st.markdown("<h1 style='color:#FACC15;'>Mutual Fund Dashboard</h1>", unsafe_allow_html=True)
    st.write("Track • Analyze • Grow your mutual fund portfolio")
    st.divider()


def render_footer():
    st.divider()
    st.write("© 2025 Suman Sourabh PMS Pvt. Ltd.")


def render_nav_chart(df):
    df = df.sort_values("date")
    fig = px.line(df, x="date", y="total_current", title="Portfolio Value Over Time")
    st.plotly_chart(fig, use_container_width=True)
