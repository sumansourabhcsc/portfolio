import streamlit as st

def global_auth():
    if "auth" not in st.session_state:
        st.session_state.auth = False

    if st.session_state.auth:
        return True

    st.title("🔐 Secure Access")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if password == st.secrets["LOGIN_PASSWORD"]:
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Incorrect password")

    st.stop()
