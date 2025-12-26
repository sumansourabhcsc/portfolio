# pages/1_Overview.py

import os
import io
import csv
import math
import base64
from datetime import datetime
from pathlib import Path

import requests
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# =============================================================================
# CONFIG & AUTH
# =============================================================================


def global_auth():
    if "auth" not in st.session_state:
        st.session_state.auth = False

    if st.session_state.auth:
        return True

    st.title("🔐 Secure Access")
    st.write("Please enter the admin password to access the dashboard.")

    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if password == st.secrets["LOGIN_PASSWORD"]:
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Incorrect password")

    st.stop()


st.set_page_config(
    page_title="Mutual Fund Dashboard",
    page_icon="favicon.png",
    layout="wide"
)

global_auth()

# =============================================================================
# CONSTANTS & LOOKUPS
# =============================================================================

BASE_FOLDER = "mutualfund"

fund_logos = {
    "125354": "mutualfund/logo/axis.png",
    "147946": "mutualfund/logo/bandhan.png",
    "150902": "mutualfund/logo/edelweiss.jpg",
    "151034": "mutualfund/logo/hsbc.png",
    "120594": "mutualfund/logo/ICICI.jpg",
    "119775": "mutualfund/logo/Kotak.png",
    "148928": "mutualfund/logo/mirae.png",
    "127042": "mutualfund/logo/motilal.png",
    "118632": "mutualfund/logo/nippon.png",
    "122639": "mutualfund/logo/parag.jpg",
    "120828": "mutualfund/logo/quant.jpeg",
    "125497": "mutualfund/logo/sbi.png",
    "118834": "mutualfund/logo/mirae.png",
    "143903": "mutualfund/logo/ICICI.jpg",
    "120841": "mutualfund/logo/quant.jpeg",
    "148490": "mutualfund/logo/sbi.png",
    "120834": "mutualfund/logo/quant.jpeg",
    "112090": "mutualfund/logo/Kotak.png",
}

mutual_funds = {
    "Bandhan Small Cap Fund": "147946",
    "Axis Small Cap Fund": "125354",
    "SBI Small Cap Fund": "125497",
    "quant Small Cap Fund": "120828",
    "Motilal Oswal Midcap Fund": "127042",
    "HSBC Midcap Fund": "151034",
    "Kotak Midcap Fund": "119775",
    "quant Mid Cap Fund": "120841",
    "Edelweiss Nifty Midcap150 Momentum 50 Index Fund": "150902",
    "Parag Parikh Flexi Cap Fund": "122639",
    "Kotak Flexicap Fund": "112090",
    "Nippon India Large Cap Fund": "118632",
    "Mirae Asset Large & Midcap Fund": "118834",
    "ICICI Pru BHARAT 22 FOF": "143903",
    "quant Focused Fund": "120834",
    "Mirae Asset FANG+": "148928",
    "ICICI Pru Technology Fund": "120594",
    "SBI Magnum Children's Benefit Fund": "148490",
}

SCHEME_URLS = {
    "125354": "https://www.axismf.com/",
    "147946": "https://bandhanmutual.com/",
    "150902": "https://www.edelweissmf.com/",
    "151034": "https://invest.assetmanagement.hsbc.co.in/auth/login",
    "120594": "https://www.icicipruamc.com/home",
    "119775": "https://www.kotakmf.com/",
    "148928": "https://www.miraeassetmf.co.in/",
    "127042": "https://www.motilaloswalmf.com/",
    "118632": "https://mf.nipponindiaim.com/",
    "122639": "https://amc.ppfas.com/index.php",
    "120828": "https://invest.quantmutualfund.com/",
    "125497": "https://www.sbimf.com/",
    "118834": "https://www.miraeassetmf.co.in/",
    "143903": "https://www.icicipruamc.com/home",
    "120841": "https://invest.quantmutualfund.com/",
    "148490": "https://www.sbimf.com/",
    "120834": "https://invest.quantmutualfund.com/",
    "112090": "https://www.kotakmf.com/",
}

# =============================================================================
# GENERIC HELPERS
# =============================================================================


def get_base64(bin_file: str) -> str:
    with open(bin_file, "rb") as f:
        return base64.b64encode(f.read()).decode()


def format_indian(n):
    n = float(n)
    if abs(n) >= 1_00_00_000:
        return f"{n/1_00_00_000:.2f} Cr"
    elif abs(n) >= 1_00_000:
        return f"{n/1_00_000:.2f} L"
    elif abs(n) >= 1_000:
        return f"{n/1_000:.2f} K"
    else:
        return f"{n:.0f}"


def detect_delimiter(sample_bytes: bytes) -> str:
    sample = sample_bytes.decode("utf-8", errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,")
        return dialect.delimiter
    except Exception:
        return ";" if ";" in sample and sample.count(";") >= sample.count(",") else ","


def load_and_clean_csv_bytes(raw_bytes: bytes) -> pd.DataFrame:
    delim = detect_delimiter(raw_bytes)
    df = pd.read_csv(io.BytesIO(raw_bytes), sep=delim, engine="python", dtype=str)

    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.dropna(how="all", inplace=True)

    col_map = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc == "date":
            col_map[c] = "Date"
        elif lc == "units":
            col_map[c] = "Units"
        elif lc == "nav":
            col_map[c] = "NAV"
        elif lc == "amount":
            col_map[c] = "Amount"
        else:
            col_map[c] = c.strip()
    df = df.rename(columns=col_map)

    if "Date" not in df.columns:
        raise ValueError("CSV does not contain a 'Date' column. Please verify file.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df[df["Date"].notna()].copy()

    if "Units" in df.columns:
        df["Units"] = (
            df["Units"]
            .astype(str)
            .str.replace(",", "")
            .str.strip()
        )
        df["Units"] = pd.to_numeric(df["Units"], errors="coerce")

    if "NAV" in df.columns:
        df["NAV"] = (
            df["NAV"]
            .astype(str)
            .str.replace("₹", "")
            .str.replace(",", "")
            .str.strip()
        )
        df["NAV"] = pd.to_numeric(df["NAV"], errors="coerce")

    if "Amount" in df.columns:
        df["Amount"] = (
            df["Amount"]
            .astype(str)
            .str.replace("₹", "")
            .str.replace(",", "")
            .str.strip()
        )
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    numeric_cols = [c for c in ["Units", "NAV", "Amount"] if c in df.columns]
    if numeric_cols:
        df = df.dropna(subset=numeric_cols, how="all")

    df = df.sort_values("Date", ascending=False).reset_index(drop=True)
    return df


def style_dataframe(df):
    return df.style.set_properties(
        **{"background-color": "#0a1a2f", "color": "#f5f5f5"}
    ).apply(
        lambda row: [
            "background-color: #162437" if i % 2 == 0 else "" for i in range(len(row))
        ],
        axis=1,
    )


# =============================================================================
# XIRR & PORTFOLIO CALCULATIONS
# =============================================================================


def xirr(cashflows, dates, guess=0.1):
    if len(cashflows) != len(dates) or len(cashflows) == 0:
        raise ValueError("cashflows and dates must be same length > 0")

    days = [(d - dates[0]).days for d in dates]

    def npv(rate):
        return sum(cf / ((1 + rate) ** (d / 365.0)) for cf, d in zip(cashflows, days))

    def npv_derivative(rate):
        return sum(
            -(d / 365.0) * cf / ((1 + rate) ** (1 + d / 365.0))
            for cf, d in zip(cashflows, days)
        )

    rate = guess
    for _ in range(100):
        f = npv(rate)
        df = npv_derivative(rate)
        if df == 0:
            break
        new_rate = rate - f / df
        if abs(new_rate - rate) < 1e-6:
            rate = new_rate
            break
        rate = new_rate
    return rate


def compute_portfolio_xirr(all_funds, total_current_all, latest_nav_dates):
    cashflows = []
    dates = []

    for fund_name, dff in all_funds:
        for _, row in dff.iterrows():
            if "Amount" in dff.columns and not pd.isna(row["Amount"]):
                cashflows.append(-float(row["Amount"]))
                dates.append(pd.to_datetime(row["Date"]))

    final_date = max(latest_nav_dates)

    cashflows.append(float(total_current_all))
    dates.append(pd.to_datetime(final_date))

    combined = sorted(zip(dates, cashflows), key=lambda x: x[0])
    dates, cashflows = zip(*combined)

    try:
        return xirr(list(cashflows), list(dates))
    except Exception:
        return None


def fetch_latest_nav_date(fund_code):
    try:
        url = f"https://api.mfapi.in/mf/{fund_code}/latest"
        r = requests.get(url, timeout=10).json()
        if "data" in r and len(r["data"]) > 0:
            mydate = r["data"][0]["date"]
            return mydate
    except Exception:
        pass
    return None


# =============================================================================
# GITHUB DAILY SUMMARY
# =============================================================================


def update_github_daily_summary(total_invested, total_current, total_gain, latest_date):
    token = st.secrets["GITHUB_TOKEN"]
    username = st.secrets["GITHUB_USERNAME"]
    repo = st.secrets["GITHUB_REPO"]
    file_path = st.secrets["GITHUB_FILE_PATH"]

    api_url = (
        f"https://api.github.com/repos/sumansourabhcse/portfolio/contents/"
        f"mutualfund/daily_summary.csv"
    )

    today = latest_date.strftime("%Y-%m-%d")

    total_invested = round(total_invested, 2)
    total_current = round(total_current, 2)
    total_gain = round(total_gain, 2)

    response = requests.get(api_url, headers={"Authorization": f"token {token}"})

    if response.status_code == 200:
        content = response.json()
        sha = content["sha"]
        file_data = base64.b64decode(content["content"]).decode("utf-8")

        df = pd.read_csv(io.StringIO(file_data))

        if today in df["date"].values:
            df.loc[df["date"] == today, ["total_invested", "total_current", "total_gain"]] = [
                total_invested,
                total_current,
                total_gain,
            ]
        else:
            df.loc[len(df)] = [today, total_invested, total_current, total_gain]

        updated_data = df.to_csv(index=False)
    else:
        sha = None
        updated_data = (
            "date,total_invested,total_current,total_gain\n"
            f"{today},{total_invested},{total_current},{total_gain}\n"
        )

    encoded_data = base64.b64encode(updated_data.encode("utf-8")).decode("utf-8")

    payload = {"message": f"Daily update {today}", "content": encoded_data, "sha": sha}

    put_response = requests.put(
        api_url, json=payload, headers={"Authorization": f"token {token}"}
    )

    if put_response.status_code not in [200, 201]:
        st.error(f"❌ Failed to update GitHub: {put_response.text}")


def load_history_from_github():
    token = st.secrets["GITHUB_TOKEN"]
    username = st.secrets["GITHUB_USERNAME"]
    repo = st.secrets["GITHUB_REPO"]
    file_path = st.secrets["GITHUB_FILE_PATH"]

    api_url = f"https://api.github.com/repos/{username}/{repo}/contents/{file_path}"

    response = requests.get(api_url, headers={"Authorization": f"token {token}"})

    if response.status_code != 200:
        st.error("Could not load history file from GitHub")
        return None

    content = response.json()
    file_data = base64.b64decode(content["content"]).decode("utf-8")

    df = pd.read_csv(io.StringIO(file_data))
    df["date"] = pd.to_datetime(df["date"])
    return df


# =============================================================================
# BRANDING & FOOTER
# =============================================================================


def render_header():
    img_path = "mutualfund/logo/ss.png"
    gif_path = "mutualfund/logo/16059837.gif"  # currently unused, but available

    img_base64 = get_base64(img_path)

    st.markdown(
        f"""
        <div style="
            display: flex; 
            align-items: center; 
            background-color: transparent; 
            padding: 10px; 
            border-radius: 10px;
            margin-bottom: 20px;
        ">
            <img src="data:image/png;base64,{img_base64}" 
                 style="
                    width: 90px; 
                    height: 90px; 
                    object-fit: contain; 
                    margin-right: 20px;
                    flex-shrink: 0;
                 ">
            <div style="display: flex; flex-direction: column; justify-content: center;">
                <h1 style="
                    margin: 0; 
                    padding: 0; 
                    font-size: 40px; 
                    line-height: 1; 
                    color: #FACC15;
                ">Mutual Fund Dashboard</h1>
                <p style="
                    margin: 5px 0 0 0; 
                    padding: 0; 
                    color: white; 
                    font-size: 15px;
                ">Track • Analyze • Grow your mutual fund portfolio</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()


def render_footer():
    logo_path = Path("mutualfund/logo/ss.png")
    logo_base64 = base64.b64encode(logo_path.read_bytes()).decode()

    footer_html = f"""
    <style>
    .footer {{
        background-color: #232f3e;
        color: white;
        padding: 30px 50px;
        font-size: 14px;
    }}
    .footer a {{
        color: #ddd;
        text-decoration: none;
    }}
    .footer a:hover {{
        text-decoration: underline;
    }}
    .footer-columns {{
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
    }}
    .footer-column {{
        flex: 1;
        margin: 10px;
        min-width: 200px;
    }}
    .footer-bottom {{
        margin-top: 30px;
        border-top: 1px solid #444;
        padding-top: 20px;
        text-align: center;
        font-size: 12px;
    }}
    .footer-logo {{
        text-align: center;
        margin-top: 15px;
        margin-bottom: 10px;
    }}
    .footer-logo img {{
        width: 120px;
        height: auto;
        object-fit: contain;
        display: inline-block;
    }}
    </style>

    <div class="footer">
      <div class="footer-columns">
        <div class="footer-column">
          <h4>Get to Know Us</h4>
          <a href="#">About</a><br>
          <a href="#">Careers</a><br>
          <a href="#">Press</a><br>
          <a href="#">Science</a>
        </div>
        <div class="footer-column">
          <h4>Connect with Us</h4>
          <a href="#">Facebook</a><br>
          <a href="#">Twitter</a><br>
          <a href="#">Instagram</a>
        </div>
        <div class="footer-column">
          <h4>Make Money with Us</h4>
          <a href="#">Sell</a><br>
          <a href="#">Affiliate</a><br>
          <a href="#">Advertise</a><br>
          <a href="#">Global Selling</a>
        </div>
        <div class="footer-column">
          <h4>Let Us Help You</h4>
          <a href="#">Your Account</a><br>
          <a href="#">Returns</a><br>
          <a href="#">Help</a><br>
          <a href="#">App Download</a>
        </div>
      </div>

      <div class="footer-bottom">
        <p>© 2025 Suman Sourabh PMS Pvt. Ltd. | All rights reserved | 
           <a href="#">Privacy</a> | <a href="#">Terms</a>
        </p>
        <p>Language: English | Country: India</p>
        <div class="footer-logo">
          <img src="data:image/png;base64,{logo_base64}" alt="Suman PMS Logo">
        </div>
      </div>
    </div>
    """

    st.markdown(footer_html, unsafe_allow_html=True)


# =============================================================================
# PORTFOLIO OVERVIEW (ALL FUNDS)
# =============================================================================


def render_portfolio_overview():
    all_funds = []
    per_fund_summary = []
    p_latest_date = datetime.min

    for fund_name in mutual_funds.keys():
        file_path = os.path.join(BASE_FOLDER, fund_name, "fund.csv")
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as f:
                    raw = f.read()
                dff = load_and_clean_csv_bytes(raw)
                all_funds.append((fund_name, dff))
            except Exception as e:
                st.error(f"Failed to parse {fund_name}: {e}")

    if not all_funds:
        st.error("No valid CSVs found in folders.")
        return

    total_invested_all = 0.0
    total_current_all = 0.0
    latest_nav_dates_all = []

    for fund_name, dff in all_funds:
        invested = dff["Amount"].sum() if "Amount" in dff.columns else 0.0
        latest_nav = None
        latest_nav_date = None

        matched_code = mutual_funds.get(fund_name)

        if matched_code:
            try:
                api_url = (
                    f"https://api.mfapi.in/mf/{matched_code}"
                    f"?startDate=2020-01-01&endDate={datetime.today().strftime('%Y-%m-%d')}"
                )
                r = requests.get(api_url, timeout=10)
                jr = r.json()

                d = fetch_latest_nav_date(matched_code)
                if d:
                    current_dt = datetime.strptime(d, "%d-%m-%Y")
                    if current_dt > p_latest_date:
                        p_latest_date = current_dt

                if "data" in jr and jr["data"]:
                    navs = pd.DataFrame(jr["data"])
                    navs["date"] = pd.to_datetime(navs["date"], format="%d-%m-%Y")
                    navs["nav"] = pd.to_numeric(navs["nav"], errors="coerce")
                    navs = navs.sort_values("date")
                    latest_nav = float(navs.iloc[-1]["nav"])
                    latest_nav_date = navs.iloc[-1]["date"]
            except Exception:
                pass

        if latest_nav_date is not None:
            latest_nav_dates_all.append(latest_nav_date)

        if latest_nav is None and "NAV" in dff.columns and dff["NAV"].notna().any():
            latest_nav = float(dff["NAV"].dropna().iloc[0])
            latest_nav_date = pd.to_datetime(dff["Date"].max())
        if latest_nav is None:
            latest_nav = 0.0
            latest_nav_date = None

        units_sum = dff["Units"].sum() if "Units" in dff.columns else 0.0
        current_value = units_sum * latest_nav
        total_invested_all += invested
        total_current_all += current_value

        cashflows, dates = [], []
        for _, row in dff.iterrows():
            if "Amount" in dff.columns and not pd.isna(row["Amount"]):
                cashflows.append(-float(row["Amount"]))
                dates.append(pd.to_datetime(row["Date"]))
        cashflows.append(float(current_value))
        dates.append(pd.to_datetime(latest_nav_date) if latest_nav_date is not None else pd.to_datetime(dff["Date"].max()))

        try:
            irr = xirr(cashflows, dates)
            irr_pct = irr * 100
        except Exception:
            irr_pct = None

        abs_return = (
            ((current_value - invested) / invested) * 100 if invested else None
        )

        gain_loss = current_value - invested

        per_fund_summary.append(
            {
                "Fund": fund_name,
                "Units": units_sum,
                "Invested": invested,
                "Current Value": current_value,
                "Gain/Loss": gain_loss,
                "Absolute Return": (
                    f"{abs_return:.2f}%" if abs_return is not None else "N/A"
                ),
                "Latest NAV": latest_nav,
                "Latest NAV Date": (
                    latest_nav_date.strftime("%Y-%m-%d")
                    if latest_nav_date is not None
                    else "N/A"
                ),
                "XIRR (%)": (
                    f"{irr_pct:.2f}%"
                    if isinstance(irr_pct, (int, float)) and not math.isnan(irr_pct)
                    else "N/A"
                ),
            }
        )

    st.subheader("📊 Portfolio Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("💰 Total Invested", f"₹ {total_invested_all:,.2f}")
    col2.metric("📈 Total Current Value", f"₹ {total_current_all:,.2f}")
    col3.metric(
        "📊 Total Gain/Loss", f"₹ {total_current_all - total_invested_all:,.2f}"
    )
    col4.metric(
        "📉 Absolute Return",
        f"{((total_current_all - total_invested_all)/total_invested_all)*100:.2f}%",
    )

    with col5:
        overall_irr = compute_portfolio_xirr(
            all_funds=all_funds,
            total_current_all=total_current_all,
            latest_nav_dates=latest_nav_dates_all,
        )
        if overall_irr is not None:
            st.metric("📅 Portfolio XIRR (annual)", f"{overall_irr*100:.2f}%")
        else:
            st.metric("📅 Portfolio XIRR (annual)", "N/A")

    col_left, col_right = st.columns([3, 1])

    df_summary = (
        pd.DataFrame(per_fund_summary)
        .sort_values("Current Value", ascending=False)
        .reset_index(drop=True)
    )

    with col_left:
        st.subheader("Individual Funds")
        st.dataframe(df_summary, use_container_width=True)

    with col_right:
        figp = px.pie(
            df_summary,
            names="Fund",
            values="Current Value",
            title="Portfolio Allocation",
            hole=0.5,
        )
        figp.update_layout(
            showlegend=False,
            margin=dict(t=40, b=40),
            height=500,
        )
        st.plotly_chart(figp, use_container_width=True)

    total_gain_all = total_current_all - total_invested_all
    update_github_daily_summary(
        total_invested_all, total_current_all, total_gain_all, p_latest_date
    )

    df_hist = load_history_from_github()
    if df_hist is not None:
        render_daily_history_section(df_hist)


def render_daily_history_section(df_hist: pd.DataFrame):
    df_hist = df_hist.copy()
    df_hist["date"] = pd.to_datetime(df_hist["date"])

    df_display = df_hist.copy()
    df_display["date"] = df_display["date"].dt.date
    df_display = df_display.rename(
        columns={
            "date": "Date",
            "total_invested": "Invested",
            "total_current": "Total Value",
            "total_gain": "Current Return",
        }
    )
    for col in ["Invested", "Total Value", "Current Return"]:
        df_display[col] = df_display[col].map(lambda x: f"{x:,.2f}")

    styled_df = df_display.style.set_table_styles(
        [
            {
                "selector": "th",
                "props": [("font-weight", "bold"), ("font-size", "14px")],
            }
        ]
    )

    st.markdown("### 📄 Daily Summary Dataset")
    st.dataframe(styled_df, use_container_width=True)

    df_hist_sorted = df_hist.sort_values("date").reset_index(drop=True)
    df_hist_sorted["date_str"] = df_hist_sorted["date"].dt.strftime("%Y-%m-%d")

    max_gain_idx = df_hist_sorted["total_gain"].idxmax()
    min_gain_idx = df_hist_sorted["total_gain"].idxmin()
    max_point = df_hist_sorted.loc[max_gain_idx]
    min_point = df_hist_sorted.loc[min_gain_idx]

    st.markdown("### 📊 Total Invested + Gain (Stacked) vs Current Value (Line)")
    fig_combo = go.Figure()

    fig_combo.add_trace(
        go.Bar(
            x=df_hist_sorted["date_str"],
            y=df_hist_sorted["total_invested"],
            name="Total Invested",
            marker_color="rgba(31,119,180,0.85)",
            text=[f"₹ {v/100000:.1f}L" for v in df_hist_sorted["total_invested"]],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=14, color="white"),
            hovertemplate="Date: %{x}<br>Invested: ₹ %{y:,.0f}<extra></extra>",
        )
    )

    fig_combo.add_trace(
        go.Bar(
            x=df_hist_sorted["date_str"],
            y=df_hist_sorted["total_gain"],
            name="Total Gain",
            marker_color="rgba(44,160,44,0.85)",
            text=[f"₹ {v/100000:.1f}L" for v in df_hist_sorted["total_gain"]],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=14, color="white"),
            hovertemplate="Date: %{x}<br>Gain: ₹ %{y:,.0f}<extra></extra>",
        )
    )

    fig_combo.add_trace(
        go.Scatter(
            x=df_hist_sorted["date_str"],
            y=df_hist_sorted["total_current"],
            mode="lines+markers",
            name="Total Current Value",
            line=dict(color="orange", width=3),
            marker=dict(size=6),
            hovertemplate="Date: %{x}<br>Total Value: ₹ %{y:,.0f}<extra></extra>",
        )
    )

    fig_combo.add_trace(
        go.Scatter(
            x=[max_point["date_str"]],
            y=[max_point["total_gain"] + max_point["total_invested"]],
            mode="markers+text",
            name="Highest Profit",
            marker=dict(color="gold", size=20, symbol="star"),
            text=[f"High: ₹ {max_point['total_gain']:,.0f}"],
            textposition="top center",
            textfont=dict(color="white", size=14),
            hovertemplate="📈 Highest Profit<br>Date: %{x}<br>Gain: ₹ %{y:,.0f}<extra></extra>",
        )
    )

    fig_combo.add_trace(
        go.Scatter(
            x=[min_point["date_str"]],
            y=[min_point["total_gain"] + min_point["total_invested"]],
            mode="markers+text",
            name="Lowest Profit",
            marker=dict(color="red", size=20, symbol="triangle-down"),
            text=[f"Low: ₹ {min_point['total_gain']:,.0f}"],
            textposition="bottom center",
            textfont=dict(color="white", size=14),
            hovertemplate="📉 Lowest Profit<br>Date: %{x}<br>Gain: ₹ %{y:,.0f}<extra></extra>",
        )
    )

    fig_combo.update_xaxes(type="category", tickangle=-45, title="Date")
    fig_combo.update_layout(
        barmode="stack",
        height=500,
        margin=dict(t=40, b=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
    )

    st.plotly_chart(fig_combo, use_container_width=True)

    st.markdown("### 💹 Portfolio Value vs Daily Gain/Loss")

    fig1 = make_subplots(specs=[[{"secondary_y": True}]])

    fig1.add_trace(
        go.Scatter(
            x=df_hist["date"],
            y=df_hist["total_current"],
            mode="lines+markers",
            name="Total Current Value",
            line=dict(color="#4C9AFF", width=3),
        ),
        secondary_y=False,
    )

    daily_gain = df_hist["total_gain"].diff().fillna(0)
    fig1.add_trace(
        go.Bar(
            x=df_hist["date"],
            y=daily_gain,
            name="Daily Gain/Loss",
            marker_color=daily_gain.apply(
                lambda x: "#6CCF7F" if x >= 0 else "#E57373"
            ),
        ),
        secondary_y=True,
    )

    fig1.update_layout(
        template="plotly_dark",
        height=550,
        bargap=0.25,
        legend_title_text="Metrics",
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.write("#### Daily Gain/Loss")

    df_gain = df_hist.copy()
    df_gain["gain_change"] = df_gain["total_gain"].diff().fillna(0)
    df_gain["date"] = pd.to_datetime(df_gain["date"])

    fig = px.bar(
        df_gain,
        x="date",
        y="gain_change",
        title="",
        color="gain_change",
        color_continuous_scale="RdYlGn",
    )

    fig.update_xaxes(tickformat="%d/%m", tickangle=45)
    fig.update_layout(template="plotly_dark", coloraxis_showscale=False)

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# SINGLE FUND VIEW
# =============================================================================


def render_single_fund_view(selected_fund: str):
    fund_code = mutual_funds[selected_fund]
    file_path = os.path.join(BASE_FOLDER, selected_fund, "fund.csv")

    if not os.path.exists(file_path):
        st.error(f"No CSV file found for {selected_fund} in {BASE_FOLDER}")
        return

    try:
        with open(file_path, "rb") as f:
            raw = f.read()
        df_invest = load_and_clean_csv_bytes(raw)
    except Exception as e:
        st.error(f"Failed to read CSV for {selected_fund}: {e}")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        logo_url = fund_logos.get(fund_code)
        if logo_url:
            st.image(logo_url, width=180)

    with col2:
        try:
            meta_url = f"https://api.mfapi.in/mf/{fund_code}"
            meta_resp = requests.get(meta_url, timeout=10)
            scheme_url = SCHEME_URLS.get(fund_code, "#")
            if meta_resp.status_code == 200:
                meta_json = meta_resp.json()
                if "meta" in meta_json:
                    meta = meta_json["meta"]
                    scheme_name = meta.get("scheme_name", "N/A")
                    st.write("### 🏦 Fund Details")
                    st.markdown(
                        f"""
                        - **Scheme Name:** [{scheme_name}]({scheme_url})
                        - **AMC:** {meta.get('fund_house','N/A')}
                        - **Category:** {meta.get('scheme_category','N/A')}
                        - **Scheme Type:** {meta.get('scheme_type','N/A')}
                        - **Fund Code:** {fund_code}
                        - **ISIN:** {meta.get('isin_growth','N/A')}
                        """
                    )
        except Exception:
            st.warning("Could not fetch fund details from API.")

    with col3:
        st.write("### 📶 Status")
        total_units = df_invest["Units"].sum()
        weighted_nav = (df_invest["Units"] * df_invest["NAV"]).sum() / total_units
        st.markdown(
            f"""
            - **Average Buy NAV:** {weighted_nav:.2f}
            - **Total Units:** {total_units:.2f}
            """
        )

    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.subheader(selected_fund)
        st.dataframe(df_invest, use_container_width=True, height=200)

        if st.button("Add New Units"):
            st.session_state.show_add_units = True

        if st.session_state.get("show_add_units", False):
            try:
                from utils.auth import authenticate  # optional external auth
                from utils.data_writer import append_transaction
            except ImportError:
                authenticate = None
                append_transaction = None

            st.write("#### Admin Authentication Required")

            if authenticate and authenticate():
                st.success("Authenticated")
                date = st.date_input("Purchase Date")
                units = st.number_input("Units Bought", min_value=0.0, step=0.01)
                nav = st.number_input("NAV", min_value=0.0, step=0.01)
                amount = st.number_input("Amount")

                if st.button("Submit Transaction"):
                    if append_transaction:
                        date_str = date.strftime("%d-%m-%Y")
                        append_transaction(
                            fund_name=selected_fund,
                            date=date_str,
                            units=units,
                            nav=nav,
                            amount=amount,
                        )
                        st.success("Transaction added successfully!")
                        st.session_state.show_add_units = False
                        st.rerun()
                    else:
                        st.error("append_transaction not available.")
            else:
                st.info("Please authenticate to continue.")

    with col_right:
        st.write("##### 📅 Select NAV Date Range")
        default_start = df_invest["Date"].min().date()
        default_end = datetime.today().date()
        selected_dates = st.date_input(
            "Select NAV Date Range",
            value=(default_start, default_end),
            min_value=default_start,
            max_value=default_end,
            key=f"{selected_fund}_nav",
        )

        if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
            start_date, end_date = selected_dates
        else:
            st.error("Please select a valid start and end date.")
            return

        fetch_nav_button = st.button("Fetch NAV")

    if not fetch_nav_button:
        return

    api_url = (
        f"https://api.mfapi.in/mf/{fund_code}?startDate={start_date}&endDate={end_date}"
    )
    resp = requests.get(api_url)
    if resp.status_code != 200:
        st.error("API fetch failed.")
        return

    j = resp.json()
    if "data" not in j or not j["data"]:
        st.error("No NAV data for selected range.")
        return

    df_nav = pd.DataFrame(j["data"])
    df_nav["date"] = pd.to_datetime(df_nav["date"], format="%d-%m-%Y")
    df_nav["nav"] = pd.to_numeric(df_nav["nav"], errors="coerce")
    df_nav = df_nav.sort_values("date")

    highest_nav = df_nav.loc[df_nav["nav"].idxmax()]
    lowest_nav = df_nav.loc[df_nav["nav"].idxmin()]

    high_nav_value = highest_nav["nav"]
    high_nav_date = highest_nav["date"].strftime("%Y-%m-%d")
    low_nav_value = lowest_nav["nav"]
    low_nav_date = lowest_nav["date"].strftime("%Y-%m-%d")

    st.subheader("📌 NAV Range in Selected Period")
    colA, colB = st.columns(2)
    with colA:
        st.markdown(
            f"""
            <div style="padding:10px; background:#1f2a40; border-radius:8px;">
                <h4 style="color:#4CAF50;">🔼 Highest NAV</h4>
                <p style="font-size:20px;">₹ {high_nav_value:.4f}</p>
                <p>Date: {high_nav_date}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with colB:
        st.markdown(
            f"""
            <div style="padding:10px; background:#1f2a40; border-radius:8px;">
                <h4 style="color:#FF5252;">🔽 Lowest NAV</h4>
                <p style="font-size:20px;">₹ {low_nav_value:.4f}</p>
                <p>Date: {low_nav_date}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    latest_nav = df_nav.iloc[-1]["nav"]

    if "Units" not in df_invest.columns or "Amount" not in df_invest.columns:
        st.warning(
            "Uploaded CSV doesn't have Units or Amount columns. "
            "Skipping value/gain calculations."
        )
        return

    df_invest_current = df_invest.copy()
    df_invest_current["Current Value"] = df_invest_current["Units"] * latest_nav
    df_invest_current["Gain/Loss"] = (
        df_invest_current["Current Value"] - df_invest_current["Amount"]
    )

    df_invest_current = df_invest_current.sort_values("Date", ascending=True).reset_index(
        drop=True
    )
    df_invest_current["Cumulative Gain"] = df_invest_current["Gain/Loss"].cumsum()
    df_invest_current["Cumulative Units"] = df_invest_current["Units"].cumsum()

    total_invested = df_invest_current["Amount"].sum()
    total_current = df_invest_current["Current Value"].sum()
    total_units = df_invest_current["Units"].sum()
    total_gain = total_current - total_invested
    total_cumulative = df_invest_current["Cumulative Gain"].iloc[-1]
    total_return_pct = (
        (total_gain / total_invested * 100) if total_invested != 0 else 0.0
    )

    cashflows = []
    dates = []
    for _, row in df_invest_current.iterrows():
        if not pd.isna(row["Amount"]):
            cashflows.append(-float(row["Amount"]))
            dates.append(pd.to_datetime(row["Date"]))
    cashflows.append(float(total_current))
    dates.append(pd.to_datetime(df_nav.iloc[-1]["date"]))
    try:
        irr = xirr(cashflows, dates)
        irr_pct = irr * 100
    except Exception:
        irr_pct = None

    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    col1.markdown(
        f"<h6>Total Invested</h6><p style='font-size:20px;'>₹ {total_invested:,.2f}</p>",
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"<h6>Current Value</h6><p style='font-size:20px;'>₹ {total_current:,.2f}</p>",
        unsafe_allow_html=True,
    )
    col3.markdown(
        f"<h6>Total Units</h6><p style='font-size:20px;'>{total_units:,.2f}</p>",
        unsafe_allow_html=True,
    )
    col4.markdown(
        f"<h6>Absolute Gain/Loss</h6>"
        f"<p style='font-size:20px;'>₹ {total_gain:,.2f}</p>",
        unsafe_allow_html=True,
    )

    return_color = "green" if total_return_pct >= 0 else "red"
    col5.markdown(
        f"<h6>Total Return (%)</h6>"
        f"<p style='font-size:20px; color:{return_color};'>{total_return_pct:.2f}%</p>",
        unsafe_allow_html=True,
    )

    if irr_pct is not None and pd.notna(irr_pct):
        xirr_str = f"{irr_pct:.2f}%"
        color = "green" if irr_pct >= 0 else "red"
    else:
        xirr_str = "N/A"
        color = "#666"

    col6.markdown(
        f"""
        <div>
        <div style="font-size:16px; font-weight:600; color:#555;">
        <b>XIRR (annual)</b></div>
        <div style="font-size:20px; color:{color};">{xirr_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    latest_nav_api = None
    latest_nav_date = None
    try:
        latest_api_url = f"https://api.mfapi.in/mf/{fund_code}/latest"
        latest_api_resp = requests.get(latest_api_url, timeout=10)
        if latest_api_resp.status_code == 200:
            latest_api_json = latest_api_resp.json()
            if "data" in latest_api_json and len(latest_api_json["data"]) > 0:
                item = latest_api_json["data"][0]
                latest_nav_api = float(item["nav"])
                latest_nav_date = item.get("date", None)
    except Exception:
        pass

    nav_str = f"₹ {latest_nav_api:,.4f}" if latest_nav_api else "N/A"
    col7.markdown(
        f"""
        <div>
        <div style="font-size:16px; font-weight:600; color:#555;">
        <b>Latest NAV (API)</b></div>
        <div style="font-size:20px; color:#333;">{nav_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_date_str = latest_nav_date if latest_nav_date else "N/A"
    col8.markdown(
        f"""
        <div>
        <div style="font-size:16px; font-weight:600; color:#555;"><b>NAV Date</b></div>
        <div style="font-size:20px; color:#333;">{nav_date_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left2, col_right2 = st.columns([3, 1])

    with col_left2:
        st.subheader(f"📈 NAV Chart: {selected_fund}")
        fig = px.line(
            df_nav,
            x="date",
            y="nav",
            labels={"date": "Date", "nav": "NAV"},
            template="plotly_white",
        )

        fig.update_layout(
            plot_bgcolor="#0a1a2f",
            paper_bgcolor="#0a1a2f",
            font=dict(color="#f5f5f5"),
            hovermode="x unified",
            legend=dict(
                bgcolor="rgba(255,255,255,0.15)",
                bordercolor="rgba(255,255,255,0.4)",
                borderwidth=1,
                font=dict(color="#ffffff", size=12),
            ),
        )
        fig.update_traces(
            mode="lines+markers", hovertemplate="Date: %{x}<br>NAV: %{y}"
        )

        highest_nav = df_nav.loc[df_nav["nav"].idxmax()]
        lowest_nav = df_nav.loc[df_nav["nav"].idxmin()]

        high_nav_value = highest_nav["nav"]
        high_nav_date = highest_nav["date"]
        low_nav_value = lowest_nav["nav"]
        low_nav_date = lowest_nav["date"]

        fig.add_annotation(
            x=high_nav_date,
            y=high_nav_value,
            text=f"High: ₹{high_nav_value:.2f}<br>{high_nav_date.strftime('%Y-%m-%d')}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowcolor="red",
            font=dict(color="white", size=12),
            bgcolor="rgba(0,0,0,0.6)",
            bordercolor="red",
            borderwidth=1,
            yshift=20,
        )

        fig.add_annotation(
            x=low_nav_date,
            y=low_nav_value,
            text=f"Low: ₹{low_nav_value:.2f}<br>{low_nav_date.strftime('%Y-%m-%d')}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowcolor="white",
            font=dict(color="white", size=12),
            bgcolor="rgba(0,0,0,0.6)",
            bordercolor="white",
            borderwidth=1,
            yshift=-20,
        )

        df_nav["date_ordinal"] = df_nav["date"].map(datetime.toordinal)
        X = sm.add_constant(df_nav["date_ordinal"])
        y = df_nav["nav"]
        model = sm.OLS(y, X).fit()
        df_nav["trend"] = model.predict(X)

        slope = model.params[1]
        trend_color = "green" if slope > 0 else "red"

        fig.add_scatter(
            x=df_nav["date"],
            y=df_nav["trend"],
            mode="lines",
            name="Trend Line",
            line=dict(color=trend_color, dash="dash"),
        )

        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with col_right2:
        st.subheader("📋 NAV Table")
        df_nav_display = df_nav.copy()
        df_nav_display["date"] = df_nav_display["date"].dt.date
        df_nav_display = df_nav_display.sort_values("date", ascending=False)
        st.dataframe(
            df_nav_display[["date", "nav"]].reset_index(drop=True),
            use_container_width=True,
        )

    totals_row = {
        "Date": "TOTAL",
        "Units": total_units,
        "Amount": total_invested,
        "Current Value": total_current,
        "Gain/Loss": total_gain,
        "Cumulative Gain": total_cumulative,
    }

    df_invest_current = pd.concat(
        [df_invest_current, pd.DataFrame([totals_row])], ignore_index=True
    )
    df_invest_current["Date"] = df_invest_current["Date"].astype(str)
    numeric_cols = df_invest_current.select_dtypes(include=["float", "int"]).columns
    df_invest_current[numeric_cols] = df_invest_current[numeric_cols].round(2)

    def highlight_total(row):
        return [
            "background-color: #f0f0f0; font-weight: bold; color: darkblue;"
            if row["Date"] == "TOTAL"
            else ""
            for _ in row
        ]

    styled_df = (
        df_invest_current.style.apply(highlight_total, axis=1).format(
            {col: "{:.2f}" for col in numeric_cols}
        )
    )

    st.subheader(
        "📋 Investment Details with Current Value, Gain/Loss & Cumulative Gain"
    )
    st.dataframe(styled_df, use_container_width=True)


# =============================================================================
# MAIN
# =============================================================================


def main():
    render_header()

    st.sidebar.header("Your Mutual Funds")
    overview_button = st.sidebar.button("📦 Complete Overview")

    st.sidebar.markdown("---")
    st.sidebar.header("Single Fund View")
    selected_fund = st.sidebar.radio(
        "Select a Mutual Fund (Single view)", list(mutual_funds.keys()), index=0
    )

    if overview_button:
        render_portfolio_overview()
    else:
        render_single_fund_view(selected_fund)

    st.divider()
    render_footer()


if __name__ == "__main__":
    main()
