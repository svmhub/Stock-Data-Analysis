# app.py
# Streamlit dashboard for stock analytics backed by SQL
# Author: Sundaralinga Nadar Vaikundamani

# ---------------------------------------------------------------
# Features (matches your problem statements exactly):
# 1) Key Metrics: Top 10 green/loss stocks, market summary (green vs red count, avg price, avg volume)
# 2) Volatility Analysis: Std dev of daily returns, top 10 most volatile (last 1 year)
# 3) Cumulative Return Over Time: Top 5 performing stocks (lines over the year)
# 4) Sector-wise Performance: Average yearly return by sector (bar chart)
# 5) Stock Price Correlation: Heatmap of correlations (daily returns)
# 6) Top 5 Gainers & Losers (Month-wise): For each month (12), top 5 gainers/losers
# ---------------------------------------------------------------

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Stocks Dashboard — Stack_DB",
    page_icon="📈",
    layout="wide",
)

st.title("Dashborad - Stock Data Analysis")

# ---------------------------------------------------------------
# Utility: Load data from SQLite with caching
# ---------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_data(db_path: str, table: str = "main") -> pd.DataFrame:
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql(f"SELECT * FROM {table}", con)
    # Normalize columns (lowercase for consistency)
    df.columns = [c.lower() for c in df.columns]
    # Ensure required columns exist
    required = {"ticker", "company", "sector", "symbol", "close", "date", "high", "low", "month", "open", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in table: {sorted(missing)}")
    # Types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Drop obvious bad rows
    df = df.dropna(subset=["ticker", "date", "open", "close"]).copy()
    # Sort for time-based ops
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df

# ---------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------

def compute_yearly_return_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with a 'yearly_return' column per row: ((close-open)/open)*100.
    Used by Key Metrics page (your problem statement #1)."""
    out = df.copy()
    out["yearly_return"] = (out["close"] - out["open"]) / out["open"] * 100.0
    return out


def last_year_slice(df: pd.DataFrame) -> pd.DataFrame:
    last_dt = df["date"].max()
    cutoff = last_dt - pd.DateOffset(years=1)
    return df[df["date"] > cutoff].copy()


def per_ticker_yearly_return(df_year: pd.DataFrame) -> pd.DataFrame:
    """Compute one yearly return per ticker within df_year window using first open vs last close."""
    g = (df_year
         .groupby("ticker")
         .agg(first_open=("open", "first"), last_close=("close", "last"), sector=("sector", "first"), company=("company", "first"))
         .reset_index())
    g["yearly_return"] = (g["last_close"] - g["first_open"]) / g["first_open"] * 100.0
    return g


def compute_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["daily_return"] = out.groupby("ticker")["close"].pct_change()
    return out


def compute_volatility_last_year(df: pd.DataFrame, min_days: int = 30) -> pd.DataFrame:
    d1y = last_year_slice(df)
    d1y = compute_daily_returns(d1y)
    vol = (d1y.groupby("ticker")["daily_return"].agg(["std", "count"]).reset_index()
           .rename(columns={"std": "volatility", "count": "n_days"}))
    vol = vol[vol["n_days"] >= min_days]
    vol = vol.dropna(subset=["volatility"])  # guard
    return vol.sort_values("volatility", ascending=False)


def cumulative_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = compute_daily_returns(df)
    out["cumulative_return"] = (1.0 + out["daily_return"]).groupby(out["ticker"]).cumprod() - 1.0
    return out


def sector_summary(df_year: pd.DataFrame, min_tickers: int = 3) -> pd.DataFrame:
    per = per_ticker_yearly_return(df_year)
    summ = (per.groupby("sector")["yearly_return"]
            .agg(avg_yearly_return="mean", median_return="median", std_return="std", n_tickers="count")
            .reset_index())
    return summ[summ["n_tickers"] >= min_tickers].sort_values("avg_yearly_return", ascending=False)


def returns_wide(df: pd.DataFrame, min_overlap_days: int = 30) -> pd.DataFrame:
    ret = compute_daily_returns(df)
    wide = ret.pivot(index="date", columns="ticker", values="daily_return")
    valid = wide.count()
    keep = valid[valid >= min_overlap_days].index
    wide = wide[keep]
    return wide


def monthly_leaders(df_year: pd.DataFrame, top_n: int = 5) -> dict:
    """Compute per-month top gainers/losers tables.
    Returns dict: { 'YYYY-MM': { 'gainers': DataFrame, 'losers': DataFrame } }"""
    out = {}
    df_year["month_period"] = df_year["date"].dt.to_period("M")
    monthly = (df_year.groupby(["ticker", "month_period"])  # first/last close each month
               .agg(first_close=("close", "first"), last_close=("close", "last"))
               .reset_index())
    monthly["monthly_return_pct"] = (monthly["last_close"] / monthly["first_close"] - 1.0) * 100.0
    for m in sorted(monthly["month_period"].unique()):
        sub = monthly[monthly["month_period"] == m].copy()
        gainers = sub.nlargest(top_n, "monthly_return_pct")[['ticker','monthly_return_pct']]
        losers = sub.nsmallest(top_n, "monthly_return_pct")[['ticker','monthly_return_pct']]
        out[str(m)] = {"gainers": gainers, "losers": losers}
    return out

# ---------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------
st.sidebar.header("⚙️ Controls")
DB_PATH = st.sidebar.text_input("SQLite DB path", value="Stack_DB.db")
TABLE = st.sidebar.text_input("Table name", value="main")

# Load data
with st.spinner("Loading data from SQLite…"):
    df = load_data(DB_PATH, TABLE)

# Global filters
years = sorted(df["date"].dt.year.dropna().astype(int).unique().tolist())
sel_year = st.sidebar.selectbox("Year to analyze", options=years, index=len(years)-1)

# Filtered dataframe for selected year
df_year = df[df["date"].dt.year == sel_year].copy()

# Optional sector filter
sectors = ["(All)"] + sorted(df["sector"].dropna().unique().tolist())
sel_sector = st.sidebar.selectbox("Sector filter (optional)", options=sectors)
if sel_sector != "(All)":
    df = df[df["sector"] == sel_sector]
    df_year = df_year[df_year["sector"] == sel_sector]

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Change year/sector to update all analyses live.")

# ---------------------------------------------------------------
# Layout: Tabs for each problem statement
# ---------------------------------------------------------------
TAB1, TAB2, TAB3, TAB4, TAB5, TAB6 = st.tabs([
    "1) Key Metrics",
    "2) Volatility (1Y)",
    "3) Cumulative Return",
    "4) Sector Performance",
    "5) Correlation",
    "6) Monthly Leaders",
])

# ---------------------------------------------------------------
# TAB 1: Key Metrics
# ---------------------------------------------------------------
with TAB1:
    st.subheader("1) Key Metrics — Top 10 Green / Loss & Market Summary")
    df_km = compute_yearly_return_rows(df_year)
    st.dataframe(df_km)
    # Top 10 green & loss
    top_green = df_km.sort_values("yearly_return", ascending=False).head(10)
    top_loss = df_km.sort_values("yearly_return", ascending=True).head(10)

    # Market summary
    green_count = int((df_km["yearly_return"] > 0).sum())
    red_count = int((df_km["yearly_return"] < 0).sum())
    avg_price = float(df_km["close"].mean())
    avg_volume = float(df_km["volume"].mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Green Stocks", green_count)
    c2.metric("Red Stocks", red_count)
    c3.metric("Avg Price", f"{avg_price:,.2f}")
    c4.metric("Avg Volume", f"{avg_volume:,.0f}")

    st.markdown("**Top 10 Green Stocks (Yearly Return %)**")
    st.dataframe(top_green[["ticker","company","sector","open","close","yearly_return","volume"]], use_container_width=True)

    st.markdown("**Top 10 Loss Stocks (Yearly Return %)**")
    st.dataframe(top_loss[["ticker","company","sector","open","close","yearly_return","volume"]], use_container_width=True)

# ---------------------------------------------------------------
# TAB 2: Volatility Analysis (last 1 year window from dataset max date)
# ---------------------------------------------------------------
with TAB2:
    st.subheader("2) Volatility — Std Dev of Daily Returns (Last 1 Year)")
    min_days = st.number_input("Min trading days per ticker (to include)", min_value=1, max_value=252, value=30, step=1)
    vol = compute_volatility_last_year(df if sel_sector == "(All)" else df[df["sector"] == sel_sector], min_days=min_days)

    st.markdown("**Top 10 Most Volatile**")
    top10_vol = vol.head(10)
    st.dataframe(top10_vol, use_container_width=True)

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(top10_vol["ticker"], top10_vol["volatility"])
    ax.set_title("Top 10 Most Volatile (Std Dev of Daily Returns)")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Volatility (std dev)")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig, use_container_width=True)

# ---------------------------------------------------------------
# TAB 3: Cumulative Return Over Time
# ---------------------------------------------------------------
with TAB3:
    st.subheader("3) Cumulative Return — Top 5 Performing Stocks")
    df_cr = cumulative_returns(df_year)

    # Rank by final cumulative return (last value per ticker)
    final_cum = df_cr.groupby("ticker")["cumulative_return"].last().sort_values(ascending=False)
    top5_tickers = final_cum.head(5).index.tolist()

    st.write("**Top 5 by final cumulative return:**", ", ".join(top5_tickers))

    fig, ax = plt.subplots(figsize=(10, 5))
    for t in top5_tickers:
        sub = df_cr[df_cr["ticker"] == t]
        ax.plot(sub["date"], sub["cumulative_return"], label=t)
    ax.set_title(f"Cumulative Return — {sel_year}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, use_container_width=True)

# ---------------------------------------------------------------
# TAB 4: Sector-wise Performance
# ---------------------------------------------------------------
with TAB4:
    st.subheader("4) Sector-wise Performance — Average Yearly Return by Sector")
    min_tickers = st.number_input("Min tickers per sector", min_value=1, max_value=50, value=3)
    summ = sector_summary(df_year, min_tickers=min_tickers)
    st.dataframe(summ, use_container_width=True)

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(summ["sector"], summ["avg_yearly_return"])
    ax.set_title(f"Average Yearly Return by Sector — {sel_year}")
    ax.set_xlabel("Sector")
    ax.set_ylabel("Average Yearly Return (%)")
    ax.tick_params(axis='x', rotation=45)
    # Annotate
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{h:.2f}%", (p.get_x() + p.get_width()/2, h), ha="center", va="bottom", xytext=(0,3), textcoords="offset points", fontsize=8)
    st.pyplot(fig, use_container_width=True)

# ---------------------------------------------------------------
# TAB 5: Correlation Heatmap (daily returns)
# ---------------------------------------------------------------
with TAB5:
    st.subheader("5) Stock Price Correlation — Heatmap (Daily % Returns)")
    min_overlap = st.number_input("Min overlapping days per ticker", min_value=1, max_value=252, value=30)
    wide = returns_wide(df_year, min_overlap_days=int(min_overlap))
    corr = wide.corr()

    st.markdown("**Correlation Matrix (downloadable)**")
    st.dataframe(corr, use_container_width=True)
    st.download_button("Download correlation CSV", corr.to_csv().encode("utf-8"), file_name=f"correlation_{sel_year}.csv")

    # Heatmap using matplotlib
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation")
    ax.set_title(f"Correlation Heatmap — {sel_year}")
    st.pyplot(fig, use_container_width=True)

# ---------------------------------------------------------------
# TAB 6: Monthly Top Gainers & Losers
# ---------------------------------------------------------------
with TAB6:
    st.subheader("6) Top 5 Gainers & Losers — Month-wise")
    top_n = st.number_input("Top N", min_value=1, max_value=10, value=5)
    ml = monthly_leaders(df_year, top_n=top_n)

    months = list(ml.keys())
    # Show all months as expandable sections
    for m in months:
        with st.expander(f"{m} — Top {top_n} Gainers & Losers"):
            g = ml[m]["gainers"].copy()
            l = ml[m]["losers"].copy()

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Gainers ({m})**")
                st.dataframe(g, use_container_width=True)
                # Bar chart
                fig_g, ax_g = plt.subplots(figsize=(6, 4))
                g_sorted = g.sort_values("monthly_return_pct", ascending=False)
                ax_g.bar(g_sorted["ticker"], g_sorted["monthly_return_pct"])
                ax_g.set_title(f"Gainers {m}")
                ax_g.set_xlabel("Ticker")
                ax_g.set_ylabel("Monthly Return (%)")
                ax_g.tick_params(axis='x', rotation=45)
                st.pyplot(fig_g, use_container_width=True)

            with c2:
                st.markdown(f"**Losers ({m})**")
                st.dataframe(l, use_container_width=True)
                fig_l, ax_l = plt.subplots(figsize=(6, 4))
                l_sorted = l.sort_values("monthly_return_pct", ascending=True)
                ax_l.bar(l_sorted["ticker"], l_sorted["monthly_return_pct"])
                ax_l.set_title(f"Losers {m}")
                ax_l.set_xlabel("Ticker")
                ax_l.set_ylabel("Monthly Return (%)")
                ax_l.tick_params(axis='x', rotation=45)
                st.pyplot(fig_l, use_container_width=True)

# ---------------------------------------------------------------

# ============ FOOTER ============
st.caption("Built with ❤️ by VAIKUNDAMANI S — Streamlit Web Application")
