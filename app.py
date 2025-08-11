import io, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="NGO Donation Forecast & Analytics", layout="wide")

st.title("Donation Forecast & Analytics (NGO)")
st.caption("Analyze last 36 months → forecast next 36 months. Upload a CSV; all columns are used when possible.")

# ---------- Helpers ----------
DATE_CANDIDATES = ["date","donation_date","transaction_date","Date","DATE"]
AMOUNT_CANDIDATES = ["amount","donation_amount","total_donations","total_donations_rwf","Amount"]

def find_col(candidates, cols):
    for c in candidates:
        if c in cols: return c
    return None

def parse_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    cols = df.columns.tolist()

    date_col = find_col(DATE_CANDIDATES, cols)
    amt_col  = find_col(AMOUNT_CANDIDATES, cols)

    if date_col is None or amt_col is None:
        st.error(f"Required columns not found. Need a date column (e.g., {DATE_CANDIDATES}) "
                 f"and an amount column (e.g., {AMOUNT_CANDIDATES}).")
        st.stop()

    # Parse dates and coerce amounts
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True, utc=False)
    df = df.dropna(subset=[date_col])
    df[amt_col] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0)

    # Optional FX normalization
    if "fx_rate" in cols:
        df["amount_ccy"] = df[amt_col] * pd.to_numeric(df["fx_rate"], errors="coerce").fillna(1.0)
    else:
        df["amount_ccy"] = df[amt_col]

    # Basic cleaning
    df = df.drop_duplicates()
    # Winsorize 1% tails to reduce outlier impact
    q1, q99 = df["amount_ccy"].quantile([0.01, 0.99])
    df["amount_ccy"] = df["amount_ccy"].clip(q1, q99)

    return df, date_col

def monthly_aggregate(df, date_col):
    df["month"] = df[date_col].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month", as_index=False)["amount_ccy"].sum().rename(columns={"amount_ccy":"donations"})
    monthly = monthly.sort_values("month")
    return monthly

def backtest_and_fit(series: pd.Series):
    """Hold-out last 6 months for quick validation; fit SARIMAX(1,1,1)(1,1,1,12)."""
    if len(series) < 18:
        st.warning("Less than 18 monthly points—forecast quality may be limited.")

    # train/valid split
    h = min(6, max(1, len(series)//6))
    train, valid = series.iloc[:-h], series.iloc[-h:]

    def fit_model(y):
        try:
            model = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
            return model.fit(disp=False)
        except Exception:
            # Fallback simpler model
            model = SARIMAX(y, order=(0,1,1), seasonal_order=(0,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
            return model.fit(disp=False)

    res = fit_model(train)
    pred = res.get_forecast(steps=h).predicted_mean
    mae = mean_absolute_error(valid, pred)
    mape = (np.abs((valid - pred) / np.maximum(valid, 1e-9))).mean() * 100

    # Refit on full data for final forecast
    final_res = fit_model(series)
    return final_res, {"MAE": mae, "MAPE_%": mape}

def forecast_next(model, steps=36, alpha=0.2):
    fc = model.get_forecast(steps=steps)
    out = pd.DataFrame({
        "month": pd.date_range(model.data.dates[-1] + pd.offsets.MonthBegin(), periods=steps, freq="MS"),
        "point_forecast": fc.predicted_mean.values,
    })
    ci95 = fc.conf_int(alpha=0.05)
    ci80 = fc.conf_int(alpha=alpha)
    out["p80_low"],  out["p80_high"]  = ci80.iloc[:,0].values, ci80.iloc[:,1].values
    out["p95_low"],  out["p95_high"]  = ci95.iloc[:,0].values, ci95.iloc[:,1].values
    return out

# ---------- UI ----------
uploaded = st.file_uploader("Upload CSV", type=["csv"])
scenario_cols = st.columns(3)
with scenario_cols[0]:
    growth_adj = st.slider("Scenario: overall growth adjustment (%)", -30, 30, 0, step=1)
with scenario_cols[1]:
    donor_acq = st.number_input("Scenario: new recurring donors per month", min_value=0, value=0, step=5)
with scenario_cols[2]:
    avg_gift = st.number_input("Assumed avg gift of new recurring donors", min_value=0.0, value=0.0, step=10.0)

if uploaded:
    df, date_col = parse_csv(uploaded)

    st.subheader("Detected columns & cleaning")
    st.write(pd.DataFrame({
        "rows":[len(df)],
        "date_column":[date_col],
        "has_fx_rate":["fx_rate" in df.columns],
        "dedup_outlier_handling":["winsorized 1% tails"]
    }))

    # Monthly totals
    monthly = monthly_aggregate(df, date_col)
    if len(monthly) == 0:
        st.error("No monthly data after cleaning.")
        st.stop()

    # Limit to last 36 months for analysis window (if available)
    end = monthly["month"].max()
    start = end - pd.DateOffset(months=35)
    window = monthly[monthly["month"].between(start, end)] if len(monthly) > 36 else monthly.copy()

    st.subheader("Monthly donations (last 36 months if available)")
    fig = px.line(window, x="month", y="donations", markers=True, title="Monthly Donations")
    st.plotly_chart(fig, use_container_width=True)

    # Breakdown charts (top 10)
    def top_bar(colname, title):
        if colname in df.columns:
            top = (df.groupby(colname)["amount_ccy"].sum()
                   .sort_values(ascending=False).head(10).reset_index())
            st.plotly_chart(px.bar(top, x=colname, y="amount_ccy", title=title), use_container_width=True)

    cols = st.columns(3)
    with cols[0]: top_bar("donor", "Top Donors (sum)")
    with cols[1]: top_bar("campaign_type", "Top Campaigns (sum)")
    with cols[2]: top_bar("region", "Top Regions (sum)")

    # Model & forecast
    st.subheader("Model & Forecast (36 months ahead)")
    series = monthly.set_index("month")["donations"].asfreq("MS")
    model, metrics = backtest_and_fit(series)
    forecast = forecast_next(model, steps=36)

    # Apply simple scenario adjustments
    adj = 1.0 + (growth_adj / 100.0)
    forecast["point_forecast_adj"] = forecast["point_forecast"] * adj

    if donor_acq > 0 and avg_gift > 0:
        recurring_stream = donor_acq * avg_gift
        forecast["point_forecast_adj"] += recurring_stream

    st.write(pd.DataFrame([metrics]).style.format({"MAE":"{:,.0f}","MAPE_%":"{:,.2f}"}))
    fc_fig = px.line(forecast, x="month", y=["point_forecast","point_forecast_adj"], title="Forecast (point & scenario-adjusted)")
    st.plotly_chart(fc_fig, use_container_width=True)

    # Download outputs
    csv_buf = io.StringIO()
    cols_out = ["month","point_forecast","p80_low","p80_high","p95_low","p95_high","point_forecast_adj"]
    forecast[cols_out].to_csv(csv_buf, index=False)
    st.download_button("Download forecast.csv", data=csv_buf.getvalue(), file_name="forecast.csv", mime="text/csv")

    st.success("Done. You can commit this app and run:  streamlit run app.py")
else:
    st.info("Upload a CSV to begin. Minimum needed: a date column and an amount column.")
