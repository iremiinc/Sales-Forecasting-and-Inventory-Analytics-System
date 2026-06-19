"""
analyse.py  –  Rossmann Sales Forecasting & Analytics
Produces every CSV / PNG that backend.py + index.html expect.

OUTPUT FILES (graphs/ folder)
──────────────────────────────
PNG charts:
  sales_trend.png, xgboost_prediction.png, error_distribution.png,
  actual_vs_predicted_scatter.png, absolute_error.png,
  holt_winters.png, sarima.png, inventory_decision.png,
  store_type_comparison.png, promo_effect.png, weekday_customers.png,
  holiday_impact.png, competition_bars.png, customer_trend.png

CSV data (admin):
  model_results.csv, predictions.csv, inventory_results.csv,
  kpi_results.csv, ai_insights.csv,
  store_type_stats.csv, store_ranking.csv,
  competition_analysis.csv

CSV data (per-store / viewer):
  store_summaries.csv, store_monthly.csv,
  store_promo.csv,     store_dow.csv,
  store_forecast.csv,  store_insights.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# ─── SETUP ────────────────────────────────────────────────────────────────────

GRAPH_DIR = "graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

def gpath(name):
    return os.path.join(GRAPH_DIR, name)

def save(fig, name):
    fig.savefig(gpath(name), dpi=100, bbox_inches="tight")
    plt.close(fig)

BLUE   = "#4f9cf9"
ORANGE = "#f97316"
PURPLE = "#a855f7"
GREEN  = "#22d3a0"
BG     = "#131826"
GRID   = "#1e2640"

def styled_fig(w=12, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors="#6b7a99")
    ax.xaxis.label.set_color("#6b7a99")
    ax.yaxis.label.set_color("#6b7a99")
    ax.title.set_color("#e8ecf4")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(color=GRID, linewidth=0.5, alpha=0.6)
    return fig, ax


# ─── 1. LOAD & MERGE ──────────────────────────────────────────────────────────

print("📂 Loading datasets...")
train = pd.read_csv("train.csv", low_memory=False)
store = pd.read_csv("store.csv")
data  = pd.merge(train, store, on="Store", how="left")


# ─── 2. FEATURE ENGINEERING ───────────────────────────────────────────────────

print("🔧 Feature engineering...")
data["Date"]    = pd.to_datetime(data["Date"])
data["Year"]    = data["Date"].dt.year
data["Month"]   = data["Date"].dt.month
data["Day"]     = data["Date"].dt.day
data["Weekday"] = data["Date"].dt.weekday
data["Promo"]   = data["Promo"].fillna(0)

data = data.sort_values(["Store", "Date"])
data["Lag_1"]          = data.groupby("Store")["Sales"].shift(1)
data["Lag_7"]          = data.groupby("Store")["Sales"].shift(7)
data["Rolling_Mean_3"] = data.groupby("Store")["Sales"].transform(lambda x: x.rolling(3).mean())
data["Rolling_Mean_7"] = data.groupby("Store")["Sales"].transform(lambda x: x.rolling(7).mean())
data = data.dropna().sort_values("Date")


# ─── 3. SALES TREND ───────────────────────────────────────────────────────────

print("📈 Sales trend chart...")
sales_trend = data.groupby("Date")["Sales"].sum()
fig, ax = styled_fig(12, 5)
ax.plot(sales_trend, color=BLUE, linewidth=1.4)
ax.fill_between(sales_trend.index, sales_trend.values, alpha=0.08, color=BLUE)
ax.set_title("Total Sales Over Time", fontsize=14, fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Total Sales")
save(fig, "sales_trend.png")


# ─── 4. CUSTOMER TREND ────────────────────────────────────────────────────────

print("👥 Customer trend chart...")
cust_trend = data.groupby("Date")["Customers"].sum()
fig, ax = styled_fig(12, 5)
ax.plot(cust_trend, color=GREEN, linewidth=1.2)
ax.fill_between(cust_trend.index, cust_trend.values, alpha=0.08, color=GREEN)
ax.set_title("Total Customers Over Time", fontsize=14, fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Total Customers")
save(fig, "customer_trend.png")


# ─── 5. XGBOOST MODEL ─────────────────────────────────────────────────────────

print("🤖 Training XGBoost model...")
features = ["Year","Month","Day","Weekday","Promo","Lag_1","Lag_7","Rolling_Mean_3","Rolling_Mean_7"]
X = data[features]; y = data["Sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# ─── 6. METRICS ───────────────────────────────────────────────────────────────

print("📊 Computing metrics...")
mae      = mean_absolute_error(y_test, y_pred)
rmse     = mean_squared_error(y_test, y_pred) ** 0.5
y_true   = y_test.values
mask     = y_true != 0
mape     = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
accuracy = 100 - mape

pd.DataFrame({
    "Metric": ["MAE", "RMSE", "MAPE", "Accuracy"],
    "Value":  [mae, rmse, mape, accuracy]
}).to_csv(gpath("model_results.csv"), index=False)

pred_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
pred_df["Error"]          = pred_df["Actual"] - pred_df["Predicted"]
pred_df["Absolute_Error"] = np.abs(pred_df["Error"])
pred_df.to_csv(gpath("predictions.csv"), index=False)


# ─── 7. PREDICTION CHARTS ─────────────────────────────────────────────────────

print("📉 Prediction charts...")

fig, ax = styled_fig(12, 5)
ax.plot(y_test.values[:100], label="Actual",    color=BLUE,   linewidth=1.5)
ax.plot(y_pred[:100],         label="Predicted", color=ORANGE, linewidth=1.5, linestyle="--")
ax.legend(facecolor=BG, edgecolor=GRID, labelcolor="#e8ecf4")
ax.set_title("Actual vs Predicted Sales"); ax.set_xlabel("Sample Index"); ax.set_ylabel("Sales")
save(fig, "xgboost_prediction.png")

errors = y_test.values - y_pred
fig, ax = styled_fig(12, 5)
ax.hist(errors, bins=50, color=BLUE, edgecolor=BG, alpha=0.9)
ax.set_title("Error Distribution"); ax.set_xlabel("Prediction Error"); ax.set_ylabel("Frequency")
save(fig, "error_distribution.png")

fig, ax = styled_fig(7, 7)
ax.scatter(y_test.values[:1000], y_pred[:1000], alpha=0.35, color=BLUE, s=8)
mn = min(y_test.values[:1000].min(), y_pred[:1000].min())
mx = max(y_test.values[:1000].max(), y_pred[:1000].max())
ax.plot([mn,mx],[mn,mx], linestyle="--", color=ORANGE, linewidth=1.5)
ax.set_title("Actual vs Predicted Scatter Plot"); ax.set_xlabel("Actual Sales"); ax.set_ylabel("Predicted Sales")
save(fig, "actual_vs_predicted_scatter.png")

fig, ax = styled_fig(12, 5)
ax.plot(np.abs(errors[:200]), color=PURPLE, linewidth=1)
ax.fill_between(range(200), np.abs(errors[:200]), alpha=0.1, color=PURPLE)
ax.set_title("Absolute Error Over Samples"); ax.set_xlabel("Sample Index"); ax.set_ylabel("Absolute Error")
save(fig, "absolute_error.png")


# ─── 8. TIME SERIES ───────────────────────────────────────────────────────────

ts_data = data.groupby("Date")["Sales"].sum().sort_index()

print("🌡 Holt-Winters model...")
hw_model = ExponentialSmoothing(ts_data, trend="add", seasonal="add", seasonal_periods=7).fit()
hw_pred  = hw_model.forecast(200)

fig, ax = styled_fig(12, 5)
ax.plot(ts_data[-200:],  label="Actual",       color=BLUE,   linewidth=1.5)
ax.plot(hw_pred,          label="Holt-Winters", color=ORANGE, linewidth=1.5)
ax.legend(facecolor=BG, edgecolor=GRID, labelcolor="#e8ecf4")
ax.set_title("Holt-Winters Forecast"); ax.set_xlabel("Date"); ax.set_ylabel("Sales")
save(fig, "holt_winters.png")

print("📡 SARIMA model (may take a while)...")
sarima_model = SARIMAX(ts_data, order=(1,1,1), seasonal_order=(1,1,1,7)).fit(disp=False)
sarima_pred  = sarima_model.forecast(200)

fig, ax = styled_fig(12, 5)
ax.plot(ts_data[-200:], label="Actual", color=BLUE,   linewidth=1.5)
ax.plot(sarima_pred,     label="SARIMA", color=PURPLE, linewidth=1.5)
ax.legend(facecolor=BG, edgecolor=GRID, labelcolor="#e8ecf4")
ax.set_title("SARIMA Forecast"); ax.set_xlabel("Date"); ax.set_ylabel("Sales")
save(fig, "sarima.png")


# ─── 9. STORE TYPE COMPARISON ─────────────────────────────────────────────────

print("🏪 Store type analysis...")
type_stats = data[data["Sales"] > 0].groupby("StoreType").agg(
    Avg_Sales     = ("Sales",     "mean"),
    Avg_Customers = ("Customers", "mean"),
    Store_Count   = ("Store",     "nunique"),
    Promo_Rate    = ("Promo",     "mean")
).reset_index().rename(columns={"StoreType":"StoreType"})
type_stats["Promo_Rate"] = (type_stats["Promo_Rate"] * 100).round(1)
type_stats = type_stats.sort_values("Avg_Sales", ascending=False)
type_stats.to_csv(gpath("store_type_stats.csv"), index=False)

colors_map = {"a": BLUE, "b": ORANGE, "c": PURPLE, "d": GREEN}
bar_colors = [colors_map.get(t, BLUE) for t in type_stats["StoreType"]]

fig, ax = styled_fig(10, 5)
bars = ax.bar(
    [f"Type {t.upper()}" for t in type_stats["StoreType"]],
    type_stats["Avg_Sales"],
    color=bar_colors, width=0.5
)
ax.set_title("Avg Sales by Store Type"); ax.set_xlabel("Store Type"); ax.set_ylabel("Avg Sales (€)")
for bar, val in zip(bars, type_stats["Avg_Sales"]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50, f"€{val:,.0f}",
            ha="center", va="bottom", fontsize=10, color="#e8ecf4")
save(fig, "store_type_comparison.png")


# ─── 10. PROMO EFFECT ─────────────────────────────────────────────────────────

print("🏷 Promo effect analysis...")
promo_stats = data[data["Sales"]>0].groupby("Promo")["Sales"].mean()
no_promo = promo_stats.get(0, 0)
yes_promo = promo_stats.get(1, 0)

fig, ax = styled_fig(8, 5)
ax.bar(["No Promotion", "With Promotion"], [no_promo, yes_promo],
       color=[BLUE, ORANGE], width=0.45)
ax.set_title("Promo vs No-Promo Sales"); ax.set_ylabel("Avg Sales (€)")
save(fig, "promo_effect.png")


# ─── 11. WEEKDAY CUSTOMERS ────────────────────────────────────────────────────

print("📅 Weekday analysis...")
day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
weekday_stats = data[data["Sales"]>0].groupby("Weekday")["Customers"].mean()

fig, ax = styled_fig(10, 5)
ax.bar([day_names[i] for i in weekday_stats.index], weekday_stats.values,
       color=PURPLE, width=0.55)
ax.set_title("Avg Customers by Day of Week"); ax.set_ylabel("Avg Customers")
save(fig, "weekday_customers.png")


# ─── 12. HOLIDAY IMPACT ───────────────────────────────────────────────────────

print("🎉 Holiday impact analysis...")
hol = data[data["Sales"]>0].groupby("SchoolHoliday")["Sales"].mean()
labels = ["No Holiday", "School Holiday"]
values = [hol.get(0,0), hol.get(1,0)]

fig, ax = styled_fig(8, 5)
ax.bar(labels, values, color=[BLUE, GREEN], width=0.45)
ax.set_title("Sales: School Holiday vs Regular Days"); ax.set_ylabel("Avg Sales (€)")
save(fig, "holiday_impact.png")


# ─── 13. COMPETITION ANALYSIS ─────────────────────────────────────────────────

print("🏆 Competition analysis...")
comp_data = data[data["Sales"]>0].copy()
comp_data["CompetitionDistance"] = pd.to_numeric(comp_data["CompetitionDistance"], errors="coerce")
comp_data = comp_data.dropna(subset=["CompetitionDistance"])
comp_data["DistBucket"] = pd.cut(
    comp_data["CompetitionDistance"],
    bins=[0, 500, 1000, 2000, 5000, 100000],
    labels=["<500m","500m-1km","1-2km","2-5km",">5km"]
)
comp_stats = comp_data.groupby("DistBucket", observed=True)["Sales"].mean().reset_index()
comp_stats.columns = ["Distance", "Avg_Sales"]
comp_stats.to_csv(gpath("competition_analysis.csv"), index=False)

fig, ax = styled_fig(10, 5)
ax.bar(comp_stats["Distance"].astype(str), comp_stats["Avg_Sales"],
       color=ORANGE, width=0.55)
ax.set_title("Avg Sales by Distance to Competitor"); ax.set_xlabel("Distance"); ax.set_ylabel("Avg Sales (€)")
save(fig, "competition_bars.png")


# ─── 14. INVENTORY DECISION ───────────────────────────────────────────────────

print("📦 Inventory decisions...")
inv = data[data["Sales"]>0].groupby("Store")["Sales"].agg(["mean","std"]).reset_index()
inv.columns = ["Store","Average_Demand","Demand_STD"]
inv["Lead_Time"]          = 7
inv["Z"]                  = 1.65
inv["ROP"]                = inv["Average_Demand"] * inv["Lead_Time"]
inv["Safety_Stock"]       = inv["Z"] * inv["Demand_STD"] * np.sqrt(inv["Lead_Time"])
inv["Recommended_Stock"]  = inv["ROP"] + inv["Safety_Stock"]
inv.to_csv(gpath("inventory_results.csv"), index=False)

top10 = inv.sort_values("Recommended_Stock", ascending=False).head(10)
fig, ax = styled_fig(12, 5)
ax.bar(top10["Store"].astype(str), top10["Recommended_Stock"], color=BLUE)
ax.set_title("Top 10 Stores by Recommended Stock Level")
ax.set_xlabel("Store"); ax.set_ylabel("Recommended Stock")
save(fig, "inventory_decision.png")


# ─── 15. STORE RANKING ────────────────────────────────────────────────────────

print("🏅 Store ranking...")
store_perf = data[data["Sales"]>0].groupby("Store").agg(
    Avg_Sales     = ("Sales",     "mean"),
    Avg_Customers = ("Customers", "mean"),
    Total_Sales   = ("Sales",     "sum"),
).reset_index()
max_sales = store_perf["Avg_Sales"].max()
store_perf["Performance_Score"] = (store_perf["Avg_Sales"] / max_sales * 100).round(2)
store_perf = store_perf.sort_values("Avg_Sales", ascending=False).reset_index(drop=True)
store_perf.insert(0, "Rank", range(1, len(store_perf)+1))
store_perf["Avg_Sales"]     = store_perf["Avg_Sales"].round(2)
store_perf["Avg_Customers"] = store_perf["Avg_Customers"].round(2)
store_perf.to_csv(gpath("store_ranking.csv"), index=False)


# ─── 16. GLOBAL KPIs ──────────────────────────────────────────────────────────

print("📋 Global KPIs...")
total_sales   = data["Sales"].sum()
avg_daily     = data.groupby("Date")["Sales"].sum().mean()
promo_lift    = ((data[data["Promo"]==1]["Sales"].mean() / data[data["Promo"]==0]["Sales"].mean()) - 1) * 100
best_type     = type_stats.iloc[0]["StoreType"].upper()
hol_lift      = ((data[data["SchoolHoliday"]==1]["Sales"].mean() / data[data["SchoolHoliday"]==0]["Sales"].mean()) - 1) * 100
active_stores = data["Store"].nunique()

kpi_df = pd.DataFrame({
    "KPI": [
        "Total Sales", "Avg Daily Sales", "Promo Sales Lift",
        "Best Store Type", "Holiday Sales Lift",
        "Active Stores", "Model Accuracy", "Forecast Horizon"
    ],
    "Value": [
        f"€{total_sales/1e9:.2f}B",
        f"€{avg_daily:,.0f}",
        f"+{promo_lift:.1f}%",
        f"Type {best_type}",
        f"+{hol_lift:.1f}%",
        str(active_stores),
        f"{accuracy:.1f}%",
        "200 days"
    ]
})
kpi_df.to_csv(gpath("kpi_results.csv"), index=False)


# ─── 17. AI INSIGHTS (global) ─────────────────────────────────────────────────

print("🧠 Generating AI insights...")

promo_pct  = data["Promo"].mean() * 100
sun_sales  = data[data["Weekday"]==6]["Sales"].mean()
mon_sales  = data[data["Weekday"]==0]["Sales"].mean()
top_store  = store_perf.iloc[0]["Store"]

insights = [
    {
        "Category": "Promotions",
        "Icon": "🏷️",
        "Insight": f"Promotions increase daily sales by {promo_lift:.1f}% on average across all stores. Currently {promo_pct:.1f}% of days have active promotions.",
        "Action": "Consider extending promotions to underperforming stores.",
        "Priority": "High"
    },
    {
        "Category": "Store Performance",
        "Icon": "🏆",
        "Insight": f"Store #{int(top_store)} is the top performer with highest average daily sales. Type {best_type} stores outperform other types.",
        "Action": f"Study Store #{int(top_store)}'s strategy for replication.",
        "Priority": "Medium"
    },
    {
        "Category": "Weekly Pattern",
        "Icon": "📅",
        "Insight": f"Monday shows {((mon_sales/sun_sales)-1)*100:.1f}% higher sales vs Sunday. Most stores are closed on Sundays.",
        "Action": "Optimize Mon/Fri staffing and stock levels.",
        "Priority": "Medium"
    },
    {
        "Category": "Performance",
        "Icon": "🤖",
        "Insight": f"The XGBoost model achieves {accuracy:.1f}% accuracy (MAE: {mae:,.0f}, RMSE: {rmse:,.0f}). Performance is strongest for typical sales days.",
        "Action": "Consider ensemble methods for peak/promo days.",
        "Priority": "Low"
    },
    {
        "Category": "Inventory",
        "Icon": "📦",
        "Insight": f"Inventory recommendations are computed for all {active_stores} stores using ROP + Safety Stock (Z=1.65, Lead Time=7 days).",
        "Action": "Review stores with Recommended Stock > 80,000 for potential overstock.",
        "Priority": "Medium"
    },
    {
        "Category": "Seasonal",
        "Icon": "🎉",
        "Insight": f"School holidays {'increase' if hol_lift >= 0 else 'decrease'} sales by {abs(hol_lift):.1f}% compared to regular school days.",
        "Action": "Pre-stock stores in holiday-heavy regions before school breaks.",
        "Priority": "High"
    },
]
pd.DataFrame(insights).to_csv(gpath("ai_insights.csv"), index=False)


# ─── 18. PER-STORE CSVs (viewer) ──────────────────────────────────────────────

print("🏪 Per-store summaries (viewer data)...")

open_data = data[data["Open"] != 0].copy() if "Open" in data.columns else data.copy()
open_data = open_data[open_data["Sales"] > 0]

# --- store_summaries.csv ---
store_info = store[["Store","StoreType","Assortment"]].drop_duplicates()

by_store = open_data.groupby("Store").agg(
    Avg_Daily_Sales = ("Sales",     "mean"),
    Avg_Customers   = ("Customers", "mean"),
    Promo_Rate      = ("Promo",     "mean"),
).reset_index()

# YoY growth (2014 vs 2013)
yoy = open_data.copy()
yoy["YearGrp"] = yoy["Year"].apply(lambda y: 2013 if y == 2013 else (2014 if y == 2014 else None))
yoy = yoy.dropna(subset=["YearGrp"])
yoy_pivot = yoy.groupby(["Store","YearGrp"])["Sales"].mean().unstack(fill_value=np.nan)
yoy_growth = pd.DataFrame(index=yoy_pivot.index)
if 2013 in yoy_pivot.columns and 2014 in yoy_pivot.columns:
    yoy_growth["YoY_Growth_Pct"] = ((yoy_pivot[2014] - yoy_pivot[2013]) / yoy_pivot[2013] * 100).round(2)
else:
    yoy_growth["YoY_Growth_Pct"] = 0.0
yoy_growth = yoy_growth.reset_index()

summaries = by_store.merge(store_info, on="Store", how="left").merge(yoy_growth, on="Store", how="left")
summaries["Store"] = summaries["Store"].astype(int)
summaries["Avg_Daily_Sales"] = summaries["Avg_Daily_Sales"].round(2)
summaries["Avg_Customers"]   = summaries["Avg_Customers"].round(2)
summaries["Promo_Rate"]      = (summaries["Promo_Rate"] * 100).round(1)
summaries["YoY_Growth_Pct"]  = summaries["YoY_Growth_Pct"].fillna(0).round(2)
summaries.to_csv(gpath("store_summaries.csv"), index=False)

# --- store_monthly.csv ---
print("   → store_monthly.csv")
monthly = open_data.copy()
monthly["MonthYear"] = monthly["Date"].dt.to_period("M").astype(str)
store_monthly = monthly.groupby(["Store","MonthYear"]).agg(
    Avg_Sales     = ("Sales",     "mean"),
    Total_Sales   = ("Sales",     "sum"),
    Avg_Customers = ("Customers", "mean"),
).reset_index()
store_monthly["Store"]        = store_monthly["Store"].astype(int)
store_monthly["Avg_Sales"]     = store_monthly["Avg_Sales"].round(2)
store_monthly["Total_Sales"]   = store_monthly["Total_Sales"].round(2)
store_monthly["Avg_Customers"] = store_monthly["Avg_Customers"].round(2)
store_monthly.to_csv(gpath("store_monthly.csv"), index=False)

# --- store_promo.csv ---
print("   → store_promo.csv")
promo_by = open_data.groupby(["Store","Promo"])["Sales"].mean().unstack(fill_value=0).reset_index()
promo_by.columns.name = None
promo_by = promo_by.rename(columns={0: "Sales_No_Promo", 1: "Sales_Promo"})
if "Sales_No_Promo" not in promo_by.columns: promo_by["Sales_No_Promo"] = 0
if "Sales_Promo"    not in promo_by.columns: promo_by["Sales_Promo"]    = 0
promo_by["Promo_Lift_Pct"] = np.where(
    promo_by["Sales_No_Promo"] > 0,
    ((promo_by["Sales_Promo"] - promo_by["Sales_No_Promo"]) / promo_by["Sales_No_Promo"] * 100).round(2),
    0
)
promo_by["Store"] = promo_by["Store"].astype(int)
promo_by.to_csv(gpath("store_promo.csv"), index=False)

# --- store_dow.csv (day of week) ---
print("   → store_dow.csv")
DAY_NAMES = {0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"}
dow = open_data.groupby(["Store","Weekday"]).agg(
    Avg_Sales     = ("Sales",     "mean"),
    Avg_Customers = ("Customers", "mean"),
).reset_index()
dow["Store"]         = dow["Store"].astype(int)
dow["Day_Name"] = dow["Weekday"].map(DAY_NAMES)
dow["Avg_Sales"]     = dow["Avg_Sales"].round(2)
dow["Avg_Customers"] = dow["Avg_Customers"].round(2)
dow.to_csv(gpath("store_dow.csv"), index=False)

# --- store_forecast.csv (Holt-Winters per-store, 30 days) ---
print("   → store_forecast.csv  (this may take a moment)...")

# Use hw model to generate 30-day per-store forecasts from last known date
last_date   = open_data["Date"].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq="D")

# For speed: apply global HW ratio per day-of-week to each store's avg
global_hw_30 = hw_model.forecast(30)
# Normalise to ratio around mean
hw_ratio = (global_hw_30 / global_hw_30.mean()).values   # shape (30,)

store_avg = summaries[["Store","Avg_Daily_Sales"]].set_index("Store")["Avg_Daily_Sales"]

forecast_rows = []
for store_id, avg in store_avg.items():
    for i, d in enumerate(future_dates):
        forecast_rows.append({
            "Store":          store_id,
            "Date":           d.strftime("%Y-%m-%d"),
            "Predicted_Sales": round(avg * hw_ratio[i], 2)
        })

store_forecast_df = pd.DataFrame(forecast_rows)
store_forecast_df["Store"] = store_forecast_df["Store"].astype(int)
store_forecast_df.to_csv(gpath("store_forecast.csv"), index=False)

# --- store_insights.csv (per-store AI insights) ---
print("   → store_insights.csv...")

store_ins_rows = []
for _, row in summaries.iterrows():
    sid    = row["Store"]
    prate  = row["Promo_Rate"]
    growth = row["YoY_Growth_Pct"]
    avg_s  = row["Avg_Daily_Sales"]

    # Promo insight
    store_ins_rows.append({
        "Store":    sid,
        "Category": "Promotion",
        "Icon":     "🏷️",
        "Insight":  f"Promotions run on {prate:.1f}% of open days. " +
                    ("High promo frequency — monitor margin impact." if prate > 50
                     else "Consider increasing promotion days to drive traffic."),
        "Action":   "Review promotion calendar.",
        "Priority": "High" if prate < 20 else "Medium"
    })

    # Growth insight
    store_ins_rows.append({
        "Store":    sid,
        "Category": "Performance",
        "Icon":     "📈" if growth >= 0 else "📉",
        "Insight":  f"Year-over-year sales {'grew' if growth >= 0 else 'declined'} by {abs(growth):.1f}%. "
                    + ("Strong performance — maintain current strategy." if growth > 5
                       else "Focus on promotions and assortment changes." if growth < 0
                       else "Steady growth — look for optimisation opportunities."),
        "Action":   "Review pricing and product mix." if growth < 0 else "Scale successful strategies.",
        "Priority": "High" if growth < -5 else "Medium"
    })

    # Inventory insight
    inv_row = inv[inv["Store"] == sid]
    if not inv_row.empty:
        rec_stock = inv_row.iloc[0]["Recommended_Stock"]
        store_ins_rows.append({
            "Store":    sid,
            "Category": "Inventory",
            "Icon":     "📦",
            "Insight":  f"Recommended stock level: {rec_stock:,.0f} units (ROP + Safety Stock at 95% service level).",
            "Action":   "Order stock before reaching ROP to avoid stockouts.",
            "Priority": "Medium"
        })

store_insights_df = pd.DataFrame(store_ins_rows)
store_insights_df["Store"] = store_insights_df["Store"].astype(int)
store_insights_df.to_csv(gpath("store_insights.csv"), index=False)


# ─── DONE ─────────────────────────────────────────────────────────────────────

print("\n✅ ALL MODELS AND EXPORTS COMPLETED SUCCESSFULLY!")
print(f"   MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%  Accuracy={accuracy:.2f}%")
print(f"   Graphs saved to: {GRAPH_DIR}/")
print(f"   Per-store forecasts: {len(store_forecast_df)} rows for {open_data['Store'].nunique()} stores")
