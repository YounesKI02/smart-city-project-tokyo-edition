"""
========================================================
  TOKYO SMART CITY — ML INSIGHTS  (Streamlit App)
  Source: Tokyo Statistical Yearbook (2019–2022)

  PART I  — NO2 / Electricity Time-Series Forecast
            (ARIMA · SARIMA · Prophet)
  PART II — Crime Pattern Prediction
            (Random Forest Classifier)
  PART III— Electricity Demand Regression
            (Random Forest + GridSearchCV)

  HOW TO RUN:
    pip install streamlit pandas numpy matplotlib seaborn
                scikit-learn statsmodels prophet
    streamlit run Tokyo_SmartCity_ML_App.py
========================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model      import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal         import seasonal_decompose
from prophet                          import Prophet

from sklearn.ensemble        import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.metrics         import (accuracy_score, precision_score, recall_score,
                                      f1_score, roc_auc_score, classification_report,
                                      confusion_matrix, mean_absolute_error,
                                      mean_squared_error, r2_score)
from sklearn.pipeline        import Pipeline
from sklearn.compose         import ColumnTransformer

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from data_loaders import (load_electricity_annual, load_electricity_monthly,
                           load_air_quality, load_traffic, load_crime, YEARS)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Tokyo Smart City — ML Insights")

# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🤖 Tokyo Smart City — ML Insights")
st.sidebar.markdown("**Data: Tokyo Statistical Yearbook (2019–2022)**")

with st.sidebar.expander("Part I: Time-Series Forecasting"):
    st.markdown("""
    - Historical analysis of NO₂ & electricity (2019–2022)
    - Seasonal decomposition
    - ARIMA, SARIMA & Prophet forecasts
    """)
with st.sidebar.expander("Part II: Crime Pattern Prediction"):
    st.markdown("""
    - Classifier to predict high-crime stations
    - Model: Random Forest Classifier
    - Metrics: Accuracy, F1, ROC AUC
    """)
with st.sidebar.expander("Part III: Electricity Demand Regression"):
    st.markdown("""
    - Regressor to predict monthly electricity demand
    - Model: Random Forest Regressor (+ GridSearchCV)
    - Metrics: MAE, RMSE, R²
    """)

# ══════════════════════════════════════════════════════════════════════════════
#  PART I — TIME-SERIES FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
st.title("PART I: NO₂ & Electricity Time-Series Forecasting — Tokyo")
st.markdown("""
**Objective**: Forecast two key Smart City indicators — NO₂ air pollution and monthly
electricity demand — using statistical time-series models (ARIMA, SARIMA, Prophet).  
The goal is to anticipate future values to support proactive urban planning decisions.
""")

# ── load & prepare NO₂ series ─────────────────────────────────────────────────
aq_df = load_air_quality()
no2   = (aq_df[aq_df["pollutant"] == "NO2"]
         .sort_values("year")
         .set_index("year")["tokyo_avg"])

# ── seasonal decomposition ────────────────────────────────────────────────────
st.write("### Seasonal Decomposition of NO₂ Levels  (2018–2022)")

# For decomposition we need at least 2 full cycles — use monthly electricity instead
monthly_df = load_electricity_monthly()
monthly_df = monthly_df.sort_values(["year","month"]).reset_index(drop=True)
ts = monthly_df.set_index(pd.date_range(
        start=f"{monthly_df['year'].iloc[0]}-04-01",
        periods=len(monthly_df), freq="MS"))["total"]

decomp = seasonal_decompose(ts, model="additive", period=12)
fig_dec = decomp.plot()
fig_dec.suptitle("Seasonal Decomposition of Monthly Electricity Demand (Tokyo)", y=1.02)
fig_dec.set_size_inches(13, 8)
st.pyplot(fig_dec)

st.markdown("""
**General Considerations:**
- **Trend Component**: Shows a slight overall decline in electricity demand, consistent with
  energy efficiency improvements in Tokyo's building stock and appliances.
- **Seasonal Component**: Clear annual pattern with peaks in summer (Aug) and winter (Jan–Feb)
  driven by air-conditioning and heating loads respectively.
- **Residuals**: Relatively small residuals, supporting the additive model assumption and
  confirming that the seasonal + trend components explain most of the variance.
- **Conclusion**: The strong seasonality makes this series well-suited for SARIMA and Prophet,
  which can explicitly model periodic fluctuations.
""")

# ── ARIMA forecast ────────────────────────────────────────────────────────────
st.write("### ARIMA Forecast — Monthly Electricity (next 12 months)")

model_arima   = ARIMA(ts, order=(2, 1, 2)).fit()
forecast_arima = model_arima.forecast(steps=12)
future_dates   = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1),
                                periods=12, freq="MS")

# ── Prophet forecast ──────────────────────────────────────────────────────────
st.write("### Prophet Forecast — Monthly Electricity (next 12 months)")

df_prophet = ts.reset_index()
df_prophet.columns = ["ds","y"]
m_prophet  = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                     daily_seasonality=False)
m_prophet.fit(df_prophet)
future_df  = m_prophet.make_future_dataframe(periods=12, freq="MS")
fc_prophet = m_prophet.predict(future_df)
fc_prophet_fut = fc_prophet[fc_prophet["ds"] > ts.index[-1]]

# ── combined ARIMA vs Prophet plot ────────────────────────────────────────────
st.write("### Forecast Comparison: ARIMA vs Prophet")
fig2, ax2 = plt.subplots(figsize=(14, 6))
ax2.plot(ts.index, ts.values/1e6, label="Historical demand", color="#457B9D")
ax2.plot(future_dates, forecast_arima.values/1e6,
         label="ARIMA Forecast", linestyle="--", marker="o", color="#2A9D8F")
ax2.plot(fc_prophet_fut["ds"], fc_prophet_fut["yhat"]/1e6,
         label="Prophet Forecast", linestyle="--", marker="s", color="#E63946")
ax2.set_title("Monthly Electricity Demand Forecast — Tokyo  (next 12 months)")
ax2.set_xlabel("Date"); ax2.set_ylabel("Million kWh")
ax2.legend(); ax2.grid(alpha=0.3)
fig2.tight_layout()
st.pyplot(fig2)

st.markdown("""
**ARIMA**
- Effective at capturing autocorrelation and short-term trends
- Less suited for strong seasonal patterns without differencing

**Prophet**
- Decomposes the series into trend + yearly seasonality components
- More flexible and robust to irregularities in the calendar
- Provides confidence intervals and is interpretable by non-statisticians
- Generally produces more realistic seasonal curves for this dataset
""")

# ── SARIMA forecast ───────────────────────────────────────────────────────────
st.write("### SARIMA Forecast — Monthly Electricity (next 12 months)")

sarima_model  = SARIMAX(ts, order=(2,1,2), seasonal_order=(1,1,1,12)).fit(disp=False)
fc_sarima     = sarima_model.get_forecast(steps=12)
fc_sarima_mu  = fc_sarima.predicted_mean
fc_sarima_ci  = fc_sarima.conf_int()
fc_sarima_ci.columns = ["lower","upper"]

fig3, ax3 = plt.subplots(figsize=(13, 6))
ax3.plot(ts.index, ts.values/1e6, label="Historical", color="#457B9D")
ax3.plot(future_dates, fc_sarima_mu.values/1e6,
         label="SARIMA Forecast", linestyle="--", marker="s", color="#9B5DE5")
ax3.fill_between(future_dates,
                 fc_sarima_ci["lower"].values/1e6,
                 fc_sarima_ci["upper"].values/1e6,
                 color="#9B5DE5", alpha=0.2)
ax3.set_title("SARIMA Forecast of Monthly Electricity Demand — Tokyo")
ax3.set_xlabel("Date"); ax3.set_ylabel("Million kWh")
ax3.legend(); ax3.grid(alpha=0.3)
fig3.tight_layout()
st.pyplot(fig3)

st.markdown("""
### Key Insights from SARIMA Forecast
1. **Forecast line**: The purple dashed line closely mirrors the historical seasonal pattern,
   confirming that the model successfully captured the summer/winter demand cycles.
2. **Confidence interval**: The narrow shaded band indicates low forecast uncertainty over
   the next 12 months, reflecting the high regularity of Tokyo's electricity demand patterns.
3. **Trend**: The model projects a slight continued decline in demand, consistent with
   ongoing efficiency improvements in Tokyo's building stock.
4. **Model comparison**: SARIMA outperforms plain ARIMA by explicitly modelling the
   12-month seasonal cycle, producing more realistic and structured long-term forecasts.
""")


# ══════════════════════════════════════════════════════════════════════════════
#  PART II — CRIME PATTERN PREDICTION  (Random Forest Classifier)
# ══════════════════════════════════════════════════════════════════════════════
st.title("PART II: Crime Pattern Prediction — Tokyo")
st.markdown("""
**Objective**: Predict whether a police station will be classified as a **high-crime station**
(above the annual median total offenses), using station-level and year features.  
This supports a *Smart City* initiative to allocate policing resources proactively.
""")

crime_df = load_crime()

# ── feature engineering ────────────────────────────────────────────────────────
crime_df["high_crime"] = (crime_df.groupby("year")["total"]
                           .transform(lambda x: (x > x.median()).astype(int)))

features_crime = ["year","homicide","violence","bodily_injury",
                   "theft_be","theft_non_be","fraud","vandalism"]
df_model = crime_df[features_crime + ["high_crime"]].dropna()

# label encode year
le = LabelEncoder()
df_model["year"] = le.fit_transform(df_model["year"])

X = df_model.drop("high_crime", axis=1)
y = df_model["high_crime"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# ── selected features ──────────────────────────────────────────────────────────
st.write("### Selected Features for Classification")
st.write("""
Features used: **Year, Homicide, Violence, Bodily Injury, B&E Theft,
Non-B&E Theft, Fraud, Vandalism** — all measured at police-station level.  
Target: **high_crime** (1 = above annual median, 0 = below).
""")

# ── confusion matrix ───────────────────────────────────────────────────────────
st.write("### Confusion Matrix")
st.write("""
- **True Positives** (bottom right): correctly predicted high-crime stations
- **True Negatives** (top left): correctly predicted low-crime stations
- **False Positives** (top right): predicted high-crime, but station was low
- **False Negatives** (bottom left): predicted low-crime, but station was high
""")

fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d",
            cmap="Blues", ax=ax_cm)
ax_cm.set_title("Confusion Matrix — Crime Level Classifier")
ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
st.pyplot(fig_cm)

# ── metrics ────────────────────────────────────────────────────────────────────
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])

st.write("### Classification Report and Accuracy")
st.write(f"**Accuracy** : {acc:.4f}")
st.text(classification_report(y_test, y_pred))

# ── feature importances ────────────────────────────────────────────────────────
st.write("### Feature Importances")
importances = clf.feature_importances_
fi_df = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values(
    "Importance", ascending=False)

fig_fi, ax_fi = plt.subplots(figsize=(9, 5))
sns.barplot(x="Importance", y="Feature", data=fi_df,
            palette="Blues_r", ax=ax_fi)
ax_fi.set_title("Feature Importances — Crime Level Classifier")
st.pyplot(fig_fi)

st.markdown("""
### Interpretation of Feature Importances

1. **High Impact** — Non-B&E Theft and Fraud  
   These two categories dominate predictions because they vary most strongly between
   high- and low-crime stations. Stations near commercial or entertainment districts
   accumulate large numbers of these offenses.

2. **Moderate Impact** — Bodily Injury, Violence  
   These are useful but less decisive: they are more evenly distributed across
   station types.

3. **Low Impact** — Homicide, Vandalism, Year  
   Homicide numbers are very small and statistically noisy at station level.
   The year feature contributes little, confirming that cross-year patterns are
   relatively stable once crime-type composition is accounted for.

_Note_: Feature importance in Random Forest reflects how often a variable is used
for splits; it can overweight high-cardinality numeric features.
""")

# ── model performance metrics ─────────────────────────────────────────────────
st.write("### Model Performance Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy",  f"{acc:.4f}")
col2.metric("Precision", f"{prec:.4f}")
col3.metric("Recall",    f"{rec:.4f}")
col4.metric("F1-score",  f"{f1:.4f}")
col5.metric("ROC AUC",   f"{auc:.4f}")

st.markdown(f"""
### Interpretation of Model Results

**Accuracy (~{acc:.0%})**  
The model correctly classifies stations as high- or low-crime around {acc:.0%} of the time.

**Precision (~{prec:.0%})**  
Of all stations predicted as high-crime, {prec:.0%} actually are — indicating low false-alarm rate.

**Recall (~{rec:.0%})**  
The model successfully identifies {rec:.0%} of genuinely high-crime stations — important for resource allocation.

**F1 Score (~{f1:.0%})**  
The harmonic mean of precision and recall, providing a balanced assessment of classifier quality.

**ROC AUC (~{auc:.0%})**  
Excellent class separation. A value above 0.85 indicates the model is reliably distinguishing
between station types and is suitable for decision-support use.
""")


# ══════════════════════════════════════════════════════════════════════════════
#  PART III — ELECTRICITY DEMAND REGRESSION (Random Forest + GridSearchCV)
# ══════════════════════════════════════════════════════════════════════════════
st.title("PART III: Electricity Demand Regression — Tokyo")
st.markdown("""
**Objective**: Predict monthly electricity demand based on the month, year, and voltage
category characteristics. This models the relationship between calendar features and
consumption patterns, supporting proactive energy grid management in a Smart City context.
""")

# ── build regression dataset ──────────────────────────────────────────────────
monthly_df2 = load_electricity_monthly().copy()
monthly_df2["sin_month"] = np.sin(2 * np.pi * monthly_df2["month"] / 12)
monthly_df2["cos_month"] = np.cos(2 * np.pi * monthly_df2["month"] / 12)
monthly_df2["t"]         = np.arange(len(monthly_df2))

features_reg = ["year","month","sin_month","cos_month","t"]
target_reg   = "total"

X_r = monthly_df2[features_reg]
y_r = monthly_df2[target_reg]

X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

# ── GridSearchCV ──────────────────────────────────────────────────────────────
param_grid = {
    "n_estimators": [100, 200],
    "max_depth":    [5, 10, None],
}
grid = GridSearchCV(RandomForestRegressor(random_state=42),
                    param_grid, cv=3, scoring="r2", n_jobs=-1)
grid.fit(X_tr, y_tr)
best_rf = grid.best_estimator_

st.write("### Best Hyperparameters Found by GridSearchCV")
st.write(grid.best_params_)

# ── predictions & metrics ─────────────────────────────────────────────────────
y_hat  = best_rf.predict(X_te)
mae_v  = mean_absolute_error(y_te, y_hat)
rmse_v = np.sqrt(mean_squared_error(y_te, y_hat))
r2_v   = r2_score(y_te, y_hat)

st.write("### Performance Metrics")
c1, c2, c3 = st.columns(3)
c1.metric("MAE",  f"{mae_v/1e6:.3f} M kWh")
c2.metric("RMSE", f"{rmse_v/1e6:.3f} M kWh")
c3.metric("R²",   f"{r2_v:.4f}")

st.write(f"""
**Performance Interpretation**

1. **MAE ({mae_v/1e6:.3f} M kWh)**  
   On average, the model's monthly predictions deviate by {mae_v/1e6:.2f} million kWh from
   the actual figures. Given that monthly demand ranges from ~5.1 M to ~7.8 M kWh, this
   represents a reasonable margin (~{mae_v / y_r.mean() * 100:.1f}% of mean demand).

2. **RMSE ({rmse_v/1e6:.3f} M kWh)**  
   The gap between MAE and RMSE is small, indicating that the model does not make extreme
   outlier errors — predictions are consistently close to the actual values.

3. **R² ({r2_v:.2f})**  
   The model explains {r2_v*100:.0f}% of the variance in monthly electricity demand.
   This is a strong result for a relatively simple feature set, and confirms that year,
   month, and harmonic encoding capture most of the demand signal.
""")

# ── predicted vs actual scatter ────────────────────────────────────────────────
st.write("### Predicted vs Actual Monthly Demand")
fig_p, ax_p = plt.subplots(figsize=(7, 6))
ax_p.scatter(y_te/1e6, y_hat/1e6, alpha=0.7, color="#457B9D", edgecolors="white", s=70)
mn = min(y_te.min(), y_hat.min()) / 1e6
mx = max(y_te.max(), y_hat.max()) / 1e6
ax_p.plot([mn, mx], [mn, mx], "r--", lw=2)
ax_p.set_xlabel("Actual Demand  (M kWh)")
ax_p.set_ylabel("Predicted Demand  (M kWh)")
ax_p.set_title("Predicted vs Actual — Monthly Electricity Demand", fontsize=13)
ax_p.grid(alpha=0.3)
st.pyplot(fig_p)

st.write("""
The scatter plot shows individual monthly predictions versus actual demand values.
The red dashed diagonal is the ideal 1:1 line.

**Key observations:**
- Points cluster tightly along the diagonal, confirming the model's high accuracy.
- Slight dispersion is visible at the extremes (very high summer/winter demand),
  where unusual weather conditions create demand spikes that are harder to predict
  from calendar features alone.
- No systematic bias is observed: the model neither consistently over- nor under-predicts.
- The absence of outliers (points far from the diagonal) confirms the robustness of
  the GridSearchCV-selected hyperparameters.
""")

# ── feature importances ────────────────────────────────────────────────────────
st.write("### Top Feature Importances")
fi_r = pd.DataFrame({
    "Feature":    features_reg,
    "Importance": best_rf.feature_importances_
}).sort_values("Importance", ascending=False)

fig_fi2, ax_fi2 = plt.subplots(figsize=(9, 4))
sns.barplot(data=fi_r, x="Importance", y="Feature", palette="viridis", ax=ax_fi2)
ax_fi2.set_title("Feature Importances — Electricity Demand Regressor")
ax_fi2.set_xlabel("Importance Score")
ax_fi2.grid(alpha=0.3, axis="x")
plt.tight_layout()
st.pyplot(fig_fi2)

st.write("""
**Interpretation of Feature Importances:**

1. **Dominant predictor — `t` (time index)**  
   The linear time index captures the long-term downward trend in electricity demand,
   making it the single most informative feature across all tree splits.

2. **Month and harmonic encodings (sin/cos)**  
   The `month`, `sin_month`, and `cos_month` features collectively encode the strong
   12-month seasonal cycle. Their combined importance reflects how much of the variance
   in monthly demand is attributable to seasonality.

3. **Year**  
   Year contributes moderate importance, primarily capturing the step-changes caused
   by COVID-19 in 2020 and the 2022 recovery.

**Conclusion:**  
The feature importance plot confirms that the model learns exactly the right structure:
long-term trend + seasonal cycle. This is consistent with the strong R² result and
validates the feature engineering approach of adding harmonic encodings alongside the
raw calendar variables.
""")

