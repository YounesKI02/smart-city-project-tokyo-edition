# 🏙 Tokyo Smart City — Data Science Project

> Data: Tokyo Statistical Yearbook (2019–2022)  
> Domains: Air Quality · Electricity · Crime · Traffic

---

## 📁 Project Structure

```
Tokyo_SmartCity/
│
├── data/                                         
│   ├── Air_Pollutant_Measurement_Averages_22.csv
│   ├── Criminal_Offenses_..._19/20/21/22.csv
│   ├── Electricity_Demand_19/20/21/22.csv
│   └── Traffic_Volume_..._19/20/21/22.csv
│
├── data_loaders.py                  ← Shared data loading & cleaning module
├── Tokyo_SmartCity_App.py           ← 📊 Streamlit Dashboard (EDA)
├── Tokyo_SmartCity_ML_App.py        ← 🤖 Streamlit ML Insights App
└── README.md
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn \
            statsmodels prophet folium streamlit-folium
```

### 2. Run the EDA Dashboard
```bash
streamlit run Tokyo_SmartCity_App.py
```

### 3. Run the ML Insights App
```bash
streamlit run Tokyo_SmartCity_ML_App.py
```

---

## 📊 App 1 — Tokyo Smart City Dashboard (EDA)

| Section | Content |
|---|---|
| **1. Air Quality & Electricity** | Pollutant trends, ward vs city avg, annual/monthly demand, market rate shift |
| **2. Crime Analysis** | Total offenses, category breakdown, % change vs 2019, top stations & districts |
| **3. Traffic & Urban Mobility** | Vehicle counts, type breakdown, busiest routes, 2019→2022 change |

---

## 🤖 App 2 — ML Insights

| Part | Model | Task |
|---|---|---|
| **Part I** | ARIMA · SARIMA · Prophet | Monthly electricity demand forecast |
| **Part II** | Random Forest Classifier | Predict high-crime vs low-crime police stations |
| **Part III** | Random Forest Regressor + GridSearchCV | Predict monthly electricity demand |

---

## 🔑 Key Findings

| Domain | Finding |
|---|---|
| Air Quality | NO₂ down 14% (2018→2022); PM2.5 improved from 12.3 → 9.0 µg/m³ |
| Electricity | Total demand fell from 78.7M kWh (2017) to 75.3M kWh (2022) |
| Crime | Total offenses dropped 28% from 2019 to 2021 (COVID effect); fraud increased |
| Traffic | Expressway volumes fully recovered by 2022; Tomei Expressway #1 busiest |

---

## 📦 Dependencies

```
pandas · numpy · matplotlib · seaborn
scikit-learn · statsmodels · prophet
streamlit · folium · streamlit-folium
```
