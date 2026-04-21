"""
========================================================
  TOKYO SMART CITY — STREAMLIT DASHBOARD
  Source: Tokyo Statistical Yearbook (2019–2022)

  Structure:
    Part I  — Air Quality & Electricity Analysis
    Part II — Crime Analysis
    Part III— Traffic & Urban Mobility Analysis
  
  ML Script (Tokyo_SmartCity_ML_Insights.py):
    Part I  — NO2 / PM2.5 Time-Series Forecast (ARIMA, SARIMA, Prophet)
    Part II — Crime Arrest Prediction (Random Forest Classifier)
    Part III— Electricity Demand Regression (Random Forest + GridSearchCV)

  HOW TO RUN:
    1. Place this file and data_loaders.py in the same folder as your data/ folder
    2. pip install streamlit pandas matplotlib seaborn scikit-learn statsmodels prophet folium streamlit-folium
    3. streamlit run Tokyo_SmartCity_App.py
========================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Tokyo Smart City Dashboard")

# ── import shared loaders ──────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from data_loaders import (load_electricity_annual, load_electricity_monthly,
                           load_air_quality, load_traffic, load_crime,
                           CRIME_TOTALS, CRIME_CATS, YEARS)

# ── sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("🏙 Tokyo Smart City Dashboard")
st.sidebar.markdown("**Data: Tokyo Statistical Yearbook (2019–2022)**")

section = st.sidebar.selectbox("Choose a section", [
    "1. Air Quality & Electricity",
    "2. Crime Analysis",
    "3. Traffic & Urban Mobility",
])

PALETTE = ["#457B9D","#E63946","#2A9D8F","#F4A261","#9B5DE5","#8D99AE"]

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — AIR QUALITY & ELECTRICITY
# ══════════════════════════════════════════════════════════════════════════════
if section == "1. Air Quality & Electricity":
    st.title("Air Quality & Electricity Analysis — Tokyo")
    st.markdown("""
    This section presents an exploratory analysis of two key environmental indicators for
    Tokyo: **air pollutant levels** measured across monitoring stations (2018–2022) and
    **electricity demand** trends (2016–2022). Together they reveal how Tokyo's energy
    consumption and urban air quality have evolved, and how the COVID-19 pandemic created
    a measurable shift in both.
    """)

    # ── load data ──────────────────────────────────────────────────────────
    aq_df      = load_air_quality()
    annual_df  = load_electricity_annual()
    monthly_df = load_electricity_monthly()

    # ── 1a: pollutant trend over time ──────────────────────────────────────
    st.subheader("Temporal Evolution of Air Pollutants")
    st.write("""
    The chart below shows how average concentrations of Tokyo's five measured pollutants
    have changed each year. A consistent downward trend in NO₂ and PM2.5 reflects the
    combined effect of stricter emission controls, a modal shift toward electric vehicles,
    and reduced industrial activity — particularly visible in the drop after 2019.
    """)

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    for poll, clr in zip(aq_df["pollutant"].unique(), PALETTE):
        sub = aq_df[aq_df["pollutant"] == poll].sort_values("year")
        ax1.plot(sub["year"], sub["tokyo_avg"], marker="o", lw=2.2, ms=7,
                 color=clr, label=poll)
    ax1.set_title("Average Pollutant Levels by Year — Tokyo-to", fontsize=13)
    ax1.set_xlabel("Fiscal Year"); ax1.set_ylabel("Concentration")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
    st.pyplot(fig1)

    st.write("""
    **Key observations:**
    - **NO₂** and **SPM** show a steady decline from 2018 to 2022, in line with Tokyo's
      Clean Air Program and post-COVID reductions in vehicle traffic.
    - **Photochemical oxidants (Ox)** remain relatively stable, as they are partly driven
      by natural photochemical reactions and regional background concentrations.
    - **PM2.5** drops significantly between 2018 and 2021 (from 12.3 to 8.5 µg/m³),
      well below the WHO guideline of 15 µg/m³, before a slight uptick in 2022.
    - **SO₂** has been near the detection limit since 2019, indicating effective
      desulfurisation in industrial and power-generation sources.
    """)

    # ── 1b: ward average vs tokyo average ──────────────────────────────────
    st.subheader("Ward (Ku) Average vs Tokyo-wide Average")
    st.write("""
    Urban wards (ku) systematically show higher pollutant concentrations than the
    Tokyo-wide average due to higher population density, traffic volumes, and
    commercial activity concentrated in the 23 special wards.
    """)

    poll_select = st.selectbox("Select pollutant", aq_df["pollutant"].unique())
    sub2 = aq_df[aq_df["pollutant"] == poll_select].sort_values("year")
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.fill_between(sub2["year"], sub2["tokyo_avg"], sub2["ku_avg"],
                     alpha=0.2, color="#457B9D", label="Gap (ku higher)")
    ax2.plot(sub2["year"], sub2["tokyo_avg"], "o-", color="#457B9D", lw=2.2, ms=8,
             label="Tokyo-to avg")
    ax2.plot(sub2["year"], sub2["ku_avg"],    "s--",color="#E63946", lw=2,   ms=8,
             label="Ward (ku) avg")
    ax2.set_title(f"{poll_select} — Ward vs City Average", fontsize=13)
    ax2.set_xlabel("Year"); ax2.legend(); ax2.grid(alpha=0.3)
    st.pyplot(fig2)

    # ── 1c: annual electricity trend ───────────────────────────────────────
    st.subheader("Annual Electricity Demand Trend (2016–2022)")
    st.write("""
    Total electricity consumption in Tokyo has declined gradually over the 2016–2022
    period, driven by energy efficiency improvements, building insulation upgrades,
    and the economic slowdown caused by COVID-19 in 2020. A slight rebound in 2021
    reflects the post-lockdown recovery before a new decline in 2022.
    """)

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    yr = annual_df.sort_values("year")
    ax3.fill_between(yr["year"], yr["total"]/1e6, color="#2A9D8F", alpha=0.2)
    ax3.plot(yr["year"], yr["total"]/1e6, "o-", color="#2A9D8F", lw=2.5, ms=8)
    for _, r in yr.iterrows():
        ax3.annotate(f'{r["total"]/1e6:.1f}M', (r["year"], r["total"]/1e6),
                     textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=9)
    ax3.set_title("Annual Total Electricity Demand  (million kWh)", fontsize=13)
    ax3.set_xlabel("Fiscal Year"); ax3.set_ylabel("Million kWh")
    ax3.set_xticks(yr["year"]); ax3.grid(alpha=0.3)
    st.pyplot(fig3)

    st.write("""
    **Key observations:**
    - Peak demand of **78.7 M kWh** was recorded in **FY2017**, followed by a
      consistent decline.
    - The 2020 dip aligns with COVID-19 restrictions: office closures, remote working,
      and reduced commercial activity lowered industrial and high-voltage consumption.
    - Despite the 2021 recovery, total demand in 2022 (**75.3 M kWh**) is the lowest
      in the observed window, suggesting that structural efficiency gains are lasting.
    """)

    # ── 1d: voltage category breakdown ────────────────────────────────────
    st.subheader("Electricity Breakdown by Voltage Category")
    st.write("""
    Electricity consumption is split into three voltage tiers: **Extra High** (large
    industrial/commercial users), **High** (mid-size businesses), and **Low** (households
    and small businesses). The most notable structural shift is within the Low-voltage
    segment, where **non-regulated (market-rate)** contracts have grown rapidly as Japan's
    electricity market has been progressively liberalised.
    """)

    fig4, ax4 = plt.subplots(figsize=(10, 4))
    yf = annual_df[annual_df["year"] >= 2019].sort_values("year")
    bot = np.zeros(len(yf))
    for col, lbl, clr in [("extra_high","Extra High Voltage","#9B5DE5"),
                           ("high",      "High Voltage",      "#2A9D8F"),
                           ("low",       "Low Voltage",       "#F4A261")]:
        v = yf[col].values / 1e6
        ax4.bar(yf["year"], v, bottom=bot, label=lbl, color=clr, width=0.6)
        bot += v
    ax4.set_title("Electricity by Voltage Category  (million kWh)", fontsize=13)
    ax4.set_xlabel("Fiscal Year"); ax4.set_ylabel("Million kWh")
    ax4.legend(); ax4.set_xticks(yf["year"]); ax4.grid(alpha=0.3, axis="y")
    st.pyplot(fig4)

    # ── 1e: regulated vs market rate ──────────────────────────────────────
    st.subheader("Regulated vs Market-Rate Electricity (Low Voltage)")
    st.write("""
    This stackplot highlights a fundamental shift in Tokyo's electricity market:
    the share of **non-regulated (market-rate)** contracts has grown from 9% of
    low-voltage demand in FY2016 to over 68% in FY2022. This reflects Tokyo residents
    and small businesses actively switching to liberalised providers for cost savings
    and green-energy options.
    """)

    fig5, ax5 = plt.subplots(figsize=(10, 4))
    yr2 = annual_df.sort_values("year")
    ax5.stackplot(yr2["year"],
                  yr2["regulated"]/1e6, yr2["non_regulated"]/1e6,
                  labels=["Regulated Rate","Non-regulated (Market)"],
                  colors=["#2A9D8F","#E63946"], alpha=0.85)
    ax5.set_title("Regulated vs Market Rate — Low Voltage Demand", fontsize=13)
    ax5.set_xlabel("Fiscal Year"); ax5.set_ylabel("Million kWh")
    ax5.legend(loc="upper left"); ax5.set_xticks(yr2["year"]); ax5.grid(alpha=0.2)
    st.pyplot(fig5)

    # ── 1f: monthly heatmap ────────────────────────────────────────────────
    st.subheader("Monthly Electricity Demand Heatmap (2019–2022)")
    st.write("""
    The heatmap reveals strong seasonal patterns: demand peaks in **August** (air
    conditioning) and **January–February** (electric heating), with a trough in
    **May–June** (mild weather). This pattern is consistent across all years, though
    absolute levels dropped in 2020.
    """)

    pivot = monthly_df[monthly_df["year"].isin(YEARS)].pivot(
        index="year", columns="month", values="total")
    fig6, ax6 = plt.subplots(figsize=(12, 3))
    im = ax6.imshow(pivot.values / 1e6, aspect="auto", cmap="YlOrRd",
                    interpolation="nearest")
    ax6.set_xticks(range(pivot.shape[1]))
    ax6.set_xticklabels(["Apr","May","Jun","Jul","Aug","Sep",
                          "Oct","Nov","Dec","Jan","Feb","Mar"])
    ax6.set_yticks(range(len(YEARS))); ax6.set_yticklabels(YEARS)
    ax6.set_title("Monthly Electricity Demand  (million kWh)", fontsize=13)
    plt.colorbar(im, ax=ax6, label="M kWh", fraction=0.03)
    st.pyplot(fig6)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — CRIME ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif section == "2. Crime Analysis":
    st.title("Criminal Offense Analysis — Tokyo  (2019–2022)")
    st.markdown("""
    This section analyses criminal offenses reported to the Tokyo Metropolitan Police
    Department across all police stations and wards. The data covers 30+ crime categories
    from felonies to intellectual offenses, enabling trend analysis, geographic hotspot
    identification, and category-level breakdowns from a Smart City safety perspective.
    """)

    crime_df = load_crime()

    # ── 2a: total offenses per year ────────────────────────────────────────
    st.subheader("Total Reported Offenses per Year")
    st.write("""
    The bar chart shows the total number of criminal offenses reported to the police
    across Tokyo each year. The sharp decline from 2019 to 2021 is partly explained
    by COVID-19 restrictions, which reduced street activity and opportunities for
    certain crimes. The 2022 uptick reflects the gradual return to normal urban life.
    """)

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    yrs  = list(CRIME_TOTALS.keys())
    vals = list(CRIME_TOTALS.values())
    bars = ax1.bar(yrs, vals,
                   color=["#E63946","#F4A261","#2A9D8F","#457B9D"],
                   width=0.55, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x()+bar.get_width()/2, v+300,
                 f'{v:,}', ha="center", fontsize=10, fontweight="bold")
    ax1.set_title("Total Criminal Offenses — Tokyo-to", fontsize=13)
    ax1.set_ylabel("Number of offenses"); ax1.set_xticks(yrs); ax1.grid(alpha=0.3, axis="y")
    st.pyplot(fig1)

    st.write("""
    **Key observations:**
    - **2019**: 104,664 total offenses — the highest in the window, establishing the baseline.
    - **2020**: A drop of ~21% to 82,764. Lockdowns, business closures, and reduced
      foot traffic directly suppressed theft, fraud in physical settings, and street violence.
    - **2021**: A further decline to 75,288 — the lowest recorded — as restrictions persisted
      and remote work became normalized.
    - **2022**: A partial recovery to 78,475 as Tokyo reopened. Fraud and theft increased,
      consistent with patterns seen in other post-pandemic cities globally.
    """)

    # ── 2b: crime category breakdown ──────────────────────────────────────
    st.subheader("Crime Category Breakdown by Year")
    st.write("""
    The stacked bar chart decomposes total offenses into major categories, allowing us to
    see which crime types drive the overall trend and which are more structurally persistent.
    """)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    cat_colors = ["#E63946","#F4A261","#457B9D","#2A9D8F","#9B5DE5","#8D99AE","#B5838D"]
    bot = np.zeros(4)
    for cat, clr in zip(CRIME_CATS.keys(), cat_colors):
        vc = [CRIME_CATS[cat][y] for y in YEARS]
        ax2.bar(YEARS, vc, bottom=bot, label=cat, color=clr,
                width=0.6, edgecolor="white", linewidth=0.8)
        bot += np.array(vc)
    ax2.set_title("Crime by Category and Year", fontsize=13)
    ax2.set_ylabel("Number of offenses")
    ax2.legend(fontsize=8, loc="upper right"); ax2.set_xticks(YEARS)
    ax2.grid(alpha=0.3, axis="y")
    st.pyplot(fig2)

    st.write("""
    **Key observations:**
    - **Non-breaking-and-entering theft** is by far the largest category (45,000–70,000
      offenses/year), and also the most COVID-sensitive, dropping ~34% from 2019 to 2021.
    - **Vandalism** (器物損壊) is the second-largest category, relatively stable across years.
    - **Fraud** rose from 6,146 (2019) to 7,615 (2021) — the only major category to
      **increase** during COVID-19, driven by online scams and pandemic-related fraud.
    - **Bodily injury and violence** both declined with mobility restrictions and fell
      to the lowest levels in 2021.
    """)

    # ── 2c: % change vs 2019 ──────────────────────────────────────────────
    st.subheader("Percentage Change vs 2019 Baseline")
    st.write("""
    This chart normalises all changes against the 2019 baseline (0%), making it easy
    to compare the magnitude of change across years regardless of the absolute scale.
    """)

    base = CRIME_TOTALS[2019]
    pct  = [(CRIME_TOTALS[y] - base) / base * 100 for y in YEARS]
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.bar(YEARS, pct,
            color=["#8D99AE" if p >= 0 else "#2A9D8F" for p in pct],
            width=0.55, edgecolor="white")
    ax3.axhline(0, color="#1D3557", lw=1.5, ls="--")
    for y, p in zip(YEARS, pct):
        ax3.text(y, p + (0.5 if p >= 0 else -1.5),
                 f'{p:.1f}%', ha="center", fontsize=10)
    ax3.set_title("Total Offense Change vs 2019  (%)", fontsize=13)
    ax3.set_ylabel("% change"); ax3.set_xticks(YEARS); ax3.grid(alpha=0.3, axis="y")
    st.pyplot(fig3)

    # ── 2d: crime category line trends ────────────────────────────────────
    st.subheader("Crime Category Trends  (2019–2022)")
    st.write("""
    The line chart tracks individual crime categories, making it easy to compare
    trajectories. Categories that recover strongly by 2022 (e.g. theft) signal the
    return of pre-pandemic urban activity; categories that keep declining (e.g. B&E
    theft) may reflect lasting structural changes in behaviour.
    """)

    fig4, ax4 = plt.subplots(figsize=(12, 5))
    lc = ["Theft (non-B&E)","Fraud","Bodily Injury","Theft (B&E)","Violence","Vandalism"]
    lc_colors = ["#E63946","#457B9D","#F4A261","#2A9D8F","#9B5DE5","#8D99AE"]
    for cat, clr in zip(lc, lc_colors):
        ax4.plot(YEARS, [CRIME_CATS[cat][y] for y in YEARS],
                 color=clr, marker="o", lw=2.2, ms=7, label=cat)
    ax4.set_title("Crime Category Trends  (2019–2022)", fontsize=13)
    ax4.set_ylabel("Number of offenses")
    ax4.legend(fontsize=9, ncol=2); ax4.set_xticks(YEARS); ax4.grid(alpha=0.3)
    st.pyplot(fig4)

    # ── 2e: top police stations ────────────────────────────────────────────
    st.subheader("Top 10 Police Stations by Total Offenses  (2022)")
    st.write("""
    This horizontal bar chart identifies the police stations that recorded the highest
    number of offenses in 2022. These stations tend to cover high-density, high-footfall
    areas — major commercial districts, entertainment zones, or transit hubs — and
    represent the most resource-intensive areas for the Tokyo Metropolitan Police.
    """)

    top10 = (crime_df[crime_df["year"] == 2022]
             .groupby("station")["total"].sum()
             .sort_values(ascending=False).head(10))
    fig5, ax5 = plt.subplots(figsize=(9, 5))
    ax5.barh(range(10), top10.values[::-1], color="#E63946", alpha=0.85, edgecolor="white")
    ax5.set_yticks(range(10))
    ax5.set_yticklabels([s[:30] for s in top10.index[::-1]], fontsize=9)
    ax5.set_title("Top 10 Police Stations by Total Offenses  (2022)", fontsize=13)
    ax5.set_xlabel("Number of offenses"); ax5.grid(alpha=0.3, axis="x")
    st.pyplot(fig5)

    st.write("""
    **Key observations:**
    - The highest-ranking stations are predominantly located in central Tokyo wards
      (Shinjuku, Shibuya, Toshima, Minato), which host major nightlife and commercial
      districts with high concentrations of tourists and commuters.
    - Stations in these areas consistently appear at the top across all four years,
      indicating structural rather than transient crime concentration.
    - This geographic concentration supports a Smart City approach to targeted policing
      and prevention resource allocation.
    """)

    # ── 2f: crime distribution by district 2022 ───────────────────────────
    st.subheader("Total Offenses by District  (2022)")
    st.write("Which districts had the highest overall crime burden in 2022?")

    dist_2022 = (crime_df[crime_df["year"] == 2022]
                 .groupby("district")["total"].sum()
                 .sort_values(ascending=False).head(15))
    fig6, ax6 = plt.subplots(figsize=(10, 5))
    sns.barplot(x=dist_2022.values, y=dist_2022.index, palette="Reds_r", ax=ax6)
    ax6.set_title("Top 15 Districts by Total Offenses  (2022)", fontsize=13)
    ax6.set_xlabel("Total offenses"); ax6.grid(alpha=0.3, axis="x")
    st.pyplot(fig6)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — TRAFFIC & URBAN MOBILITY
# ══════════════════════════════════════════════════════════════════════════════
elif section == "3. Traffic & Urban Mobility":
    st.title("Traffic & Urban Mobility Analysis — Tokyo  (2019–2022)")
    st.markdown("""
    This section analyses vehicle counts at traffic monitoring sites across Tokyo's
    expressways and national roads. The dataset covers five vehicle categories —
    passenger cars, buses, small trucks, large trucks, and motorcycles — providing
    a detailed picture of traffic composition, congestion hotspots, and the long-term
    impact of the COVID-19 pandemic on urban mobility.
    """)

    traffic_df = load_traffic()

    # ── 3a: total vehicle count per year ──────────────────────────────────
    st.subheader("Total Vehicle Count per Year  (all monitoring sites)")
    st.write("""
    The area chart shows the aggregate daily vehicle count across all monitored road
    segments. Unlike crime or electricity, traffic volume remained remarkably stable
    through the pandemic — partly because expressways were less affected by urban
    lockdowns than local streets.
    """)

    yt = traffic_df.groupby("year")["total"].sum().reset_index().sort_values("year")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.fill_between(yt["year"], yt["total"] / 1e6, color="#2A9D8F", alpha=0.2)
    ax1.plot(yt["year"], yt["total"] / 1e6, "o-", color="#2A9D8F", lw=2.5, ms=9)
    for _, r in yt.iterrows():
        ax1.annotate(f'{r["total"]/1e6:.2f}M', (r["year"], r["total"]/1e6),
                     textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=9)
    ax1.set_title("Total Daily Vehicle Count  (million vehicles/day)", fontsize=13)
    ax1.set_xlabel("Year"); ax1.set_ylabel("Million vehicles/day")
    ax1.set_xticks(YEARS); ax1.grid(alpha=0.3)
    st.pyplot(fig1)

    st.write("""
    **Key observations:**
    - Total monitored volume stayed in the **2.77–2.86 M vehicles/day** range,
      showing the structural stability of Tokyo's expressway usage.
    - A minor dip in 2020–2021 corresponds to reduced commuting during COVID-19
      state-of-emergency periods and reduced tourism.
    - The 2022 rebound to the highest level in the window confirms the full recovery
      of Tokyo's road network utilisation.
    """)

    # ── 3b: vehicle type breakdown ────────────────────────────────────────
    st.subheader("Vehicle Composition by Year")
    st.write("""
    The stacked bar chart breaks down total traffic into vehicle categories. Passenger
    cars dominate (around 60%), while large trucks represent a significant share
    (~25%), reflecting Tokyo's role as a major logistics hub. Bus volumes are small
    relative to the total but important for modal-share analysis.
    """)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    bot = np.zeros(4)
    for vt, vl, vc in [("cars","Passenger Cars","#457B9D"),
                        ("buses","Buses","#F4A261"),
                        ("small_t","Small Trucks","#2A9D8F"),
                        ("large_t","Large Trucks","#E63946"),
                        ("motos","Motorcycles","#9B5DE5")]:
        by = [traffic_df[traffic_df["year"]==y][vt].sum()/1e6 for y in YEARS]
        ax2.bar(YEARS, by, bottom=bot, label=vl, color=vc, width=0.55, edgecolor="white")
        bot += np.array(by)
    ax2.set_title("Daily Traffic by Vehicle Type  (million vehicles/day)", fontsize=13)
    ax2.set_ylabel("Million vehicles/day")
    ax2.legend(fontsize=9); ax2.set_xticks(YEARS); ax2.grid(alpha=0.3, axis="y")
    st.pyplot(fig2)

    # ── 3c: top 10 busiest routes ─────────────────────────────────────────
    st.subheader("Top 10 Busiest Monitoring Sites  (2022)")
    st.write("""
    These are the highest-traffic road segments in Tokyo's expressway network. The
    Tomei Expressway near Yokohama consistently ranks highest, reflecting its role
    as the primary arterial connecting Tokyo to central Japan.
    """)

    top10 = (traffic_df[traffic_df["year"]==2022]
             .sort_values("total", ascending=False).head(10))
    labels_t = [f'{r["route"][:20]}\n{r["site"][:25]}' for _, r in top10.iterrows()]
    fig3, ax3 = plt.subplots(figsize=(13, 5))
    ax3.bar(range(10), top10["total"]/1000, color="#2A9D8F", edgecolor="white")
    ax3.set_xticks(range(10))
    ax3.set_xticklabels(labels_t, fontsize=7, rotation=20, ha="right")
    ax3.set_title("Top 10 Busiest Monitoring Sites  (2022)", fontsize=13)
    ax3.set_ylabel("Daily vehicles  (thousands)"); ax3.grid(alpha=0.3, axis="y")
    st.pyplot(fig3)

    # ── 3d: cars vs large trucks scatter ──────────────────────────────────
    st.subheader("Passenger Cars vs Large Trucks  (per site, all years)")
    st.write("""
    The scatter plot maps each monitoring site's passenger car count against its
    large truck count. Sites that cluster towards the upper-right are major mixed-use
    corridors; sites appearing mainly along the horizontal axis are commuter roads
    with minimal freight; sites along the vertical axis are primarily freight corridors.
    """)

    fig4, ax4 = plt.subplots(figsize=(9, 5))
    for yr, clr in zip(YEARS, ["#457B9D","#E63946","#2A9D8F","#F4A261"]):
        sub = traffic_df[traffic_df["year"] == yr].dropna(subset=["cars","large_t"])
        ax4.scatter(sub["cars"]/1000, sub["large_t"]/1000,
                    color=clr, alpha=0.7, s=60, label=str(yr))
    ax4.set_xlabel("Passenger Cars  (thousands/day)")
    ax4.set_ylabel("Large Trucks  (thousands/day)")
    ax4.set_title("Cars vs Large Trucks per Monitoring Site", fontsize=13)
    ax4.legend(fontsize=9); ax4.grid(alpha=0.3)
    st.pyplot(fig4)

    st.write("""
    **Key observations:**
    - Most sites fall along a positive correlation, confirming that high-traffic roads
      carry both high car and high truck volumes.
    - Outlier sites with a disproportionately high truck share correspond to ring-road
      and logistics expressway segments used for industrial freight.
    - No dramatic shift is visible between years, suggesting that the freight/passenger
      split on expressways is structurally stable and did not fundamentally change
      during COVID-19.
    """)

    # ── 3e: % change 2019→2022 ────────────────────────────────────────────
    st.subheader("Traffic Volume Change per Site  (2019 → 2022)")
    st.write("""
    This chart compares each monitoring site's 2022 volume against its 2019 baseline.
    Green bars indicate routes that recovered or grew; red bars show routes that still
    have not returned to pre-pandemic levels.
    """)

    t19 = traffic_df[traffic_df["year"]==2019].set_index("site")["total"]
    t22 = traffic_df[traffic_df["year"]==2022].set_index("site")["total"]
    common = t19.index.intersection(t22.index)
    pct_t  = ((t22[common] - t19[common]) / t19[common] * 100).sort_values()
    fig5, ax5 = plt.subplots(figsize=(10, max(4, len(pct_t)*0.35)))
    ax5.barh(range(len(pct_t)), pct_t.values,
             color=["#2A9D8F" if v > 0 else "#E63946" for v in pct_t])
    ax5.set_yticks(range(len(pct_t)))
    ax5.set_yticklabels([s[:28] for s in pct_t.index], fontsize=8)
    ax5.axvline(0, color="#1D3557", lw=1.5, ls="--")
    ax5.set_title("Traffic Volume Change  2019 → 2022  (%)", fontsize=13)
    ax5.set_xlabel("% change"); ax5.grid(alpha=0.3, axis="x")
    st.pyplot(fig5)

    st.write("""
    **Key observations:**
    - The majority of sites show **positive change**, confirming that traffic has
      not only recovered but exceeded 2019 levels at many locations.
    - Sites with negative change tend to be urban surface roads where remote working
      and reduced commuting have created a durable reduction in peak-hour traffic.
    - This asymmetry between expressway recovery (strong) and urban surface roads
      (weaker) is characteristic of post-pandemic mobility in large Japanese cities.
    """)

    # ── 3f: road type comparison ──────────────────────────────────────────
    st.subheader("Average Traffic by Road Type  (2022)")
    st.write("""
    Comparing national expressways to ordinary national roads reveals very different
    traffic profiles. Expressways carry far higher daily volumes but fewer vehicle types,
    while national roads serve more diverse traffic including buses and motorcycles.
    """)

    road_avg = (traffic_df[traffic_df["year"]==2022]
                .groupby("road")["total"].mean().sort_values(ascending=False))
    fig6, ax6 = plt.subplots(figsize=(9, 4))
    sns.barplot(x=road_avg.values/1000, y=road_avg.index, palette="Blues_r", ax=ax6)
    ax6.set_title("Average Daily Vehicles by Road Type  (2022, thousands)", fontsize=13)
    ax6.set_xlabel("Avg daily vehicles  (thousands)"); ax6.grid(alpha=0.3, axis="x")
    st.pyplot(fig6)

