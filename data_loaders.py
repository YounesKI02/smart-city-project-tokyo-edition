"""Shared data loaders for Tokyo Smart City project."""
import os, pandas as pd, numpy as np

DATA_DIR = os.path.dirname(__file__)
YEARS = [2019, 2020, 2021, 2022]

def load_electricity_annual():
    rows = []
    for tag in ["19","20","21","22"]:
        df = pd.read_csv(f"{DATA_DIR}/Electricity data/Electricity Demand {tag}.csv",
                         encoding="utf-8-sig", header=0)
        df.columns = (["jp_year","yr","jp_month","mth","tot","xhigh","high","low","reg","nonreg"]
                      + list(df.columns[10:]))
        annual = df[df["jp_month"].astype(str).str.strip()=="総数"].copy()
        annual["yr"] = pd.to_numeric(annual["yr"], errors="coerce")
        for _, r in annual.dropna(subset=["yr"]).iterrows():
            rows.append({
                "year":          int(r["yr"]),
                "total":         pd.to_numeric(r["tot"],    errors="coerce"),
                "extra_high":    pd.to_numeric(r["xhigh"],  errors="coerce"),
                "high":          pd.to_numeric(r["high"],   errors="coerce"),
                "low":           pd.to_numeric(r["low"],    errors="coerce"),
                "regulated":     pd.to_numeric(r["reg"],    errors="coerce"),
                "non_regulated": pd.to_numeric(r["nonreg"], errors="coerce"),
            })
    return (pd.DataFrame(rows).drop_duplicates("year")
              .sort_values("year").reset_index(drop=True))

def load_electricity_monthly():
    rows = []
    for tag in ["19","20","21","22"]:
        df = pd.read_csv(f"{DATA_DIR}/Electricity data/Electricity Demand {tag}.csv",
                         encoding="utf-8-sig", header=0)
        df.columns = (["jp_year","yr","jp_month","mth","tot","xhigh","high","low","reg","nonreg"]
                      + list(df.columns[10:]))
        df["yr"] = pd.to_numeric(df["yr"], errors="coerce")
        monthly = df[df["jp_month"].astype(str).str.strip().str.match(r"^\d+$", na=False)].dropna(subset=["yr"])
        for _, r in monthly.iterrows():
            rows.append({"year": int(r["yr"]), "month": int(r["jp_month"]), 
                         "total": pd.to_numeric(r["tot"], errors="coerce")})
    return (pd.DataFrame(rows).drop_duplicates(["year","month"])
              .sort_values(["year","month"]).reset_index(drop=True))

def load_air_quality():
    df = pd.read_csv(f"{DATA_DIR}/Air quality/Air Pollutant Measurement Averages 22.csv",
                     encoding="utf-8-sig", header=0)
    df.columns = (["jp_year","yr","jp_poll","poll","tavg","kavg"]
                  + list(df.columns[6:]))
    df["yr"]   = pd.to_numeric(df["yr"],   errors="coerce")
    df["tavg"] = pd.to_numeric(df["tavg"], errors="coerce")
    df["kavg"] = pd.to_numeric(df["kavg"], errors="coerce")
    pmap = {"Nitrogen dioxide":"NO2","Suspended particulate":"SPM",
            "Photochemical oxidant":"Ox","Sulfur dioxide":"SO2",
            "Particulate matter":"PM2.5"}
    def cp(n):
        for k,v in pmap.items():
            if k.lower() in str(n).lower(): return v
        return str(n).strip()
    df["pollutant"] = df["poll"].apply(cp)
    return (df.rename(columns={"yr":"year","tavg":"tokyo_avg","kavg":"ku_avg"})
              [["year","pollutant","tokyo_avg","ku_avg"]].dropna(subset=["year"]))

def load_traffic():
    rows = []
    for yr in YEARS:
        tag = str(yr)[2:]
        df = pd.read_csv(f"{DATA_DIR}/Traffic data/Traffic Volume by Traffic Monitoring Site {tag}.csv",
                         encoding="utf-8-sig", header=0)
        df.columns = (["road_jp","road","dist_jp","dist","item",
                        "route_jp","route","site_jp","site",
                        "total","cars","buses","small_t","large_t",
                        "peds","bikes","motos"])
        df["year"] = yr
        for c in ["total","cars","buses","small_t","large_t","motos"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        rows.append(df.dropna(subset=["total"]))
    return pd.concat(rows, ignore_index=True)

def load_crime():
    rows = []
    for yr, tag in [(2019,"19"),(2020,"20"),(2021,"21"),(2022,"22")]:
        df = pd.read_csv(
            f"{DATA_DIR}/Crime data/Criminal Offenses Known to the Police by Type of Crime and Police Station {tag}.csv",
            encoding="utf-8-sig", header=0)
        nc = len(df.columns)
        df.columns = (["tier","code","dist_jp","district","item","sta_jp","station","total"]
                      + [f"c{i}" for i in range(nc-8)])
        df = df[df["tier"].astype(str).str.strip()=="2"].copy()
        df["year"]  = yr
        df["total"] = pd.to_numeric(df["total"], errors="coerce")
        for cat, col in [("homicide","c0"),("violence","c9"),("bodily_injury","c10"),
                          ("theft_be","c13"),("theft_non_be","c14"),
                          ("fraud","c15"),("vandalism","c25")]:
            df[cat] = pd.to_numeric(df[col], errors="coerce").fillna(0) if col in df.columns else 0
        rows.append(df[["year","district","station","total",
                         "homicide","violence","bodily_injury",
                         "theft_be","theft_non_be","fraud","vandalism"]])
    return pd.concat(rows, ignore_index=True).dropna(subset=["total"])

CRIME_TOTALS = {2019:104664, 2020:82764, 2021:75288, 2022:78475}
CRIME_CATS = {
    "Theft (non-B&E)": {2019:69438,2020:52077,2021:45966,2022:49120},
    "Fraud":           {2019:6146, 2020:5772, 2021:7615, 2022:6945},
    "Bodily Injury":   {2019:4221, 2020:3571, 2021:3302, 2022:3719},
    "Theft (B&E)":     {2019:4550, 2020:3149, 2021:2254, 2022:2111},
    "Violence":        {2019:2682, 2020:2203, 2021:2209, 2022:2445},
    "Vandalism":       {2019:10089,2020:8927, 2021:7695, 2022:7352},
    "Other":           {2019:7538, 2020:7065, 2021:6247, 2022:6783},
}
