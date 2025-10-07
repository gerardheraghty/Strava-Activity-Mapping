# app.py
#may need to run: pip install streamlit folium streamlit-folium pandas pyarrow
# app.py
from pathlib import Path
import pandas as pd
import streamlit as st

import folium
from branca.element import Figure
from streamlit_folium import st_folium

# ---------- Paths ----------
DATA_DIR = Path("data/curated")
PATHS_DIR = DATA_DIR / "paths"

# ---------- Page setup ----------
st.set_page_config(page_title="Gerards Strava Records", layout="wide")
st.title("Gerards Strava Records")

# ---------- Helpers (perf + safety) ----------
@st.cache_data
def load_activities():
    """Load activities (cached)."""
    df = pd.read_parquet(DATA_DIR / "activities.parquet")
    # Normalize columns/types
    if "sport_type" not in df.columns and "type" in df.columns:
        df["sport_type"] = df["type"]
    if "distance_km_reported" not in df.columns and "distance_m_reported" in df.columns:
        df["distance_km_reported"] = df["distance_m_reported"] / 1000
    if "start_date_utc" in df.columns:
        df["start_date_utc"] = pd.to_datetime(df["start_date_utc"], errors="coerce")
    else:
        df["start_date_utc"] = pd.NaT
    return df

def _thin_points(pts, max_points=2000):
    """Lightly downsample a list of [lat, lon] points to speed up rendering."""
    n = len(pts)
    if n <= max_points:
        return pts
    step = max(1, n // max_points)
    return pts[::step]

@st.cache_data
def load_paths_for_ids(activity_ids_tuple):
    """
    Load path points for a tuple of activity IDs and return a dict: id -> [[lat, lon], ...]
    Cached based on the set of IDs passed in.
    """
    out = {}
    for aid in activity_ids_tuple:
        fp = PATHS_DIR / f"ACTIVITY_{aid}.csv"
        if not fp.exists():
            continue
        try:
            dfp = pd.read_csv(fp, usecols=["lat", "lon"]).dropna()
        except Exception:
            continue
        if dfp.empty:
            continue
        pts = dfp[["lat","lon"]].values.tolist()
        out[aid] = _thin_points(pts, max_points=2000)
    return out

def compute_center_from_paths(paths_dict, max_tracks=50):
    """Compute a reasonable map center from up to N tracks."""
    if not paths_dict:
        return (20.0, 0.0)
    lat_sum = lon_sum = count = 0
    for i, (aid, pts) in enumerate(paths_dict.items()):
        if i >= max_tracks:
            break
        if not pts:
            continue
        # average of first up-to 2000 points for each track
        clip = pts[:2000]
        lat_mean = sum(p[0] for p in clip) / len(clip)
        lon_mean = sum(p[1] for p in clip) / len(clip)
        lat_sum += lat_mean
        lon_sum += lon_mean
        count += 1
    if count == 0:
        return (20.0, 0.0)
    return (lat_sum / count, lon_sum / count)

# ---------- Load data ----------
df_all = load_activities()

# ---------- Global summary (not affected by dropdown) ----------
st.subheader("Distance covered per activity (all time)")
summary_activity = (
    df_all.groupby("sport_type", dropna=True)["distance_km_reported"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)
st.bar_chart(summary_activity.set_index("sport_type"), use_container_width=True)

st.caption("The chart above is global. The selector below filters **only** the map.")
st.divider()

# ---------- Map controls ----------
types = sorted([t for t in df_all["sport_type"].dropna().unique().tolist()])
# prepend an All option
options = ["All"] + types
selection = st.multiselect(
    "Activities to show on the map",
    options=options,
    default=["All"],
)

# Resolve selection -> sel_types
if ("All" in selection) or (not [x for x in selection if x != "All"]):
    sel_types = types  # all types
    selection_label = "All"
else:
    sel_types = [x for x in selection if x != "All"]
    selection_label = ", ".join(sel_types)

st.info(f"Map is showing: {selection_label}")

# ---------- Filter data for the map ----------
df_map = df_all[df_all["sport_type"].isin(sel_types)]
# load paths for selected activity IDs (cached by tuple)
activity_ids_tuple = tuple(df_map["activity_id"].tolist())
paths_dict = load_paths_for_ids(activity_ids_tuple)

# ---------- Build the map ----------
fig = Figure(width="100%", height="740px")
center = compute_center_from_paths(paths_dict)
m = folium.Map(location=center, zoom_start=2, tiles="cartodbpositron")
fig.add_child(m)

# draw orange routes with fixed low opacity so overlaps get darker
FIXED_OPACITY = 0.25

for _, row in df_map.iterrows():
    aid = row["activity_id"]
    pts = paths_dict.get(aid, [])
    if not pts:
        continue

    date_val = row.get("start_date_utc")
    date_str = (date_val.date() if hasattr(date_val, "date") and pd.notna(date_val)
                else str(date_val)[:10])
    dist_km = row.get("distance_km_reported", None)
    dist_str = f"{dist_km:.1f} km" if pd.notna(dist_km) else "— km"
    tooltip = f"{row.get('name', '')} • {row.get('sport_type', '')} • {date_str} • {dist_str}"

    folium.PolyLine(
        pts,
        weight=3,
        opacity=FIXED_OPACITY,
        color="orange",
        tooltip=tooltip
    ).add_to(m)

st_folium(m, width=None, height=740)

##to run: streamlit run app.py