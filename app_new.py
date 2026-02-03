import os
import datetime as dt
import math
from pathlib import Path
import re

import streamlit as st
import joblib
import pandas as pd
import requests
from dotenv import load_dotenv

from streamlit_searchbox import st_searchbox

# ---------------------------
# Streamlit config MUST be first Streamlit call
# ---------------------------
st.set_page_config(page_title="Wycena transportu", layout="centered")

# ---------------------------
# Env
# ---------------------------
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# ---------------------------
# UI cosmetics
# ---------------------------
st.markdown(
    """
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 70%;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Kalkulator wycen")

# ---------------------------
# Config helpers
# ---------------------------
def get_secret(name: str, default: str | None = None):
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name, default)

MODEL_PATH = get_secret("MODEL_PATH", "model_wycena_raw.joblib")
API_KEY = get_secret("GOOGLE_MAPS_API_KEY")

APPS_SCRIPT_URL = get_secret("APPS_SCRIPT_URL")
APPS_SCRIPT_TOKEN = get_secret("APPS_SCRIPT_TOKEN", "")
DB_HISTORY_SHEET = get_secret("DB_HISTORY_SHEET", "history")
DB_CORRECTIONS_SHEET = get_secret("DB_CORRECTIONS_SHEET", "corrections")

HUBS = [
    ("Poznań", 52.4056786, 16.9312766, 1),
    ("Łódź", 51.7592924, 19.4558778, 1),
    ("Warszawa", 52.2296756, 21.0122287, 1),
    ("Katowice", 50.2648919, 19.0237815, 1),
    ("Gdańsk", 54.3520252, 18.6466384, 1),
    ("Kraków", 50.0646501, 19.9449799, 1),
    ("Nowy Sącz", 49.6174535, 20.7153326, 0),
    ("Lublin", 51.2509458, 22.5747275, 0),
    ("Wrocław", 51.1092948, 17.0386019, 0),
    ("Opole", 50.6683223, 17.9230651, 0),
    ("Radom", 51.4027236, 21.1471333, 0),
    ("Olsztyn", 53.7784220, 20.4801193, 0),
    ("Szczecin", 53.4285438, 14.5528116, -1),
    ("Suwałki", 54.1115218, 22.9307881, -1),
    ("Rzeszów", 50.0411867, 21.9991196, -1),
    ("Wałbrzych", 50.7840092, 16.2843553, -1),
    ("Zielona góra", 51.9356214, 15.5061862, -1),
    ("Jelenia góra", 50.9034749, 15.7316540, -1),
    ("Berlin", 52.5200066, 13.4049540, -2),
    ("Hamburg", 53.5488282, 9.9871703, -2),
    ("Monachium", 48.1351253, 11.5819806, -2),
    ("Amsterdam", 52.3675734, 4.9041389, -2),
    ("Dortmund", 51.5135872, 7.4652981, -2),
    ("Brema", 53.0792962, 8.8016937, -2),
]

OPTIONS_PLUS = [
    ("Większa naczepa", "wieksza_naczepa"),
    ("Winda", "winda"),
    ("Express", "express"),
]

TRANSPORT_UI_TO_MODEL = {
    "Naczepa": "naczepa",
    "Solo": "solo",
    "Bus": "bus",
}

COLUMN_MAP = {
    "run_id": "ID wyceny",
    "timestamp": "Data zapisu",
    "origin": "Skąd",
    "destination": "Dokąd",
    "stops": "Przystanki",
    "typ_transportu": "Typ transportu",
    "dlugosc_trasy_km": "Długość trasy (km)",
    "dest_score": "Ocena kierunku",
    "hub_name": "Najbliższy hub",
    "base_price": "Cena bazowa",
    "surcharge": "Dopłaty",
    "final_price": "Cena końcowa",
    "price_km": "cena/km",
}

QUICK_LOADS = [
    "Bogdanowo, 64-600",
    "Czarnków, 64-700",
    "Gromadka, 59-706",
]

# ---------------------------
# Helpers: loads + geo
# ---------------------------
def search_loads(searchterm: str):
    term = (searchterm or "").strip().lower()
    if not term:
        return QUICK_LOADS
    return [x for x in QUICK_LOADS if term in x.lower()]

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def nearest_hub_and_score(dest_lat: float, dest_lon: float) -> tuple[str, float, float]:
    if not HUBS:
        raise ValueError("Brak zdefiniowanych hubów.")
    best = None  # (name, km, score)
    for name, lat, lon, score in HUBS:
        d = haversine_km(dest_lat, dest_lon, lat, lon)
        if best is None or d < best[1]:
            best = (name, d, score)
    return best[0], float(best[1]), float(best[2])

def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    brng = math.degrees(math.atan2(y, x))
    return (brng + 360.0) % 360.0

def bearing_to_compass_16(brng: float) -> str:
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = int((brng + 11.25) // 22.5) % 16
    return dirs[idx]

def round_up_to_10(x: float) -> float:
    return math.ceil(x / 10) * 10

def norm_place_no_zip(s: str) -> str:
    s = (s or "").strip().lower()
    if not s:
        return ""
    return s.split(",", 1)[0].strip()

# ZIP helpers (full + prefix fallback)
ZIP_FULL_RE = re.compile(r"\b(\d{2}-\d{3})\b")
ZIP_PREFIX_RE = re.compile(r"\b(\d{2})\b")

def extract_zip_full(s: str) -> str:
    m = ZIP_FULL_RE.search(str(s or ""))
    return m.group(1) if m else ""

def extract_zip_prefix(s: str) -> str:
    full = extract_zip_full(s)
    if full:
        return full[:2]
    m = ZIP_PREFIX_RE.search(str(s or ""))
    return m.group(1) if m else ""

def to_zip_full_or_blank(x) -> str:
    s = str(x or "").strip()
    m = ZIP_FULL_RE.search(s)
    return m.group(1) if m else ""

def to_zip_prefix_or_blank(x) -> str:
    s = str(x or "").strip()
    m = ZIP_FULL_RE.search(s)
    if m:
        return m.group(1)[:2]
    m2 = ZIP_PREFIX_RE.search(s)
    return m2.group(1) if m2 else ""

# ---------------------------
# Google API
# ---------------------------
class GoogleAPIError(RuntimeError):
    pass

def _assert_key():
    if not API_KEY:
        raise GoogleAPIError("Brak GOOGLE_MAPS_API_KEY. Dodaj do .env / secrets.")

@st.cache_data(ttl=24 * 3600)
def geocode(place: str) -> tuple[float, float, str]:
    _assert_key()
    g_url = "https://maps.googleapis.com/maps/api/geocode/json"
    r = requests.get(g_url, params={"address": place, "key": API_KEY}, timeout=20)
    r.raise_for_status()
    data = r.json()

    if data.get("status") != "OK":
        raise GoogleAPIError(f"Geocode error: {data.get('status')} {data.get('error_message','')}".strip())

    res = data["results"][0]
    loc = res["geometry"]["location"]
    return float(loc["lat"]), float(loc["lng"]), res.get("formatted_address", place)

@st.cache_data(ttl=6 * 3600)
def route_km(origin: str, destination: str, mode: str, waypoints: tuple[str, ...]) -> float:
    _assert_key()
    g_url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {"origin": origin, "destination": destination, "mode": mode, "units": "metric", "key": API_KEY}
    if waypoints:
        params["waypoints"] = "|".join(waypoints)

    r = requests.get(g_url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    if data.get("status") != "OK":
        raise GoogleAPIError(f"Directions error: {data.get('status')} {data.get('error_message','')}".strip())

    if not data.get("routes"):
        raise GoogleAPIError("Nie znaleziono trasy dla podanych punktów.")

    legs = data["routes"][0]["legs"]
    meters = sum(leg["distance"]["value"] for leg in legs)
    return float(meters) / 1000.0

# ---------------------------
# Apps Script DB
# ---------------------------
def db_append_row(sheet: str, row: list):
    if not APPS_SCRIPT_URL:
        return

    payload = {"row": row}
    r = requests.post(
        APPS_SCRIPT_URL,
        params={"sheet": sheet, "token": APPS_SCRIPT_TOKEN},
        json=payload,
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "ok":
        raise RuntimeError(f"Apps Script DB error: {data}")

@st.cache_data(ttl=60)
def db_read_df(sheet: str, limit: int = 300) -> tuple[pd.DataFrame, str | None]:
    if not APPS_SCRIPT_URL:
        return pd.DataFrame(), None

    r = requests.get(
        APPS_SCRIPT_URL,
        params={"sheet": sheet, "token": APPS_SCRIPT_TOKEN, "limit": str(limit)},
        timeout=15,
    )
    r.raise_for_status()

    data = r.json()
    if data.get("status") != "ok":
        return pd.DataFrame(), f"Apps Script error payload: {data}"

    header = data.get("header", [])
    rows = data.get("rows", [])
    if not header or not rows:
        return pd.DataFrame(), None

    df = pd.DataFrame(rows, columns=header)

    reverse_column_map = {v: k for k, v in COLUMN_MAP.items()}
    df = df.rename(columns=reverse_column_map)

    return df, None

# ---------------------------
# Model
# ---------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ---------------------------
# Session state
# ---------------------------
if "addresses" not in st.session_state:
    st.session_state.addresses = ["", ""]  # start + koniec

if "selected_opts" not in st.session_state:
    st.session_state.selected_opts = []

if "transport_ui" not in st.session_state:
    st.session_state.transport_ui = "Naczepa"

# ---------------------------
# UI: Trasa
# ---------------------------
st.subheader("Trasa")

top = st.columns([1, 1, 6])
with top[0]:
    if st.button("➕ Dodaj punkt"):
        st.session_state.addresses.append("")
        st.rerun()


last = len(st.session_state.addresses) - 1

for i, addr in enumerate(st.session_state.addresses):
    cols = st.columns([6, 1])

    label = "Punkt startowy" if i == 0 else ("Punkt końcowy" if i == last else f"Stop {i}")
    placeholder = "Załadunek" if i == 0 else ("Rozładunek" if i == last else "Przystanek pośredni")

    with cols[0]:
        if i == 0:
            def reset_origin():
                st.session_state.pop("origin_searchbox", None)

            try:
                picked = st_searchbox(
                    search_loads,
                    key="origin_searchbox",
                    label=label,
                    placeholder=placeholder,
                    clear_on_submit=False,
                    default_options=QUICK_LOADS,
                    reset_function=reset_origin,
                    edit_after_submit="option",
                )
            except IndexError:
                reset_origin()
                st.rerun()

            state = st.session_state.get("origin_searchbox", {})
            typed = ""
            if isinstance(state, dict):
                typed = (state.get("searchterm") or state.get("search") or state.get("value") or "").strip()

            st.session_state.addresses[0] = (picked or typed or "").strip()
        else:
            st.session_state.addresses[i] = st.text_input(
                label,
                value=addr,
                placeholder=placeholder,
                key=f"address_{i}",
            )

    with cols[1]:
        if i > 0 and st.button("❌", key=f"remove_{i}"):
            st.session_state.addresses.pop(i)
            st.rerun()

st.session_state.transport_ui = st.selectbox(
    "Typ transportu",
    list(TRANSPORT_UI_TO_MODEL.keys()),
    index=list(TRANSPORT_UI_TO_MODEL.keys()).index(st.session_state.transport_ui),
)

with st.expander("Zaawansowane (opcjonalne)"):
    st.markdown("**Opcje + (każda +100 zł):**")
    selected = []
    for opt_label, opt_key in OPTIONS_PLUS:
        if st.checkbox(opt_label, value=(opt_key in st.session_state.selected_opts), key=f"opt_{opt_key}"):
            selected.append(opt_key)
    st.session_state.selected_opts = selected

submitted = st.button("Policz ✅")

# ---------------------------
# Wynik
# ---------------------------
if submitted:
    run_id = dt.datetime.now().strftime("%Y-%m-%d")

    with st.spinner("Liczenie trasy i wyceny..."):
        try:
            addresses = [a.strip() for a in st.session_state.addresses if a.strip()]
            if len(addresses) < 2:
                st.error("Podaj co najmniej punkt startowy i końcowy.")
                st.stop()

            origin = addresses[0].strip()
            if origin:
                matches = search_loads(origin)
                if matches and origin.lower() not in [m.lower() for m in matches]:
                    origin = matches[0]
                    st.session_state.addresses[0] = origin

            destination = addresses[-1]
            stops = addresses[1:-1]
            stops_count = len(stops)

            options_count = len(st.session_state.selected_opts)

            model_transport = TRANSPORT_UI_TO_MODEL[st.session_state.transport_ui]
            google_mode = "driving"

            km = route_km(origin, destination, google_mode, tuple(stops))

            o_lat, o_lon, o_fmt = geocode(origin)
            d_lat, d_lon, d_fmt = geocode(destination)

            brng = bearing_deg(o_lat, o_lon, d_lat, d_lon)
            direction = bearing_to_compass_16(brng)

            hub_name, hub_km, dest_score = nearest_hub_and_score(d_lat, d_lon)
            dest_score = float(dest_score)

            month = dt.datetime.now().month
            X = pd.DataFrame([{
                "miesiąc": month,
                "typ_transportu": model_transport,
                "długosc_trasy": km,
                "dest_score": dest_score,
            }])

            base_price = float(model.predict(X)[0])

            surcharge = 100.0 * stops_count + 100.0 * options_count
            raw_price = base_price + surcharge
            final_price = round_up_to_10(raw_price)
            price_km = round(final_price / km, 1)

            st.success("Gotowe ✅")
            st.metric("Wycena końcowa", f"{final_price:,.2f} zł")
            st.caption(f"Baza z modelu: {base_price:,.2f} zł | Dopłaty: {surcharge:,.2f} zł")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Długość trasy", f"{km:.2f} km")
            c2.metric("Najbliższy hub", f"{hub_name}")
            c3.metric("dest_score", f"{int(dest_score)}")
            c4.metric("Kierunek", direction)
            c5.metric("cena/KM", price_km)

            # ---- Moduł: podobne transporty
            st.subheader("Podobne transporty w przeszłości")

            hist, hist_err = db_read_df(DB_HISTORY_SHEET)
            if hist_err:
                st.warning(hist_err)

            if hist.empty:
                st.info("Brak danych historycznych w Sheets.")
            else:
                # normalizacja nazw miejscowości (bez kodu)
                for col in ["origin", "destination"]:
                    if col in hist.columns:
                        hist[col] = hist[col].astype(str).apply(norm_place_no_zip)

                cur_origin = norm_place_no_zip(origin)
                cur_destination = norm_place_no_zip(destination)

                # --- NOWE: przygotuj kolumny ZIP (full/pref) do filtrowania ---
                if "origin_zip" in hist.columns:
                    hist["origin_zip_full"] = hist["origin_zip"].apply(to_zip_full_or_blank)
                    hist["origin_zip_pref"] = hist["origin_zip"].apply(to_zip_prefix_or_blank)
                else:
                    hist["origin_zip_full"] = hist["origin"].astype(str).apply(extract_zip_full)
                    hist["origin_zip_pref"] = hist["origin"].astype(str).apply(extract_zip_prefix)

                if "destination_zip" in hist.columns:
                    hist["destination_zip_full"] = hist["destination_zip"].apply(to_zip_full_or_blank)
                    hist["destination_zip_pref"] = hist["destination_zip"].apply(to_zip_prefix_or_blank)
                else:
                    hist["destination_zip_full"] = hist["destination"].astype(str).apply(extract_zip_full)
                    hist["destination_zip_pref"] = hist["destination"].astype(str).apply(extract_zip_prefix)

                cur_o_full = extract_zip_full(origin)
                cur_d_full = extract_zip_full(destination)
                cur_o_pref = extract_zip_prefix(origin)
                cur_d_pref = extract_zip_prefix(destination)

                # 1) dokładnie ta sama relacja A → B po nazwie
                df = hist[(hist["origin"] == cur_origin) & (hist["destination"] == cur_destination)]
                if "run_id" in df.columns:
                    df = df[df["run_id"] != run_id]

                # 1b) fallback: full zip → full zip
                if df.empty and cur_o_full and cur_d_full:
                    df = hist[
                        (hist["origin_zip_full"] == cur_o_full) &
                        (hist["destination_zip_full"] == cur_d_full)
                    ]

                # 1c) fallback: prefix → prefix (np. "57")
                if df.empty and cur_o_pref and cur_d_pref:
                    df = hist[
                        (hist["origin_zip_pref"] == cur_o_pref) &
                        (hist["destination_zip_pref"] == cur_d_pref)
                    ]

                # 2) jeśli brak — ten sam destination (miasto → full zip → pref)
                if df.empty:
                    df = hist[hist["destination"] == cur_destination]

                    if df.empty and cur_d_full:
                        df = hist[hist["destination_zip_full"] == cur_d_full]

                    if df.empty and cur_d_pref:
                        df = hist[hist["destination_zip_pref"] == cur_d_pref]

                # 3) jeśli nadal brak — podobny dystans + dest_score
                if df.empty:
                    tol_km = 50.0
                    if "dlugosc_trasy_km" in hist.columns and "dest_score" in hist.columns:
                        df = hist[
                            (hist["dest_score"] == int(dest_score)) &
                            (hist["dlugosc_trasy_km"].between(km - tol_km, km + tol_km))
                        ]

                if "timestamp" in df.columns:
                    df = df.sort_values("timestamp", ascending=False)

                show_cols = [
                    c for c in [
                        "timestamp",
                        "origin",
                        "destination",
                        "typ_transportu",
                        "dlugosc_trasy_km",
                        "dest_score",
                        "final_price",
                        "price_km",
                    ] if c in df.columns
                ]

                if df.empty:
                    st.info("Nie znaleziono podobnych transportów.")
                else:
                    st.dataframe(df[show_cols].rename(columns=COLUMN_MAP))

                    # zapis do history (opcjonalne)
                    ts = dt.datetime.now().strftime("%Y-%m-%d")
                    if APPS_SCRIPT_URL:
                        row = [
                            run_id,
                            ts,
                            origin.strip().title(),
                            destination.strip().title(),
                            " | ".join(stops).title(),
                            model_transport,
                            km,
                            int(dest_score),
                            hub_name,
                            base_price,
                            surcharge,
                            final_price,
                            price_km,
                        ]
                        db_append_row(DB_HISTORY_SHEET, row)
                        db_read_df.clear()

            # ---- Moduł: popraw cenę
            st.subheader("Popraw wycenę")

            corrected = st.number_input(
                "Cena poprawiona (zł)",
                min_value=0.0,
                step=50.0,
                value=float(final_price),
            )

            if st.button("Zapisz poprawkę"):
                ts = dt.datetime.now().strftime("%Y-%m-%d")
                delta = round(float(corrected) - float(final_price), 2)
                st.write(f"Delta (poprawiona - wyliczona): **{delta:,.2f} zł**")

                corr_row = [
                    ts,
                    origin,
                    destination,
                    km,
                    int(dest_score),
                    final_price,
                    float(corrected),
                    delta,
                ]

                db_append_row(DB_CORRECTIONS_SHEET, corr_row)
                st.success("Poprawka zapisana ✅")

            with st.expander("Szczegóły (debug)"):
                st.write("Punkty trasy:", addresses)
                st.write("Origin (Google):", o_fmt)
                st.write("Destination (Google):", d_fmt)
                st.write("Stopy:", stops)
                st.write("Opcje +:", st.session_state.selected_opts)
                st.write("Bearing (deg):", brng)
                st.write("Feature’y do modelu:", X)

        except GoogleAPIError as e:
            st.error(f"Google API: {e}")
        except Exception as e:
            st.exception(e)
