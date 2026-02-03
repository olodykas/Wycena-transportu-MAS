import os
import datetime as dt
import math
import streamlit as st
import joblib
import pandas as pd
import requests
from dotenv import load_dotenv
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# ---------- Streamlit setup ----------
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 70%;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Wycena transportu",
                   layout="centered",
)

st.title("Kalkulator wycen")


# ---------------------------
# Config
# ---------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "model_wycena_raw.joblib")
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

APPS_SCRIPT_URL = os.getenv("APPS_SCRIPT_URL")
APPS_SCRIPT_TOKEN = os.getenv("APPS_SCRIPT_TOKEN", "")
DB_HISTORY_SHEET = os.getenv("DB_HISTORY_SHEET", "history")
DB_CORRECTIONS_SHEET = os.getenv("DB_CORRECTIONS_SHEET", "corrections")

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
    #"direction": "Kierunek",
    "dlugosc_trasy_km": "Długość trasy (km)",
    "dest_score": "Ocena kierunku",
    "hub_name": "Najbliższy hub",
    "base_price": "Cena bazowa",
    "surcharge": "Dopłaty",
    "final_price": "Cena końcowa",
    "price_km": "cena/km"
}

QUICK_LOADS = [
    "Bogdanowo, 64-600",
    "Czarnków, 64-700",
    "Gromadka, 59-706",
]
OTHER_LABEL = "Inny adres…"

# ---------------------------
# Helpers: haversine + hub score + kierunek
# ---------------------------
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
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    idx = int((brng + 11.25) // 22.5) % 16
    return dirs[idx]

def round_up_to_10(x: float) -> float:
    return math.ceil(x / 10) * 10

def norm_city(s: str) -> str:
    return s.strip().lower()

# ---------------------------
# Helpers: Google API
# ---------------------------
class GoogleAPIError(RuntimeError):
    pass

def _assert_key():
    if not API_KEY:
        raise GoogleAPIError("Brak GOOGLE_MAPS_API_KEY. Dodaj do .env / secrets.")

@st.cache_data(ttl=24*3600)
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

origin = st.session_state.addresses[0].strip()
if not origin:
    st.error("Uzupełnij pole Załadunek.")
    st.stop()

@st.cache_data(ttl=6*3600)
def route_km(origin: str, destination: str, mode: str, waypoints: tuple[str, ...]) -> float:
    _assert_key()
    g_url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": origin,
        "destination": destination,
        "mode": mode,
        "units": "metric",
        "key": API_KEY,
    }
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

# zapis do google sheets
def db_append_row(sheet: str, row: list):
    """Zapis jednego wiersza do Google Sheets przez Apps Script."""
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
def db_read_df(sheet: str, limit: int = 300) -> pd.DataFrame:
    """Odczyt danych z Sheets przez Apps Script (ostatnie N wierszy)."""
    if not APPS_SCRIPT_URL:
        return pd.DataFrame()

    r = requests.get(
        APPS_SCRIPT_URL,
        params={"sheet": sheet, "token": APPS_SCRIPT_TOKEN, "limit": str(limit)},
        timeout=15,
    )
    r.raise_for_status()
    #data = r.json()
    #if data.get("status") != "ok":
        #return pd.DataFrame()
    data = r.json()
    if data.get("status") != "ok":
        st.write("Apps Script error payload:", data)
        return pd.DataFrame()

    header = data.get("header", [])
    rows = data.get("rows", [])
    if not header or not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=header)

    # 👇 NOWE: mapowanie nazw z ludzkich → techniczne
    REVERSE_COLUMN_MAP = {v: k for k, v in COLUMN_MAP.items()}
    df = df.rename(columns=REVERSE_COLUMN_MAP)

    return df


# ---------------------------
# Model
# ---------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ---------------------------
# UI: trasa dynamiczna (punkty) + dodawanie/usuwanie
# ---------------------------
if "addresses" not in st.session_state:
    st.session_state.addresses = ["", ""]

st.subheader("Trasa")

# ➕ poza formą
if st.button("➕ Dodaj punkt"):
    st.session_state.addresses.append("")
    st.rerun()

with st.form("main"):
    for i, addr in enumerate(st.session_state.addresses):
        cols = st.columns([6, 1])

        label = (
            "Punkt startowy (origin)" if i == 0
            else ("Punkt końcowy (destination)"
                  if i == len(st.session_state.addresses) - 1
                  else f"Stop {i}")
        )

        placeholder = (
            "Załadunek" if i == 0
            else ("Rozładunek"
                  if i == len(st.session_state.addresses) - 1
                  else "Przystanek pośredni")
        )

        with cols[0]:
            if i == 0:
                # „combo” w miejscu inputa: selectbox jako główne pole
                options = ["— wybierz —"] + QUICK_LOADS + [OTHER_LABEL]
        
                # jeśli obecna wartość jest jednym z gotowców, ustaw ją jako wybraną
                default_idx = 0
                if st.session_state.addresses[0] in QUICK_LOADS:
                    default_idx = options.index(st.session_state.addresses[0])
        
                picked = st.selectbox(
                    label,                        # to jest dokładnie "Punkt startowy (origin)" w tym samym miejscu
                    options,
                    index=default_idx,
                    key="origin_pick",
                    placeholder="Załadunek",      # placeholder zostaje
                )
        
                # jeśli użytkownik wybrał gotowy adres → wpisujemy go do addresses[0]
                if picked in QUICK_LOADS:
                    st.session_state.addresses[0] = picked
        
                # jeśli wybrał Inny adres → pokazujemy normalny input z placeholderem
                if picked == OTHER_LABEL:
                    st.session_state.addresses[0] = st.text_input(
                        label,                    # nadal to samo pole „w tym miejscu”
                        value="" if st.session_state.addresses[0] in QUICK_LOADS else st.session_state.addresses[0],
                        placeholder="Załadunek",
                        key="origin_manual",
                    )
        
                # jeśli nic nie wybrał — zostawiamy wartość jaka była (zwykle pustą)
                if picked == "— wybierz —" and st.session_state.addresses[0] in QUICK_LOADS:
                    st.session_state.addresses[0] = ""
        
            else:
                st.session_state.addresses[i] = st.text_input(
                    label,
                    value=addr,
                    placeholder=placeholder,
                    key=f"address_{i}",
                )



        # ❌ usuń punkt (nie usuwamy origin)
        if i > 0:
            with cols[1]:
                if st.form_submit_button("❌", key=f"remove_{i}"):
                    st.session_state.addresses.pop(i)
                    st.rerun()

    transport_ui = st.selectbox(
        "Typ transportu",
        list(TRANSPORT_UI_TO_MODEL.keys()),
        index=0
    )

    with st.expander("Zaawansowane (opcjonalne)"):
        st.markdown("**Opcje + (każda +100 zł):**")
        selected_opts = []
        for label, key in OPTIONS_PLUS:
            if st.checkbox(label, value=False, key=f"opt_{key}"):
                selected_opts.append(key)

    submitted = st.form_submit_button("Policz")


# ---------------------------
# Wynik i moduły pod spodem
# ---------------------------
if submitted:
    run_id = dt.datetime.now().strftime("%Y%m%d%H%M%S%f")
    with st.spinner("Liczenie trasy i wyceny..."):
        try:
            # adresy
            addresses = [a.strip() for a in st.session_state.addresses if a.strip()]
            if len(addresses) < 2:
                st.error("Podaj co najmniej punkt startowy i końcowy.")
                st.stop()

            origin = addresses[0]
            destination = addresses[-1]
            stops = addresses[1:-1]
            stops_count = len(stops)

            options_count = len(selected_opts)

            # transport: do modelu (naczepa/solo/bus), do Google zawsze driving
            model_transport = TRANSPORT_UI_TO_MODEL[transport_ui]
            google_mode = "driving"

            # trasa km
            km = route_km(origin, destination, google_mode, tuple(stops))

            # geocode origin + destination (kierunek)
            o_lat, o_lon, o_fmt = geocode(origin)
            d_lat, d_lon, d_fmt = geocode(destination)

            brng = bearing_deg(o_lat, o_lon, d_lat, d_lon)
            direction = bearing_to_compass_16(brng)

            # dest_score z ostatniego punktu
            hub_name, hub_km, dest_score = nearest_hub_and_score(d_lat, d_lon)
            dest_score = float(dest_score)

            # feature’y do modelu
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

            # ---- UI: wyniki
            st.success("Gotowe ✅")
            st.metric("Wycena końcowa", f"{final_price:,.2f} zł")
            st.caption(f"Baza z modelu: {base_price:,.2f} zł | Dopłaty: {surcharge:,.2f} zł")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Długość trasy", f"{km:.2f} km")
            c2.metric("Najbliższy hub", f"{hub_name}")          # kosmetyka: tylko miasto
            c3.metric("dest_score", f"{int(dest_score)}")       # kosmetyka: int
            c4.metric("Kierunek", direction)
            c5.metric("cena/KM", price_km)


            # ---- Moduł: podobne transporty
            st.subheader("Podobne transporty w przeszłości")
            hist = db_read_df(DB_HISTORY_SHEET)

            if hist.empty:
                st.info("Brak danych historycznych w Sheets.")
            else:
                # normalizacja kolumn tekstowych
                for col in ["origin", "destination"]:
                    if col in hist.columns:
                        hist[col] = hist[col].astype(str).str.strip().str.lower()

                cur_origin = norm_city(origin)
                cur_destination = norm_city(destination)

                # 1️⃣ NAJPIERW: dokładnie ta sama relacja A → B
                df = hist[
                    (hist["origin"] == cur_origin) &
                    (hist["destination"] == cur_destination)
                    ]

                if "run_id" in df.columns:
                    df = df[df["run_id"] != run_id]

                # 2️⃣ jeśli brak — ten sam destination (różne originy)
                if df.empty:
                    df = hist[hist["destination"] == cur_destination]

                # 3️⃣ jeśli nadal brak — podobny dystans + dest_score
                if df.empty:
                    tol_km = 50.0
                    df = hist[
                        (hist["dest_score"] == int(dest_score)) &
                        (hist["dlugosc_trasy_km"].between(km - tol_km, km + tol_km))
                        ]


                # sortuj: najpierw dokładna relacja, potem dystans

                if "timestamp" in df.columns:
                    df = df.sort_values("timestamp", ascending=False)

                if "final_price" in df.columns and "dlugosc_trasy_km" in df.columns:
                    df["price_per_km"] = (df["final_price"] / df["dlugosc_trasy_km"]).round(2)

                show_cols = [
                    c for c in [
                        "timestamp",
                        "origin",
                        "destination",
                        "typ_transportu",
                        "dlugosc_trasy_km",
                        "dest_score",
                        "final_price"
                        "price_km"
                    ]
                    if c in df.columns
                ]

                if df.empty:
                    st.info("Nie znaleziono podobnych transportów.")
                else:
                    df_ui = df.rename(columns=COLUMN_MAP)
                    st.dataframe(df_ui)

                    # ---- zapis do history w Sheets (opcjonalne)
                    ts = dt.datetime.now().strftime("%Y-%m-%d")
                    if APPS_SCRIPT_URL:
                        row = [
                            run_id,
                            ts,
                            origin.strip().title(),
                            destination.strip().title(),
                            " | ".join(stops).title(),
                            model_transport,
                            #direction,
                            km,
                            int(dest_score),
                            hub_name,
                            base_price,
                            surcharge,
                            final_price,
                            price_km,
                        ]

                        db_append_row(DB_HISTORY_SHEET, row)
                        db_read_df.clear()  # odśwież cache

            # ---- Moduł: popraw cenę (zapis do corrections)
            st.subheader("Popraw wycenę")

            corrected = st.number_input(
                "Cena poprawiona (zł)",
                min_value=0.0, step=50.0,
                value=float(final_price)
            )
            if st.button("Zapisz poprawkę"):
                delta = round(float(corrected) - float(final_price), 2)
                st.write(f"Delta (poprawiona - wyliczona): **{delta:,.2f} zł**")

                corr_row = [
                    ts,
                    origin,
                    destination,
                    #direction,
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
                st.write("Opcje +:", selected_opts)
                st.write("Bearing (deg):", brng)
                st.write("Feature’y do modelu:", X)

        except GoogleAPIError as e:
            st.error(f"Google API: {e}")
        except Exception as e:
            st.exception(e)
