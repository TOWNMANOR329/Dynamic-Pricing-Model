import os
import time
import json
import threading
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

API_KEYS = {
    "OPENWEATHERMAP": os.getenv("OPENWEATHERMAP_API_KEY"),
    "TICKETMASTER_CONSUMER": os.getenv("TICKETMASTER_CONSUMER_KEY"),
    "TICKETMASTER_SECRET": os.getenv("TICKETMASTER_SECRET"),
    "CALENDARIFIC": os.getenv("CALENDARIFIC_API_KEY"),
    "RAPIDAPI": os.getenv("RAPIDAPI_KEY")
}

# ============================================================
# EXISTING CACHES (kept, but re-keyed to use city where noted)
# ============================================================
WEATHER_CACHE = {}
CALENDAR_CACHE = {}
TICKETMASTER_CACHE = {}
COMPETITOR_CACHE = {}

# ============================================================
# NEW: RAPIDAPI MONTHLY RATE LIMIT (hard cap: 110 calls/month)
# RapidAPI's plan resets on a monthly cycle, and the cap must hold
# across separate runs of this script within the same month -- not
# just within a single run -- so the count is persisted to a small
# JSON file on disk and reloaded/reset automatically each month.
# ============================================================
# ============================================================
# NEW: RAPIDAPI MONTHLY RATE LIMIT (hard cap: 110 calls/month)
# + CIRCUIT BREAKER for 403/429 responses
#
# RapidAPI's plan resets on a monthly cycle, and the cap must hold
# across separate runs of this script within the same month -- not
# just within a single run -- so the count is persisted to a small
# JSON file on disk and reloaded/reset automatically each month.
#
# A 403 (not subscribed to this endpoint / bad host header) or 429
# (an actual rate/quota limit tighter than we assumed) means every
# further call is likely to fail too. Without a circuit breaker the
# script would keep "spending" the 110-call budget on doomed calls.
# The breaker trips on the first 403/429, immediately switches
# everything else in this run to fallback values, and persists that
# state so later runs this month don't retry a dead endpoint either.
# ============================================================
RAPIDAPI_MONTHLY_LIMIT = 110
RAPIDAPI_USAGE_FILE = "rapidapi_usage.json"
_rapidapi_lock = threading.Lock()  # ThreadPoolExecutor uses >1 worker, so this must be thread-safe


def _current_month_key():
    return datetime.now().strftime("%Y-%m")


def _load_rapidapi_state():
    """Reads the persisted call count + circuit-breaker state for the
    current calendar month. Any previous month's state is treated as
    expired (both reset)."""
    if os.path.exists(RAPIDAPI_USAGE_FILE):
        try:
            with open(RAPIDAPI_USAGE_FILE, "r") as f:
                data = json.load(f)
            if data.get("month") == _current_month_key():
                return int(data.get("count", 0)), bool(data.get("circuit_open", False)), data.get("circuit_reason")
        except Exception as e:
            print(f"⚠️ Could not read {RAPIDAPI_USAGE_FILE}, starting fresh: {e}")
    return 0, False, None


def _save_rapidapi_state(count, circuit_open, circuit_reason=None):
    try:
        with open(RAPIDAPI_USAGE_FILE, "w") as f:
            json.dump({
                "month": _current_month_key(),
                "count": count,
                "circuit_open": circuit_open,
                "circuit_reason": circuit_reason,
            }, f)
    except Exception as e:
        print(f"⚠️ Could not persist RapidAPI usage state: {e}")


RAPIDAPI_CALLS_USED, RAPIDAPI_CIRCUIT_OPEN, RAPIDAPI_CIRCUIT_REASON = _load_rapidapi_state()

if RAPIDAPI_CIRCUIT_OPEN:
    print(f"⚠️ RapidAPI circuit breaker is OPEN from a previous run this month ({RAPIDAPI_CIRCUIT_REASON}). "
          f"All RapidAPI calls will use fallback values until next month, or until you manually clear "
          f"'circuit_open' in {RAPIDAPI_USAGE_FILE} after fixing the underlying issue.")


def rapidapi_budget_available():
    """
    Thread-safe check-and-reserve against the 110-calls/month hard cap.
    Returns False immediately (no reservation) if the circuit breaker
    is open. Otherwise reserves (and persists) one call and returns
    True if budget remains; returns False once the cap is hit, so the
    caller can fall back to estimated values instead.
    """
    global RAPIDAPI_CALLS_USED
    with _rapidapi_lock:
        if RAPIDAPI_CIRCUIT_OPEN:
            return False
        if RAPIDAPI_CALLS_USED >= RAPIDAPI_MONTHLY_LIMIT:
            return False
        RAPIDAPI_CALLS_USED += 1
        _save_rapidapi_state(RAPIDAPI_CALLS_USED, RAPIDAPI_CIRCUIT_OPEN, RAPIDAPI_CIRCUIT_REASON)
        return True


def trip_rapidapi_circuit(reason):
    """
    Trips the circuit breaker on a 403/429 response. Thread-safe.
    Persists immediately so later runs this month also skip RapidAPI.
    """
    global RAPIDAPI_CIRCUIT_OPEN, RAPIDAPI_CIRCUIT_REASON
    with _rapidapi_lock:
        if not RAPIDAPI_CIRCUIT_OPEN:
            RAPIDAPI_CIRCUIT_OPEN = True
            RAPIDAPI_CIRCUIT_REASON = reason
            print(f"🛑 RapidAPI circuit breaker tripped: {reason}. "
                  f"Switching to fallback values for all remaining properties this run.")
            _save_rapidapi_state(RAPIDAPI_CALLS_USED, RAPIDAPI_CIRCUIT_OPEN, RAPIDAPI_CIRCUIT_REASON)


# ============================================================
# NEW: PERSISTENT CATEGORY ENCODING
# pandas' `.astype('category').cat.codes` assigns codes based on
# whichever unique values happen to be present in the current batch,
# sorted alphabetically. That's fine for a single one-off training
# run, but it silently breaks the moment this pipeline is used for
# live, one-property-at-a-time inference (e.g. from a Node.js
# backend): the same property_type string can get a *different* code
# depending on what else is in the batch, so a model trained on one
# encoding scores garbage on another -- with no error, just wrong
# numbers. Instead, a fixed mapping is persisted to disk the first
# time a category value is seen, and reused forever after (extended,
# never renumbered, if a brand-new category shows up later).
# ============================================================
CATEGORY_MAPPING_FILE = "category_mappings.json"
_category_lock = threading.Lock()


def _load_category_mappings():
    if os.path.exists(CATEGORY_MAPPING_FILE):
        try:
            with open(CATEGORY_MAPPING_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Could not read {CATEGORY_MAPPING_FILE}, starting fresh: {e}")
    return {}


def _save_category_mappings(mappings):
    try:
        with open(CATEGORY_MAPPING_FILE, "w") as f:
            json.dump(mappings, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not persist {CATEGORY_MAPPING_FILE}: {e}")


CATEGORY_MAPPINGS = _load_category_mappings()  # { "property_type": {"villa": 0, "apartment": 1, ...}, ... }


def encode_categorical_column(series, column_name):
    """
    Maps a raw string column to persisted integer codes, assigning a
    new code only for values never seen before (existing codes never
    change). This is what makes the encoding safe to reuse for live,
    single-property inference later -- the mapping is identical no
    matter how many (or how few) rows are being encoded at once.
    """
    with _category_lock:
        mapping = CATEGORY_MAPPINGS.setdefault(column_name, {})
        changed = False
        codes = []
        for raw_val in series.fillna("Unknown").astype(str):
            key = raw_val.strip().lower()
            if key == "":
                key = "unknown"
            if key not in mapping:
                mapping[key] = len(mapping)
                changed = True
            codes.append(mapping[key])
        if changed:
            _save_category_mappings(CATEGORY_MAPPINGS)
    return pd.Series(codes, index=series.index, dtype="int32")



# NEW: cache so we only ever geocode a given city once per run
CITY_CACHE = {}

# NEW: normalization map so variant spellings collapse onto one
# canonical city name before geocoding / grouping / caching.
CITY_NORMALIZATION = {
    "greater noida": "Greater Noida",
    "greater noida west": "Greater Noida",
    "noida extension": "Greater Noida",
    "noida": "Noida",
    "new delhi": "Delhi",
    "delhi ncr": "Delhi",
    "delhi": "Delhi",
    "gurgaon": "Gurugram",
    "gurugram": "Gurugram",
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "mumbai": "Mumbai",
    "dehradun": "Dehradun",
}

# NEW: generic fallback coordinates (geographic center of India) used
# only if a city cannot be normalized/geocoded at all.
FALLBACK_COORDINATES = (20.5937, 78.9629)


def normalize_city(city_raw):
    """
    NEW: Normalize a raw city string into a canonical city name using
    CITY_NORMALIZATION. Falls back to a title-cased version of the
    original string if there is no explicit mapping, so that "Mumbai"
    and "mumbai" still collapse to the same cache/group key.
    """
    if city_raw is None:
        return "Unknown"
    city_str = str(city_raw).strip()
    if city_str == "" or city_str.lower() in ["nan", "none", "null"]:
        return "Unknown"
    key = city_str.lower().strip()
    return CITY_NORMALIZATION.get(key, city_str.title())


def geocode_city(city_raw, country_code="IN"):
    """
    NEW: Reusable city -> (lat, lon) geocoder.

    * Normalizes the city name first (see normalize_city).
    * Uses CITY_CACHE so each unique city is only geocoded once,
      no matter how many properties share that city.
    * Calls the OpenWeatherMap Geocoding API.
    * Falls back to FALLBACK_COORDINATES on any failure (missing key,
      network error, bad response, city not found) so the pipeline
      never crashes because of a geocoding problem.
    """
    city = normalize_city(city_raw)

    if city in CITY_CACHE:
        return CITY_CACHE[city]

    if not API_KEYS["OPENWEATHERMAP"]:
        print(f"⚠️ OpenWeatherMap API Key missing. Using fallback coordinates for {city}.")
        CITY_CACHE[city] = FALLBACK_COORDINATES
        return FALLBACK_COORDINATES

    try:
        url = "http://api.openweathermap.org/geo/1.0/direct"
        params = {
            "q": f"{city},{country_code}",
            "limit": 1,
            "appid": API_KEYS["OPENWEATHERMAP"],
        }
        response = requests.get(url, params=params, timeout=19)

        if response.status_code == 200:
            results = response.json()
            if results:
                lat = float(results[0]["lat"])
                lon = float(results[0]["lon"])
                CITY_CACHE[city] = (lat, lon)
                print(f"✅ Geocoded {city} -> ({lat}, {lon})")
                return (lat, lon)
            else:
                print(f"⚠️ Geocoding returned no results for {city}. Using fallback coordinates.")
        else:
            print(f"⚠️ OpenWeatherMap Geocoding Error {response.status_code} for {city}")
    except Exception as e:
        print(f"⚠️ OpenWeatherMap Geocoding Exception for {city}: {e}")

    CITY_CACHE[city] = FALLBACK_COORDINATES
    return FALLBACK_COORDINATES


# ============================================================
# EXISTING ENRICHMENT FUNCTIONS
# (signatures updated to accept `city` and cache primarily by
#  city / (city, date) instead of lat/lon, per the new city-based
#  grouping strategy. Lat/lon are still passed through and used for
#  the actual API calls, since some providers require coordinates.)
# ============================================================

def fetch_weather_features(city, lat, lon, date_str):
    # CHANGED: cache key is now the city (was rounded lat/lon), since
    # every property in the same city shares one set of coordinates.
    cache_key = city
    if cache_key in WEATHER_CACHE:
        base_temp, max_temp, is_rain = WEATHER_CACHE[cache_key]
        variance = np.random.uniform(-1.5, 1.5)
        return (base_temp, round(max_temp + variance, 1), is_rain)

    fallback = (28.0, 32.0, 0)
    if not API_KEYS["OPENWEATHERMAP"]:
        print("⚠️ OpenWeatherMap API Key missing.")
        WEATHER_CACHE[cache_key] = fallback
        return fallback

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"lat": lat, "lon": lon, "appid": API_KEYS["OPENWEATHERMAP"], "units": "metric"}
        response = requests.get(url, params=params, timeout=19)

        if response.status_code == 200:
            data = response.json()
            current_temp = data.get("main", {}).get("temp_max", 32.0)
            is_rain = 1 if "rain" in data else 0

            output = (round(current_temp - 3.0, 1), round(current_temp, 1), is_rain)
            WEATHER_CACHE[cache_key] = output
            print(f"✅ OpenWeatherMap Success for {city} ({lat},{lon})")
            return output
        else:
            print(f"⚠️ OpenWeatherMap Error {response.status_code}")
    except Exception as e:
        print(f"⚠️ OpenWeatherMap Exception: {e}")

    WEATHER_CACHE[cache_key] = fallback
    return fallback


def fetch_calendar_features(city, date_str, country_code="IN"):
    # CHANGED: cache key is now (city, date) instead of date alone, so
    # it groups the same way as weather/Ticketmaster/RapidAPI on the
    # (city, date) unique-context set built in extract_and_engineer_features.
    cache_key = (city, date_str)
    if cache_key in CALENDAR_CACHE:
        return CALENDAR_CACHE[cache_key]

    fallback = (0, 0)
    if not API_KEYS["CALENDARIFIC"]:
        CALENDAR_CACHE[cache_key] = fallback
        return fallback

    try:
        dt = pd.to_datetime(date_str)
        url = "https://calendarific.com/api/v2/holidays"
        params = {"api_key": API_KEYS["CALENDARIFIC"], "country": country_code, "year": dt.year, "month": dt.month, "day": dt.day}
        response = requests.get(url, params=params, timeout=19)

        if response.status_code == 200:
            holidays = response.json().get("response", {}).get("holidays", [])
            is_major_festival = 1 if len(holidays) > 0 else 0
            is_long = 1 if (len(holidays) > 0 and dt.dayofweek in [4, 0]) else 0
            output = (is_major_festival, is_long)
            CALENDAR_CACHE[cache_key] = output
            print(f"✅ Calendarific Success for {city} {date_str}")
            return output
        else:
            print(f"⚠️ Calendarific Error {response.status_code}")
    except Exception as e:
        print(f"⚠️ Calendarific Exception: {e}")

    CALENDAR_CACHE[cache_key] = fallback
    return fallback


def fetch_ticketmaster_events(city, lat, lon, date_str):
    # CHANGED: cache key is now (city, date) instead of (lat, lon, date).
    cache_key = (city, date_str)
    if cache_key in TICKETMASTER_CACHE:
        return TICKETMASTER_CACHE[cache_key]

    fallback = (0, 0, 999.0)
    if not API_KEYS["TICKETMASTER_CONSUMER"]:
        TICKETMASTER_CACHE[cache_key] = fallback
        return fallback

    try:
        url = "https://app.ticketmaster.com/discovery/v2/events.json"
        start_dt = pd.to_datetime(date_str).strftime('%Y-%m-%dT00:00:00Z')
        params = {"apikey": API_KEYS["TICKETMASTER_CONSUMER"], "latlong": f"{lat},{lon}", "radius": "10", "unit": "km", "startDateTime": start_dt}
        response = requests.get(url, params=params, timeout=19)

        if response.status_code == 200:
            events = response.json().get("_embedded", {}).get("events", [])
            if events:
                first_event = events[0]
                event_type_str = first_event.get("classifications", [{}])[0].get("segment", {}).get("name", "Other")
                type_mapping = {"Sports": 1, "Music": 2, "Arts & Theatre": 3, "Other": 0}
                output = (1, type_mapping.get(event_type_str, 0), 2.5)
                TICKETMASTER_CACHE[cache_key] = output
                print(f"✅ Ticketmaster Success for {city} {date_str}")
                return output
            else:
                output = (0, 0, 999.0)
                TICKETMASTER_CACHE[cache_key] = output
                print(f"✅ Ticketmaster Success for {city} {date_str} (No events)")
                return output
        else:
            print(f"⚠️ Ticketmaster Error {response.status_code}")
    except Exception as e:
        print(f"⚠️ Ticketmaster Exception: {e}")

    TICKETMASTER_CACHE[cache_key] = fallback
    return fallback


def fetch_rapidapi_competitor_insights(city, lat, lon, date_str):
    # CHANGED: cache key is now (city, date) instead of (lat, lon, date).
    # Endpoint itself is untouched -- it still receives lat/lon, which
    # are now generated from the city via geocode_city().
    cache_key = (city, date_str)
    if cache_key in COMPETITOR_CACHE:
        return COMPETITOR_CACHE[cache_key]

    fallback = (0.60, 2400.0, 8000.0)
    if not API_KEYS["RAPIDAPI"]:
        COMPETITOR_CACHE[cache_key] = fallback
        return fallback

    # NEW: hard monthly cap (110 calls/month, persisted across runs) +
    # circuit breaker. Once either is tripped, fall back to estimated
    # competitor values instead of calling the API.
    if not rapidapi_budget_available():
        if RAPIDAPI_CIRCUIT_OPEN:
            print(f"⚠️ RapidAPI circuit breaker open ({RAPIDAPI_CIRCUIT_REASON}) — using fallback for {city} {date_str}")
        else:
            print(f"⚠️ RapidAPI monthly budget ({RAPIDAPI_MONTHLY_LIMIT}) exhausted — using fallback values for {city} {date_str}")
        COMPETITOR_CACHE[cache_key] = fallback
        return fallback

    try:
        url = "https://booking-com.p.rapidapi.com/v1/hotels/search-by-coordinates"
        headers = {"x-rapidapi-key": API_KEYS["RAPIDAPI"], "x-rapidapi-host": "booking-com.p.rapidapi.com"}
        params = {"latitude": lat, "longitude": lon, "checkin_date": pd.to_datetime(date_str).strftime('%Y-%m-%d'), "checkout_date": (pd.to_datetime(date_str) + pd.Timedelta(days=1)).strftime('%Y-%m-%d'), "units": "metric", "room_number": "1", "adults_number": "1"}
        response = requests.get(url, headers=headers, params=params, timeout=19)

        if response.status_code == 200:
            results = response.json().get("result", [])
            prices = [float(hotel.get("min_total_price", 0)) for hotel in results if float(hotel.get("min_total_price", 0)) > 0]
            if prices:
                prices.sort()
                output = (0.65, prices[len(prices) // 2], max(prices))
                COMPETITOR_CACHE[cache_key] = output
                print(f"✅ RapidAPI Competitor Success for {city} {date_str}")
                return output
            else:
                COMPETITOR_CACHE[cache_key] = fallback
                print(f"✅ RapidAPI Competitor Success for {city} {date_str} (No prices)")
                return fallback
        elif response.status_code == 403:
            # NEW: 403 almost always means the key isn't subscribed to
            # this endpoint, or the x-rapidapi-host header doesn't match
            # the subscription -- every further call will fail the same
            # way, so stop trying instead of burning the monthly budget.
            trip_rapidapi_circuit("403 Forbidden — check RapidAPI subscription/host header for this endpoint")
        elif response.status_code == 429:
            # NEW: 429 means an actual rate/quota limit was hit that is
            # tighter than the 110/month we assumed (per-second/day cap,
            # or the monthly quota was already partly used elsewhere).
            trip_rapidapi_circuit("429 Too Many Requests — RapidAPI rate/quota limit hit sooner than expected")
        else:
            print(f"⚠️ RapidAPI Error {response.status_code}")
    except Exception as e:
        print(f"⚠️ RapidAPI Exception: {e}")

    COMPETITOR_CACHE[cache_key] = fallback
    return fallback


def parse_room_count(val, guest_capacity):
    if isinstance(val, (list, dict, np.ndarray)):
        return 1
    val_str = str(val).strip()
    if val_str == '' or val_str == '0' or val_str.lower() in ['none', 'nan', 'null']:
        return max(1, int(guest_capacity) // 2)
    if '{' in val_str or '[' in val_str or ':' in val_str:
        return 1
    try:
        parsed = int(float(val_str))
        return parsed if parsed > 0 else max(1, int(guest_capacity) // 2)
    except:
        return 1


def get_bool_column(df_raw, col_name):
    """
    FIX: df_raw.get(col_name, False) returns a plain Python bool (not a
    Series) whenever col_name is missing, and .fillna() on a bool then
    raises AttributeError. This helper always returns a proper boolean
    Series aligned to df_raw's index, whether or not the column exists.
    """
    if col_name in df_raw.columns:
        return df_raw[col_name].fillna(False).astype(bool)
    return pd.Series(False, index=df_raw.index)


def extract_and_engineer_features(df_raw):
    df_features = pd.DataFrame()

    df_features['property_id'] = pd.to_numeric(df_raw.get('id', 0), errors='coerce').fillna(0).astype('int32')

    city_mapping = {'Delhi': 1, 'Mumbai': 1, 'Noida': 1, 'Dehradun': 2}
    df_features['city_tier'] = df_raw.get('city', '').map(city_mapping).fillna(3).astype('int32')

    # CHANGED: use the persisted mapping (encode_categorical_column) instead
    # of pandas .cat.codes, so codes stay stable across runs and are safe
    # to reuse for live single-property inference later.
    df_features['property_type'] = encode_categorical_column(df_raw.get('property_type', pd.Series('', index=df_raw.index)), 'property_type')
    df_features['property_category'] = encode_categorical_column(df_raw.get('property_category', pd.Series('', index=df_raw.index)), 'property_category')

    df_features['guest_capacity'] = pd.to_numeric(df_raw.get('max_guests', 1), errors='coerce').fillna(1).astype('int32')
    df_features['booking_lead_time'] = 0

    amenities_str = df_raw.get('amenities', '').astype(str).str.lower()

    important_amenities = {
        'has_powerbackup': 'power backup|inverter',
        'has_ac_heater': 'air conditioning|ac|air conditioner|heating',
        'has_wifi': 'wi-fi|wifi|internet',
        'has_tv': 'tv|television|smart tv',
        'has_parking': 'parking|garage|vehicle space',
        'has_kitchen': 'kitchen|cooking',
        'has_pool': 'pool|swimming pool|jacuzzi',
        'has_gym': 'gym|fitness center'
    }

    for feature_name, keywords in important_amenities.items():
        df_features[feature_name] = amenities_str.str.contains(keywords, case=False, regex=True, na=False).astype('int32')

    # FIX: use get_bool_column so missing columns don't crash .fillna()
    df_features['is_couple_friendly'] = get_bool_column(df_raw, 'guest_policy.unmarried_couple_allowed').astype('int32')
    df_features['pets_allowed'] = get_bool_column(df_raw, 'pets_allowed').astype('int32')
    df_features['events_allowed'] = get_bool_column(df_raw, 'events_allowed').astype('int32')
    df_features['family_allowed'] = get_bool_column(df_raw, 'guest_policy.family_allowed').astype('int32')
    df_features['bachelors_allowed'] = get_bool_column(df_raw, 'guest_policy.bachelors_allowed').astype('int32')

    df_features['area'] = pd.to_numeric(df_raw.get('area', 0), errors='coerce').fillna(0).astype('float32')

    raw_bedrooms = df_raw.get('bedrooms', pd.Series(0, index=df_raw.index))
    raw_bathrooms = df_raw.get('bathrooms', pd.Series(0, index=df_raw.index))

    df_features['bedrooms'] = [parse_room_count(b, g) for b, g in zip(raw_bedrooms, df_features['guest_capacity'])]
    df_features['bathrooms'] = [parse_room_count(b, g) for b, g in zip(raw_bathrooms, df_features['guest_capacity'])]

    df_features['bedrooms'] = df_features['bedrooms'].astype('int32')
    df_features['bathrooms'] = df_features['bathrooms'].astype('int32')

    # ============================================================
    # NEW: CITY-BASED LAT/LON GENERATION
    # Replaces the old "read lat/lon straight from the dataset" logic.
    # Most properties are missing latitude/longitude but do have a
    # valid city, so we normalize + geocode the city instead, and
    # only geocode each unique city once (via CITY_CACHE).
    # ============================================================
    df_features['city'] = df_raw.get('city', 'Unknown').apply(normalize_city)

    unique_cities = df_features['city'].unique()
    print(f"\nGeocoding {len(unique_cities)} unique cities (from {len(df_features)} properties)...")
    for city in unique_cities:
        geocode_city(city)  # populates CITY_CACHE

    df_features['latitude'] = df_features['city'].map(lambda c: CITY_CACHE.get(c, FALLBACK_COORDINATES)[0]).astype('float64')
    df_features['longitude'] = df_features['city'].map(lambda c: CITY_CACHE.get(c, FALLBACK_COORDINATES)[1]).astype('float64')

    def text_to_binary(col_name):
        series = df_raw.get(col_name, pd.Series(0, index=df_raw.index))
        return series.fillna('').astype(str).str.strip().str.lower().apply(
            lambda x: 0 if x in ['', 'false', 'none', 'nan', '0'] else 1
        ).astype('int32')

    df_features['taxi_nearby'] = text_to_binary('guidebook.transport_tips.taxi')
    df_features['parking_nearby'] = text_to_binary('guidebook.transport_tips.parking')
    df_features['restaurants_nearby'] = text_to_binary('guidebook.cafes_restaurants')
    df_features['grocery_nearby'] = text_to_binary('guidebook.essentials_nearby.grocery')

    df_features['target_price'] = pd.to_numeric(df_raw.get('price', 0), errors='coerce').fillna(0).astype('float32')

    print("\nSynthesizing dynamic search dates and extracting market contexts...")

    np.random.seed(42)
    future_days_offset = np.random.randint(1, 30, size=len(df_raw))
    current_time = pd.Timestamp(datetime.now())
    synthetic_dates = current_time + pd.to_timedelta(future_days_offset, unit='D')

    df_features['raw_date_str'] = synthetic_dates.strftime('%Y-%m-%d')

    # ============================================================
    # NEW: CITY-BASED GROUPING FOR API RATE-LIMIT CONTROL
    # CHANGED: unique_contexts is now deduplicated on (city, date)
    # instead of (latitude, longitude, date). Functionally this
    # produces the same grouping (same city => same geocoded lat/lon),
    # but keying explicitly by city makes the intent clear and lets
    # the enrichment functions cache by city instead of raw
    # coordinates, which is required by the new caching strategy.
    # ============================================================
    unique_contexts = (
        df_features
        .groupby(['city', 'latitude', 'longitude', 'raw_date_str'])
        .size()
        .reset_index(name='property_count')
        .sort_values('property_count', ascending=False)
        .reset_index(drop=True)
    )
    print(f"Reduced {len(df_features)} properties to {len(unique_contexts)} unique (city, date) API queries. Fetching concurrently...\n")
    print(f"RapidAPI budget: {RAPIDAPI_CALLS_USED}/{RAPIDAPI_MONTHLY_LIMIT} used this month "
          f"({RAPIDAPI_MONTHLY_LIMIT - RAPIDAPI_CALLS_USED} remaining). "
          f"Contexts are processed largest-property-count-first so the budget covers the most properties.\n")

    def fetch_all_for_context(row):
        time.sleep(1.5)
        city, lat, lon, date_str = row['city'], row['latitude'], row['longitude'], row['raw_date_str']
        dt = pd.to_datetime(date_str)

        festival, long_wknd = fetch_calendar_features(city, date_str)
        avg_temp, max_temp, monsoon = fetch_weather_features(city, lat, lon, date_str)
        city_event, ev_type, dist_ev = fetch_ticketmaster_events(city, lat, lon, date_str)
        occupancy, median_p, ceiling_p = fetch_rapidapi_competitor_insights(city, lat, lon, date_str)

        return {
            'city': city, 'latitude': lat, 'longitude': lon, 'raw_date_str': date_str,
            'checkin_day': dt.day, 'checkin_month': dt.month,
            'is_weekend': 1 if dt.dayofweek in [4, 5, 6] else 0,
            'is_wedding_season': 1 if dt.month in [11, 12, 1, 2] else 0,
            'is_major_festival': festival, 'is_long_weekend': long_wknd,
            'historical_avg_temp': avg_temp, 'realtime_max_temp': max_temp, 'is_monsoon_season': monsoon,
            'is_city_event': city_event, 'event_type': ev_type, 'distance_to_event': dist_ev,
            'monthly_occupancy_rate': occupancy, 'competition_median_price': median_p, 'price_ceiling': ceiling_p
        }

    enriched_data = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        tasks = [row for _, row in unique_contexts.iterrows()]
        results = executor.map(fetch_all_for_context, tasks)
        for res in results:
            enriched_data.append(res)

    df_external = pd.DataFrame(enriched_data)
    # CHANGED: merge key now includes 'city' to match the new grouping key.
    df_features = pd.merge(df_features, df_external, on=['city', 'latitude', 'longitude', 'raw_date_str'], how='left')

    df_features = df_features.drop(columns=['city'])
    # CHANGED: raw_date_str is now kept (renamed to search_date) instead
    # of being dropped, so downstream model training has a real date
    # column to sort/split on for a time-based train/test split.
    df_features = df_features.rename(columns={'raw_date_str': 'search_date'})

    return df_features


def main():
    api_url = 'https://www.townmanor.ai/api/ovika/properties'

    print("Fetching data from the API...")
    response = requests.get(api_url)

    if response.status_code == 200:
        property_data = response.json()

        if isinstance(property_data, list):
            df_raw = pd.json_normalize(property_data)
        elif isinstance(property_data, dict) and "data" in property_data:
            df_raw = pd.json_normalize(property_data["data"])
        else:
            df_raw = None

        if df_raw is not None:
            df_model_ready = extract_and_engineer_features(df_raw)

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_file = f'LightGBM_Ready_Data_{timestamp}.xlsx'

            df_model_ready.to_excel(output_file, index=False)
            print(f"\n✅ Success! Generated {len(df_model_ready.columns)} columns for {len(df_model_ready)} properties.")
            print(f"💾 Saved clean ML matrix to {output_file}")
    else:
        print(f"❌ Failed to fetch data. Status code: {response.status_code}")


# CHANGED: guarded with __main__ so other scripts (e.g.
# generate_pricing_calendar.py) can safely `import dynamic_pricing`
# to reuse its geocoding/weather/event/competitor functions without
# triggering a live property fetch as a side effect of the import.
if __name__ == "__main__":
    main()

