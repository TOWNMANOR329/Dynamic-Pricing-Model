"""
Generates a real, live pricing calendar (default: next 30 days) for
every property.

This is the LEGITIMATE version of "expand each property across many
future dates": nothing here is a fabricated ground-truth label.
Instead, every dynamic input is either:
  - honest calendar arithmetic (day of week, month, days-until-date),
  - a REAL near-term weather forecast (OpenWeatherMap's free 5-day
    forecast) or an explicitly-labeled coarse seasonal estimate beyond
    that horizon,
  - a REAL holiday lookup (Calendarific) for that real date,
  - a REAL event lookup (Ticketmaster) for that real date,
  - REAL (budget-limited, circuit-breaker-protected) competitor
    pricing via the same RapidAPI plumbing as dynamic_pricing.py.

Stage 1 (base price) comes from the trained model in
train_base_price_model.py. Stage 2 (the date-driven adjustment) is
the transparent rules engine in dynamic_multiplier.py -- there is no
training or fake-label generation anywhere in this script.

Usage:
    python generate_pricing_calendar.py [model.txt] [data.xlsx] [days]

If omitted, the newest base_price_model_*.txt and
LightGBM_Ready_Data_*.xlsx in the current directory are used, and
days defaults to 30.
"""

import sys
import glob
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

try:
    import lightgbm as lgb
except ImportError:
    sys.exit("❌ lightgbm is not installed. Install it with:\n    pip install lightgbm --break-system-packages\n")

import dynamic_pricing as dp
import train_base_price_model as bpm
import dynamic_multiplier as dm


# ============================================================
# WEATHER: real forecast where possible, honest estimate otherwise
# ============================================================
FORECAST_HORIZON_DAYS = 5  # OpenWeatherMap's free /forecast endpoint only covers ~5 days
_FORECAST_RAW_CACHE = {}  # city_key -> raw forecast entries (or None) -- the API returns the
                          # SAME 5-day/3-hour dataset regardless of which date you ask about,
                          # so this is cached once per city and filtered locally per date,
                          # rather than re-fetched for every date checked.

# Coarse, generic North-Indian seasonal pattern used ONLY beyond the
# 5-day forecast horizon. This is explicitly an ESTIMATE, not a
# forecast, and is not city-specific (true per-city climatology would
# need a paid OpenWeatherMap tier) -- every row using it is labeled
# "seasonal_estimate" in the output so it's never confused with real data.
SEASONAL_AVG_TEMP_BY_MONTH = {
    1: 20.0, 2: 23.0, 3: 28.0, 4: 34.0, 5: 38.0, 6: 36.0,
    7: 32.0, 8: 31.0, 9: 31.0, 10: 29.0, 11: 24.0, 12: 20.0,
}
MONSOON_MONTHS = {6, 7, 8, 9}  # calendar fact, not a random simulation


def fetch_raw_forecast(city_key, lat, lon):
    """Fetches (and caches, once per city) the raw 5-day/3-hour forecast
    payload. Returns the list of forecast entries, or None if the key
    is missing or the call fails."""
    if city_key in _FORECAST_RAW_CACHE:
        return _FORECAST_RAW_CACHE[city_key]

    if not dp.API_KEYS["OPENWEATHERMAP"]:
        _FORECAST_RAW_CACHE[city_key] = None
        return None
    try:
        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {"lat": lat, "lon": lon, "appid": dp.API_KEYS["OPENWEATHERMAP"], "units": "metric"}
        response = requests.get(url, params=params, timeout=19)
        if response.status_code != 200:
            print(f"⚠️ Weather forecast error {response.status_code}")
            _FORECAST_RAW_CACHE[city_key] = None
            return None
        entries = response.json().get("list", [])
        _FORECAST_RAW_CACHE[city_key] = entries
        return entries
    except Exception as e:
        print(f"⚠️ Weather forecast exception: {e}")
        _FORECAST_RAW_CACHE[city_key] = None
        return None


def fetch_weather_forecast(city_key, lat, lon, target_date):
    """Real forecast for dates within FORECAST_HORIZON_DAYS, filtered
    locally out of the cached raw payload. Returns (max_temp, is_rain)
    or None if the date is out of range or no data is available."""
    entries = fetch_raw_forecast(city_key, lat, lon)
    if not entries:
        return None

    date_str = target_date.strftime("%Y-%m-%d")
    day_entries = [e for e in entries if e.get("dt_txt", "").startswith(date_str)]
    if not day_entries:
        return None  # date falls outside the 5-day forecast window

    max_temp = max(e["main"]["temp_max"] for e in day_entries)
    is_rain = 1 if any("rain" in e for e in day_entries) else 0
    return (round(max_temp, 1), is_rain)


def get_weather_for_date(city_key, lat, lon, target_date):
    """Returns (max_temp, is_monsoon_season, source_label). source_label
    is 'forecast' (real) or 'seasonal_estimate' (honest fallback) --
    always included in the output so nobody mistakes one for the other."""
    forecast = fetch_weather_forecast(city_key, lat, lon, target_date)
    month = target_date.month
    is_monsoon = 1 if month in MONSOON_MONTHS else 0

    if forecast is not None:
        max_temp, is_rain = forecast
        return max_temp, (1 if (is_rain or is_monsoon) else 0), "forecast"

    return SEASONAL_AVG_TEMP_BY_MONTH.get(month, 30.0), is_monsoon, "seasonal_estimate"


# ============================================================
# CALENDAR ROW FEATURES (real dynamic inputs for one property+date)
# ============================================================

def compute_calendar_row_features(city_key, lat, lon, target_date, today):
    date_str = target_date.strftime("%Y-%m-%d")
    lead_time = (target_date.normalize() - today.normalize()).days

    # Real calendar facts -- same convention as dynamic_pricing.py
    # (Fri/Sat/Sun counted as the weekend window for a nightly stay).
    is_weekend = 1 if target_date.dayofweek in [4, 5, 6] else 0
    is_wedding_season = 1 if target_date.month in [11, 12, 1, 2] else 0

    # Real holiday lookup (Calendarific), cached by (city, date)
    is_major_festival, is_long_weekend = dp.fetch_calendar_features(city_key, date_str)

    # Real forecast, or an explicitly-labeled seasonal estimate
    max_temp, is_monsoon, weather_source = get_weather_for_date(city_key, lat, lon, target_date)

    # Real event lookup (Ticketmaster), cached by (city, date)
    is_city_event, event_type, distance_to_event = dp.fetch_ticketmaster_events(city_key, lat, lon, date_str)

    # Real, budget-limited, circuit-breaker-protected competitor pricing
    occupancy, median_p, ceiling_p = dp.fetch_rapidapi_competitor_insights(city_key, lat, lon, date_str)

    return {
        "booking_lead_time": lead_time,
        "is_weekend": is_weekend,
        "is_wedding_season": is_wedding_season,
        "is_major_festival": is_major_festival,
        "is_long_weekend": is_long_weekend,
        "realtime_max_temp": max_temp,
        "is_monsoon_season": is_monsoon,
        "weather_source": weather_source,
        "is_city_event": is_city_event,
        "event_type": event_type,
        "distance_to_event": distance_to_event,
        "monthly_occupancy_rate": occupancy,
        "competition_median_price": median_p,
        "price_ceiling": ceiling_p,
    }


# ============================================================
# MAIN
# ============================================================

def find_latest(pattern, label):
    candidates = sorted(glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime)
    if not candidates:
        sys.exit(f"❌ No {label} file found matching '{pattern}'.")
    return Path(candidates[-1])


def main():
    model_arg = sys.argv[1] if len(sys.argv) > 1 else None
    data_arg = sys.argv[2] if len(sys.argv) > 2 else None
    n_days = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    model_path = Path(model_arg) if model_arg else find_latest("base_price_model_*.txt", "base price model")
    data_path = Path(data_arg) if data_arg else find_latest("LightGBM_Ready_Data_*.xlsx", "property data")

    print(f"Loading base price model from {model_path} ...")
    model = lgb.Booster(model_file=str(model_path))

    print(f"Loading property data from {data_path} ...")
    df = pd.read_excel(data_path)
    for col in bpm.CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    print("Predicting base price for every property...")
    df["base_price"] = model.predict(df[bpm.STATIC_FEATURE_COLS])

    rules = dm.load_rules()
    today = pd.Timestamp(datetime.now())
    dates = [today + timedelta(days=i) for i in range(n_days)]

    print(f"Generating a {n_days}-day pricing calendar for {len(df)} properties "
          f"({len(df) * n_days} property-days total)...\n")

    records = []
    for _, prop in df.iterrows():
        lat, lon = prop["latitude"], prop["longitude"]
        city_key = f"{lat:.2f}_{lon:.2f}"  # groups properties in the same geocoded location,
                                            # without needing the city name string (dropped
                                            # from the training data) -- only used for caching.
        for target_date in dates:
            dynamic_fields = compute_calendar_row_features(city_key, lat, lon, target_date, today)
            row_for_rules = {**prop.to_dict(), **dynamic_fields}

            result = dm.recommend_price(
                base_price=prop["base_price"],
                current_price=prop["target_price"],
                row=row_for_rules,
                rules=rules,
            )
            records.append({
                "property_id": prop.get("property_id"),
                "date": target_date.strftime("%Y-%m-%d"),
                "lead_time_days": dynamic_fields["booking_lead_time"],
                "current_price": prop["target_price"],
                **result,
                "weather_source": dynamic_fields["weather_source"],
            })

    calendar_df = pd.DataFrame(records)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = Path(f"pricing_calendar_{timestamp}.xlsx")
    calendar_df.to_excel(output_path, index=False)

    seasonal_estimate_pct = (calendar_df["weather_source"] == "seasonal_estimate").mean() * 100
    print(f"\n💾 Saved {len(calendar_df)} property-day rows to {output_path}")
    print(f"{seasonal_estimate_pct:.1f}% of rows used the seasonal weather estimate "
          f"(beyond the {FORECAST_HORIZON_DAYS}-day real forecast horizon) rather than a real forecast.")
    print(f"RapidAPI budget after this run: {dp.RAPIDAPI_CALLS_USED}/{dp.RAPIDAPI_MONTHLY_LIMIT} used this month.")


if __name__ == "__main__":
    main()
