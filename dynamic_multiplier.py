"""
Stage 2: Dynamic price adjustment layer.

WHY RULE-BASED, NOT LEARNED (read this before changing it):
A *learned* dynamic multiplier needs real historical data -- the same
property's actual realized price or booking outcome across many
different real dates, so a model can learn how demand should move
with weekday, season, and events. Right now the dataset has exactly
ONE static current price per property, paired with enrichment data
for a single arbitrary future date. There's no historical price
variation for a model to learn from -- a "learned" multiplier trained
on this data would just be fitting noise and would look good on paper
while being meaningless.

This module instead encodes standard hospitality revenue-management
heuristics as a configurable, editable multiplier -- a reasonable
starting point used by many real systems before enough transaction
history exists. See the bottom of this file for how to graduate to a
learned model once your live site has accumulated real booking data.

Usage as a library:
    from dynamic_multiplier import recommend_price
    result = recommend_price(base_price=6000, current_price=5800, row=row_dict)

Usage as a CLI (batch-scores a base-price-model output file):
    python dynamic_multiplier.py base_price_predictions.xlsx
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG -- tune to your market. All multipliers are
# MULTIPLICATIVE and compose together, e.g. a weekend during a
# festival applies both factors, not just the bigger one.
# Override any of these via dynamic_pricing_rules.json (see load_rules).
# ============================================================
DEFAULT_RULES = {
    "weekend_multiplier": 1.12,
    "long_weekend_multiplier": 1.08,        # applied ON TOP OF weekend_multiplier if both true
    "major_festival_multiplier": 1.20,
    "wedding_season_multiplier": 1.10,
    "monsoon_multiplier": 0.92,
    "city_event_near_multiplier": 1.15,     # distance_to_event < event_near_km
    "city_event_far_multiplier": 1.08,      # event_near_km <= distance <= event_far_km
    "event_near_km": 5.0,
    "event_far_km": 10.0,

    # Occupancy is tiered from most to least extreme so only ONE
    # occupancy-based multiplier ever applies -- otherwise "very high
    # occupancy" would double-stack both the scarcity premium AND the
    # regular high-occupancy bump for the same underlying signal.
    "extreme_scarcity_threshold": 0.85,      # near-total sellout in the area
    "scarcity_premium_multiplier": 1.25,
    "high_occupancy_threshold": 0.75,
    "high_occupancy_multiplier": 1.10,
    "low_occupancy_threshold": 0.40,
    "low_occupancy_multiplier": 0.90,

    # Last-minute desperation discount: ONLY applies when a date is
    # both imminent (within last_minute_days) AND still empty
    # (occupancy <= desperation_occupancy_threshold) -- takes priority
    # over the generic low-occupancy multiplier when both would fire,
    # since "empty AND about to happen" is a stronger signal than
    # "empty" alone. Requires booking_lead_time (days until the date
    # being priced) -- only meaningful when scoring a real future date
    # (see generate_pricing_calendar.py), not the static training data.
    "last_minute_days": 3,
    "desperation_occupancy_threshold": 0.40,
    "desperation_discount_multiplier": 0.85,

    "competitor_premium_multiplier": 1.05,  # nearby competitors pricing meaningfully above base
    "competitor_discount_multiplier": 0.95, # nearby competitors pricing meaningfully below base
    "competitor_premium_ratio": 1.10,       # competition_median_price / base_price >= this
    "competitor_discount_ratio": 0.90,      # competition_median_price / base_price <= this
    "max_price_move_pct": 0.20,             # final guardrail vs. CURRENT listed price
}

RULES_FILE = "dynamic_pricing_rules.json"


def load_rules(path=RULES_FILE):
    """Loads rule overrides from JSON if present, so you can tune the
    dynamic layer without touching code. Missing keys fall back to
    DEFAULT_RULES."""
    rules = dict(DEFAULT_RULES)
    if Path(path).exists():
        try:
            with open(path) as f:
                overrides = json.load(f)
            rules.update(overrides)
        except Exception as e:
            print(f"⚠️ Could not read {path}, using defaults: {e}")
    return rules


def compute_dynamic_multiplier(row, base_price, rules):
    """
    row: dict-like (or pandas Series) with the dynamic/date-driven
         fields: is_weekend, is_long_weekend, is_major_festival,
         is_wedding_season, is_monsoon_season, is_city_event,
         distance_to_event, monthly_occupancy_rate,
         competition_median_price.
    base_price: this property's predicted base price (stage 1 output).
    Returns the combined multiplicative adjustment factor.
    """
    multiplier = 1.0

    if row.get("is_weekend"):
        multiplier *= rules["weekend_multiplier"]
    if row.get("is_long_weekend"):
        multiplier *= rules["long_weekend_multiplier"]
    if row.get("is_major_festival"):
        multiplier *= rules["major_festival_multiplier"]
    if row.get("is_wedding_season"):
        multiplier *= rules["wedding_season_multiplier"]
    if row.get("is_monsoon_season"):
        multiplier *= rules["monsoon_multiplier"]

    if row.get("is_city_event"):
        dist = row.get("distance_to_event", 999.0)
        if dist < rules["event_near_km"]:
            multiplier *= rules["city_event_near_multiplier"]
        elif dist <= rules["event_far_km"]:
            multiplier *= rules["city_event_far_multiplier"]

    occ = row.get("monthly_occupancy_rate")
    lead_time = row.get("booking_lead_time")
    has_occ = occ is not None and not (isinstance(occ, float) and np.isnan(occ))
    if has_occ:
        if occ >= rules["extreme_scarcity_threshold"]:
            multiplier *= rules["scarcity_premium_multiplier"]
        elif occ >= rules["high_occupancy_threshold"]:
            multiplier *= rules["high_occupancy_multiplier"]
        elif (
            lead_time is not None
            and lead_time <= rules["last_minute_days"]
            and occ <= rules["desperation_occupancy_threshold"]
        ):
            multiplier *= rules["desperation_discount_multiplier"]
        elif occ <= rules["low_occupancy_threshold"]:
            multiplier *= rules["low_occupancy_multiplier"]

    comp_median = row.get("competition_median_price")
    if comp_median and base_price:
        ratio = comp_median / base_price
        if ratio >= rules["competitor_premium_ratio"]:
            multiplier *= rules["competitor_premium_multiplier"]
        elif ratio <= rules["competitor_discount_ratio"]:
            multiplier *= rules["competitor_discount_multiplier"]

    return multiplier


def recommend_price(base_price, current_price, row, rules=None):
    """
    Combines the stage-1 base price with the stage-2 dynamic
    multiplier, then clips the result to +/-max_price_move_pct of the
    property's CURRENT listed price -- the multiplier can never
    produce a shock jump regardless of how many rules stack up.
    """
    rules = rules or load_rules()
    multiplier = compute_dynamic_multiplier(row, base_price, rules)
    dynamic_price = base_price * multiplier

    max_move = rules["max_price_move_pct"]
    lower = current_price * (1 - max_move)
    upper = current_price * (1 + max_move)
    recommended = min(max(dynamic_price, lower), upper)

    return {
        "base_price": round(float(base_price), 2),
        "dynamic_multiplier": round(float(multiplier), 4),
        "dynamic_price_uncapped": round(float(dynamic_price), 2),
        "recommended_price": round(float(recommended), 2),
        "was_clipped_to_bound": bool(dynamic_price < lower or dynamic_price > upper),
    }


def score_dataframe(df, base_price_col="base_price", current_price_col="current_price", rules=None):
    """Batch version of recommend_price for a pandas DataFrame."""
    rules = rules or load_rules()
    records = []
    for _, row in df.iterrows():
        result = recommend_price(row[base_price_col], row[current_price_col], row, rules)
        records.append(result)
    result_df = pd.DataFrame(records)
    id_cols = [c for c in df.columns if c not in result_df.columns]
    return pd.concat([df[id_cols].reset_index(drop=True), result_df], axis=1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(
            "Usage: python dynamic_multiplier.py <base_price_predictions.xlsx>\n"
            "Expects columns: base_price, current_price, plus the dynamic feature "
            "columns (is_weekend, is_major_festival, etc.)"
        )
    df = pd.read_excel(sys.argv[1])
    scored = score_dataframe(df)
    out_path = Path(sys.argv[1]).with_name("dynamic_price_recommendations.xlsx")
    scored.to_excel(out_path, index=False)
    print(f"💾 Saved {out_path}")


# ============================================================
# GRADUATING TO A LEARNED DYNAMIC MODEL LATER
# ============================================================
# Once your live site has real booking/price history over time (the
# same property observed on many real dates, with what it actually
# sold for), you can replace this rules engine with a learned one:
#   1. For each historical (property, date) pair, compute
#      residual = actual_price / base_price_model_prediction
#   2. Train a small regression (even just LightGBM again) on
#      residual ~ is_weekend + is_major_festival + ... + real
#      weather/event data for that ACTUAL date (not a synthetic one)
#   3. Replace compute_dynamic_multiplier's rule lookups with that
#      model's prediction.
# This is the standard "base price x learned demand multiplier"
# architecture used by real revenue management systems -- the rules
# engine here is a placeholder for the multiplier until you have the
# transaction history to learn it properly.
