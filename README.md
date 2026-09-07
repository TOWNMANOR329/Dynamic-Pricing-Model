# Dynamic Pricing System

A two-stage pricing model for the hotel/vacation-rental listings:

1. **Base price model** (real, trained on actual data) — predicts a fair
   current price from a property's static characteristics (location,
   size, amenities, capacity).
2. **Dynamic multiplier** (transparent rules, not a trained model) —
   adjusts that base price for weekday, season, festivals, nearby
   events, occupancy/scarcity, and competitor pricing, then clips the
   result to ±20% of the property's current listed price so it can
   never jump unpredictably.

Read the **"Why base + rules, not one trained dynamic model"** section
before changing the architecture — this was a deliberate decision, not
an oversight.

---

## Pipeline at a glance

```
data.py                        →  LightGBM_Ready_Data_<ts>.xlsx
        │                            (property data + real
        │                             weather/events/holidays/
        │                             competitor context)
        ▼
Light_GBM.py                   →  base_price_model_<ts>.txt   (native LightGBM)
                                   base_price_model_<ts>.onnx  (for Node inference)
                                   base_price_feature_schema_<ts>.json
                                   base_price_metrics_<ts>.json
                                   base_price_feature_importance_<ts>.png

dynamic_multiplier.py          →  no output file — imported by generate_pricing_calendar.py
                                   and by the Node backend to apply Stage 2 rules

generate_pricing_calendar.py   →  pricing_calendar_<ts>.xlsx
                                   (offline preview of a live 30-day calendar;
                                    optional, not required to ship the model)

weatherAndCompetitorService.js →  Node-side equivalent of data.py's weather/
                                   competitor calls, for the live backend's
                                   nightly pricing job
```

---

## ⚠️ Naming gotcha — read before renaming any file

**Do not name the training script `lightgbm.py`** (exactly that, all
lowercase, no separator). It does `import lightgbm as lgb` internally
— if the file itself is named `lightgbm.py`, Python imports the file
instead of the installed package, and the script silently breaks
(`lgb.Booster` won't exist, no useful error). The current name,
**`Light_GBM.py`**, is safe — the underscore and capitalization make
it a distinct module name from the `lightgbm` package, confirmed by
testing. If it gets renamed again, just avoid the exact lowercase
`lightgbm.py` (same caution applies to `optuna.py`, `pandas.py`, or
any other name matching an imported package exactly).

---

## Setup

**Python** (3.10+):
```
pip install pandas numpy requests python-dotenv scikit-learn matplotlib \
            lightgbm optuna onnxmltools onnxconverter-common onnx
```

**Node** (18+, for `weatherAndCompetitorService.js` — built-in `fetch`
requires it).

**Environment variables** (`.env` in the same directory as `data.py`):
```
OPENWEATHERMAP_API_KEY=...
TICKETMASTER_CONSUMER_KEY=...
TICKETMASTER_SECRET=...
CALENDARIFIC_API_KEY=...
RAPIDAPI_KEY=...
```
Same keys are reused by `weatherAndCompetitorService.js` on the Node
side — set them as env vars there too.

---

## How to run

```bash
# 1. Generate the training data (fetches live property data + enrichment)
python data.py

# 2. Train the base price model
python Light_GBM.py                      # auto-finds the newest xlsx
python Light_GBM.py path/to/specific.xlsx # or point at one explicitly

# 3. (Optional) Preview a 30-day live pricing calendar offline
python generate_pricing_calendar.py               # auto-finds newest model + data
```

Step 3 is not required to ship the model — it's a way to sanity-check
what the full base+rules pipeline would output before wiring it into
the live site.

**Always use a matched `.onnx` + `feature_schema.json` pair** — they're
written together, in the same run, with the same timestamp. Don't mix
a schema from one run with a model file from another; the feature
order/encoding is only guaranteed consistent within one run.

---

## File-by-file

### `data.py`
Fetches property data from the internal API, geocodes each property's
city (OpenWeatherMap Geocoding, cached in-memory per run — **not**
persisted to disk, re-geocoded every run), and enriches each property
with weather, Ticketmaster events, Calendarific holidays, and RapidAPI
competitor pricing. Groups properties by city (and city+date) to keep
external API calls to a minimum instead of one call per property.

Outputs `LightGBM_Ready_Data_<timestamp>.xlsx`.

Safe to `import` from other scripts — the live fetch only runs under
`if __name__ == "__main__"`.

**Two state files it persists to disk (don't delete mid-month):**
- `category_mappings.json` — fixed `property_type`/`property_category`
  string → integer code mapping. Codes are assigned once and reused
  forever (new categories get appended, existing ones never renumbered)
  so the same property always encodes the same way whether it's in a
  500-row batch or scored alone later. **Deleting this file resets all
  codes and will silently make an already-trained model wrong** — back
  it up before touching it.
- `rapidapi_usage.json` — RapidAPI call counter (hard-capped at
  110/month) plus a circuit-breaker flag that trips on a 403 (bad
  subscription/host) or 429 (rate limit) response, so a broken endpoint
  doesn't keep burning the monthly budget on calls that will keep
  failing. Resets automatically at the start of each calendar month.

### `Light_GBM.py`
Trains on **only real data**: `target_price` (each property's actual
current listing price) against a **static/intrinsic feature set only**
— see `STATIC_FEATURE_COLS` in the file for the exact 26 columns
(location, size, amenities, capacity, nearby conveniences). Deliberately
excludes every date-driven column (weekday, festival, weather, event,
competitor/occupancy signals) — see "Why base + rules" below for why.

Validation is a plain K-fold split, not time-based — there's no real
temporal structure in this feature set to hold out against, since a
property's base price doesn't logically depend on today's date.

Hyperparameters tuned with Optuna. Outputs the native model, ONNX
export (skipped with a warning if `onnxmltools`/`onnx` aren't
installed — training still completes), the feature schema, metrics,
and a feature-importance chart.

### `dynamic_multiplier.py`
The Stage 2 rules engine. Not trained — just configurable, editable
multipliers (weekend, long weekend, major festival, wedding season,
monsoon, nearby-event proximity, occupancy tiers including scarcity
premium and last-minute desperation discount, competitor pricing
premium/discount). Multipliers compose multiplicatively.

Override any rule via `dynamic_pricing_rules.json` in the working
directory without touching code — see `DEFAULT_RULES` in the file for
every key and its default.

Key function: `recommend_price(base_price, current_price, row, rules)`
→ combines the Stage 1 base price with the Stage 2 multiplier, then
clips to ±20% of `current_price` (see `max_price_move_pct` in the
rules). This guardrail is what stops the dynamic layer from ever
producing a shock price change.

### `generate_pricing_calendar.py`
Generates a real (not simulated) 30-day pricing preview per property:
real calendar math, a real 5-day OpenWeatherMap forecast (cached once
per city, filtered locally per date — not re-fetched per date), an
explicitly-labeled seasonal estimate beyond that 5-day horizon, real
Calendarific/Ticketmaster lookups, and real budget-limited competitor
pricing, all fed into `dynamic_multiplier.recommend_price()`. Every
output row is labeled with its weather `source` (`forecast` vs.
`seasonal_estimate`) so it's never ambiguous which one was used.

Optional / for offline preview — the live site should use its own
nightly job (see below), not this script directly.

### `weatherAndCompetitorService.js`
Node-side equivalent of `data.py`'s weather + competitor logic, for
the live backend's nightly pricing job. Same 110/month RapidAPI cap
and circuit-breaker behavior as the Python side, persisted to
`rapidapi_usage.json` (Node-local copy, separate file from Python's).

**Read the comment at the top of the file** if the backend ever runs
as more than one process (load balancer, serverless) — the local-file
budget counter won't be shared across instances in that case, and
needs to move to a database row instead.

---

## Why base + rules, not one trained dynamic model

Earlier iterations tried training a single model on data with
simulated date-driven price variation (a formula like `price =
base_price × weekend_multiplier × festival_multiplier × ... + noise`).
That produced a very high-looking R² (~0.90), which was **not
meaningful** — the model was just reverse-engineering a formula that
used the same features it was given, not learning real guest/host
behavior. Any "improvement" from tuning that further would have been
fitting our own arithmetic more tightly, never getting closer to
reality.

The current split avoids this: Stage 1 trains on data that's
genuinely real (current listed prices), and Stage 2 is rules we wrote
and can inspect directly — no model pretending to have learned
something it didn't.

**Do not replace `dynamic_multiplier.py` with a trained model until**
the live site has accumulated real historical data: actual realized
prices/bookings for the same properties across many real dates over
time. Only then does a learned residual model (predicted vs. actual,
explained by weekday/festival/event/weather) have real labels to learn
from, and only then will its validation metrics mean what they claim.

---

## Serving in production (recommended shape)

- Load the `.onnx` model **once** at server startup, not per-request.
- Run a **nightly batch job**: for each active property, for the next
  ~30 days, compute base price (ONNX) → dynamic multiplier
  (`dynamic_multiplier.recommend_price`, fed with real per-day
  weather/event/competitor data from `weatherAndCompetitorService.js`)
  → write the result into a `property_id | date | price` table.
- Guest-facing pages just read from that table — no live model call on
  the request path, and thus no dependency between model retraining
  and site request latency.
- Retraining is a fully offline background process. Swap in a new
  `.onnx` file (with its matching schema) once validated against a
  held-out slice; keep the previous model file around for a quick
  rollback.

---

## Open items / known gaps

- **RapidAPI competitor pricing is currently unverified.** The exact
  host/endpoint used in `data.py` and `weatherAndCompetitorService.js`
  (`booking-com.p.rapidapi.com`) was carried over from an earlier
  script and has returned 403s — confirm the correct product/host from
  the RapidAPI dashboard (Code Snippets tab for the subscribed app)
  before relying on this signal in production. Until fixed, competitor
  pricing falls back to neutral placeholder values automatically.
- **Node ONNX inference wiring is not yet built.** `Light_GBM.py`
  produces a valid `.onnx` + schema pair; the Node-side code that loads
  it via `onnxruntime-node` and constructs the feature vector still
  needs to be written once the correct model/schema files are on hand.
