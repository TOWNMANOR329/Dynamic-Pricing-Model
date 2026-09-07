/**
 * weatherAndCompetitorService.js
 *
 * Fetches real weather forecasts and real (budget-limited) competitor
 * pricing for your nightly pricing job -- the Node-side equivalent of
 * the fetch_weather_features / fetch_rapidapi_competitor_insights
 * logic in dynamic_pricing.py.
 *
 * Requires Node 18+ (built-in fetch) and two environment variables:
 *   OPENWEATHERMAP_API_KEY
 *   RAPIDAPI_KEY
 *
 * IMPORTANT -- read before deploying behind more than one server
 * instance: the RapidAPI usage counter below is persisted to a local
 * JSON file, same as the Python version. That's fine for a single
 * server process. If you run multiple instances (load balancer,
 * serverless functions, etc.), a local file is NOT shared between
 * them and the 110/month cap could be exceeded without anyone
 * instance knowing about the others' usage. In that case, replace
 * _loadRapidApiState / _saveRapidApiState with reads/writes to a row
 * in your actual database -- everything else stays the same.
 */

const fs = require('fs');
const path = require('path');

const OPENWEATHERMAP_API_KEY = process.env.OPENWEATHERMAP_API_KEY;
const RAPIDAPI_KEY = process.env.RAPIDAPI_KEY;

const RAPIDAPI_MONTHLY_LIMIT = 110;
const RAPIDAPI_USAGE_FILE = path.join(__dirname, 'rapidapi_usage.json');

// Coarse, generic North-Indian seasonal pattern used ONLY beyond the
// ~5-day real forecast horizon. This is an ESTIMATE, not a forecast --
// every result says so explicitly via `source`, so it's never mistaken
// for real data downstream.
const SEASONAL_AVG_TEMP_BY_MONTH = {
  1: 20.0, 2: 23.0, 3: 28.0, 4: 34.0, 5: 38.0, 6: 36.0,
  7: 32.0, 8: 31.0, 9: 31.0, 10: 29.0, 11: 24.0, 12: 20.0,
};
const MONSOON_MONTHS = new Set([6, 7, 8, 9]);

const COMPETITOR_FALLBACK = { occupancyRate: 0.60, medianPrice: 2400.0, ceilingPrice: 8000.0 };

// ============================================================
// RAPIDAPI MONTHLY BUDGET + CIRCUIT BREAKER (persisted across runs)
// Same behavior as dynamic_pricing.py: hard cap at 110 calls/month,
// and a circuit breaker that trips on 403 (subscription/host issue)
// or 429 (rate/quota hit sooner than expected) so a dead endpoint
// doesn't keep burning the budget on calls that will keep failing.
// ============================================================
function _currentMonthKey() {
  const now = new Date();
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`;
}

function _loadRapidApiState() {
  if (fs.existsSync(RAPIDAPI_USAGE_FILE)) {
    try {
      const data = JSON.parse(fs.readFileSync(RAPIDAPI_USAGE_FILE, 'utf8'));
      if (data.month === _currentMonthKey()) {
        return { count: data.count || 0, circuitOpen: !!data.circuitOpen, circuitReason: data.circuitReason || null };
      }
    } catch (e) {
      console.warn('⚠️ Could not read rapidapi_usage.json, starting fresh:', e.message);
    }
  }
  return { count: 0, circuitOpen: false, circuitReason: null };
}

function _saveRapidApiState(state) {
  try {
    fs.writeFileSync(RAPIDAPI_USAGE_FILE, JSON.stringify({ month: _currentMonthKey(), ...state }));
  } catch (e) {
    console.warn('⚠️ Could not persist rapidapi_usage.json:', e.message);
  }
}

let rapidApiState = _loadRapidApiState();
if (rapidApiState.circuitOpen) {
  console.warn(`⚠️ RapidAPI circuit breaker is OPEN from a previous run (${rapidApiState.circuitReason}). Using fallback values until next month.`);
}

function rapidApiBudgetAvailable() {
  if (rapidApiState.circuitOpen) return false;
  if (rapidApiState.count >= RAPIDAPI_MONTHLY_LIMIT) return false;
  rapidApiState.count += 1;
  _saveRapidApiState(rapidApiState);
  return true;
}

function _tripRapidApiCircuit(reason) {
  if (!rapidApiState.circuitOpen) {
    rapidApiState.circuitOpen = true;
    rapidApiState.circuitReason = reason;
    console.warn(`🛑 RapidAPI circuit breaker tripped: ${reason}. Falling back to estimated values for the rest of this run.`);
    _saveRapidApiState(rapidApiState);
  }
}

// ============================================================
// WEATHER: real forecast (cached once per city, not per date --
// OpenWeatherMap's free /forecast endpoint returns the SAME 5-day
// payload regardless of which date you ask about) + honest fallback
// beyond the forecast horizon.
// ============================================================
const _forecastRawCache = new Map(); // cityKey -> raw forecast entries (or null)

async function _fetchRawForecast(cityKey, lat, lon) {
  if (_forecastRawCache.has(cityKey)) return _forecastRawCache.get(cityKey);

  if (!OPENWEATHERMAP_API_KEY) {
    _forecastRawCache.set(cityKey, null);
    return null;
  }
  try {
    const url = `https://api.openweathermap.org/data/2.5/forecast?lat=${lat}&lon=${lon}&appid=${OPENWEATHERMAP_API_KEY}&units=metric`;
    const response = await fetch(url);
    if (!response.ok) {
      console.warn(`⚠️ Weather forecast error ${response.status}`);
      _forecastRawCache.set(cityKey, null);
      return null;
    }
    const data = await response.json();
    _forecastRawCache.set(cityKey, data.list || []);
    return data.list || [];
  } catch (e) {
    console.warn('⚠️ Weather forecast exception:', e.message);
    _forecastRawCache.set(cityKey, null);
    return null;
  }
}

/**
 * Returns { maxTemp, isMonsoonSeason, source } for one property+date.
 * source is 'forecast' (real, within ~5 days) or 'seasonal_estimate'
 * (honest fallback beyond that) -- always check which one you got.
 */
async function getWeatherForDate(cityKey, lat, lon, targetDate) {
  const dateStr = targetDate.toISOString().slice(0, 10); // YYYY-MM-DD
  const month = targetDate.getMonth() + 1;
  const isMonsoonCalendar = MONSOON_MONTHS.has(month) ? 1 : 0;

  const entries = await _fetchRawForecast(cityKey, lat, lon);
  if (entries) {
    const dayEntries = entries.filter(e => e.dt_txt && e.dt_txt.startsWith(dateStr));
    if (dayEntries.length > 0) {
      const maxTemp = Math.max(...dayEntries.map(e => e.main.temp_max));
      const isRain = dayEntries.some(e => 'rain' in e) ? 1 : 0;
      return {
        maxTemp: Math.round(maxTemp * 10) / 10,
        isMonsoonSeason: (isRain || isMonsoonCalendar) ? 1 : 0,
        source: 'forecast',
      };
    }
  }

  return {
    maxTemp: SEASONAL_AVG_TEMP_BY_MONTH[month] ?? 30.0,
    isMonsoonSeason: isMonsoonCalendar,
    source: 'seasonal_estimate',
  };
}

// ============================================================
// COMPETITOR PRICING: RapidAPI (Booking.com), budget-limited.
// Cached per (city, date) for the duration of one run -- grouping
// properties by city before calling this is what keeps you inside
// the 110/month budget as your property count grows.
// ============================================================
const _competitorCache = new Map(); // `${cityKey}|${dateStr}` -> result, in-memory for one run

async function getCompetitorPricing(cityKey, lat, lon, dateStr) {
  const cacheKey = `${cityKey}|${dateStr}`;
  if (_competitorCache.has(cacheKey)) return _competitorCache.get(cacheKey);

  if (!RAPIDAPI_KEY || !rapidApiBudgetAvailable()) {
    _competitorCache.set(cacheKey, COMPETITOR_FALLBACK);
    return COMPETITOR_FALLBACK;
  }

  try {
    const checkin = dateStr;
    const checkout = new Date(new Date(dateStr).getTime() + 86400000).toISOString().slice(0, 10);
    const url = `https://booking-com.p.rapidapi.com/v1/hotels/search-by-coordinates?latitude=${lat}&longitude=${lon}&checkin_date=${checkin}&checkout_date=${checkout}&units=metric&room_number=1&adults_number=1`;
    const response = await fetch(url, {
      headers: {
        'x-rapidapi-key': RAPIDAPI_KEY,
        'x-rapidapi-host': 'booking-com.p.rapidapi.com',
      },
    });

    if (response.status === 403) {
      _tripRapidApiCircuit('403 Forbidden — check RapidAPI subscription/host header for this endpoint');
    } else if (response.status === 429) {
      _tripRapidApiCircuit('429 Too Many Requests — RapidAPI rate/quota limit hit sooner than expected');
    } else if (response.ok) {
      const data = await response.json();
      const prices = (data.result || [])
        .map(hotel => parseFloat(hotel.min_total_price || 0))
        .filter(p => p > 0)
        .sort((a, b) => a - b);
      if (prices.length > 0) {
        const result = {
          occupancyRate: 0.65,
          medianPrice: prices[Math.floor(prices.length / 2)],
          ceilingPrice: Math.max(...prices),
        };
        _competitorCache.set(cacheKey, result);
        return result;
      }
    } else {
      console.warn(`⚠️ RapidAPI error ${response.status}`);
    }
  } catch (e) {
    console.warn('⚠️ RapidAPI exception:', e.message);
  }

  _competitorCache.set(cacheKey, COMPETITOR_FALLBACK);
  return COMPETITOR_FALLBACK;
}

module.exports = {
  getWeatherForDate,
  getCompetitorPricing,
  rapidApiBudgetAvailable,
};
