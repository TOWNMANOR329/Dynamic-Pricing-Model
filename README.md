# Dynamic Pricing Model for Hospitality & Rental Properties

## Overview

This project is a Dynamic Pricing Engine designed for hospitality, vacation rental, PG, villa, apartment, and short-term accommodation platforms.

The system collects property data, enriches it with external market signals, engineers machine learning features, and generates a model-ready dataset that can be used to train pricing prediction models.

The objective is to estimate the optimal listing price by considering:

* Property characteristics
* Amenities and capacity
* Local weather conditions
* Events and festivals
* Competitor pricing
* Seasonal demand
* Market occupancy trends
* Location-based demand indicators

---

# Architecture

```
Property Database/API
         │
         ▼
 Feature Engineering Pipeline
         │
         ├── Weather Enrichment
         ├── Festival Detection
         ├── Event Detection
         ├── Competitor Analysis
         ├── Location Intelligence
         │
         ▼
 Model Ready Dataset
         │
         ▼
 Machine Learning Model
         │
         ▼
 Recommended Dynamic Price
```

---

# Data Sources

## Internal Data

Property information is fetched from:

```python
https://www.townmanor.ai/api/ovika/properties
```

The dataset contains:

* Property Type
* Property Category
* Guest Capacity
* Area
* Bedrooms
* Bathrooms
* Amenities
* Guest Policies
* Base Price
* City Information

---

## External Data Sources

### OpenWeatherMap

Used for:

* City Geocoding
* Temperature Data
* Rain Detection
* Weather-Based Demand Signals

Features Generated:

* historical_avg_temp
* realtime_max_temp
* is_monsoon_season

---

### Calendarific

Used for:

* Public Holiday Detection
* Festival Detection
* Long Weekend Detection

Features Generated:

* is_major_festival
* is_long_weekend

---

### Ticketmaster

Used for:

* Event Discovery
* Event Category Detection
* Distance from Event

Features Generated:

* is_city_event
* event_type
* distance_to_event

---

### RapidAPI (Booking.com)

Used for:

* Competitor Price Intelligence
* Market Occupancy Estimation
* Price Ceiling Calculation

Features Generated:

* monthly_occupancy_rate
* competition_median_price
* price_ceiling

---

# Feature Engineering

The pipeline automatically generates machine learning features from raw property data.

## Property Features

* property_type
* property_category
* city_tier
* guest_capacity
* bedrooms
* bathrooms
* area

---

## Amenity Features

Binary features generated from text analysis:

* has_wifi
* has_ac_heater
* has_powerbackup
* has_tv
* has_parking
* has_kitchen
* has_pool
* has_gym

---

## Guest Policy Features

* is_couple_friendly
* family_allowed
* bachelors_allowed
* pets_allowed
* events_allowed

---

## Location Features

Cities are automatically:

1. Normalized
2. Geocoded
3. Cached

Generated Features:

* latitude
* longitude
* city_tier

---

## Demand Features

Generated using synthetic search dates:

* checkin_day
* checkin_month
* is_weekend
* is_wedding_season

---

## Weather Features

Generated through OpenWeatherMap:

* historical_avg_temp
* realtime_max_temp
* is_monsoon_season

---

## Event Features

Generated through Ticketmaster:

* is_city_event
* event_type
* distance_to_event

---

## Competitor Features

Generated through Booking.com APIs:

* monthly_occupancy_rate
* competition_median_price
* price_ceiling

---

# Machine Learning Approach

## Current State

The current version focuses on:

1. Data Collection
2. Feature Engineering
3. Dataset Generation

The output is a clean machine-learning-ready dataset.

Example Output:

```text
LightGBM_Ready_Data_YYYYMMDD.xlsx
```

---

## Recommended Model

The project is designed primarily for:

### LightGBM

Why LightGBM?

* Fast training
* Handles mixed feature types
* Works well with tabular pricing data
* Excellent performance on structured datasets

Other supported models:

* XGBoost
* CatBoost
* Random Forest
* Gradient Boosting Regressor

---

# Synthetic Data Usage

## Important Note

The current model training process relies heavily on synthetic demand generation.

Examples include:

* Synthetic booking dates
* Simulated demand patterns
* Estimated occupancy values
* Generated search behavior

This approach was used because historical booking and transaction data were limited during the initial development phase.

---

# Current Limitations

The current model is not trained on large-scale real booking data.

Some demand indicators are estimated through:

* Synthetic date generation
* Simulated demand behavior
* Market approximations

As a result:

* Predictions provide directional guidance.
* Accuracy will improve significantly with real-world booking data.

---

# Future Improvements

As more production data becomes available, the model should be retrained using:

## Real Booking Data

Examples:

* Check-in dates
* Booking dates
* Conversion rates
* Occupancy percentages
* Revenue history
* Cancellation rates

---

## User Behavior Data

Examples:

* Page views
* Search impressions
* Click-through rates
* Booking funnel events

---

## Market Data

Examples:

* Competitor availability
* Dynamic occupancy trends
* Seasonal demand changes

---

# Production Recommendations

When moving to production:

### Store and Reuse

* Category mappings
* Geocoded cities
* API cache responses

---

### Schedule Daily Updates

Recommended:

```bash
0 3 * * * python main.py
```

This refreshes:

* Weather signals
* Event data
* Competitor pricing
* Holiday information

---

### Retrain Model

Recommended retraining frequency:

| Data Volume              | Retraining Frequency |
| ------------------------ | -------------------- |
| < 10,000 bookings        | Monthly              |
| 10,000 - 50,000 bookings | Weekly               |
| > 50,000 bookings        | Daily                |

---

# Environment Variables

Create a `.env` file:

```env
OPENWEATHERMAP_API_KEY=YOUR_KEY

TICKETMASTER_CONSUMER_KEY=YOUR_KEY
TICKETMASTER_SECRET=YOUR_KEY

CALENDARIFIC_API_KEY=YOUR_KEY

RAPIDAPI_KEY=YOUR_KEY
```

---

# Installation

```bash
git clone https://github.com/your-org/dynamic-pricing-model.git

cd dynamic-pricing-model

pip install -r requirements.txt
```

---

# Run

```bash
python main.py
```

Generated output:

```text
LightGBM_Ready_Data_20260729.xlsx
```

---

# Project Goal

The long-term goal is to build a fully automated revenue management system capable of:

* Dynamic property pricing
* Demand forecasting
* Occupancy prediction
* Competitor monitoring
* Revenue optimization

using continuously growing real-world booking data and market intelligence.

---

# Disclaimer

This project is currently in the feature-engineering and data-generation stage. The model relies substantially on synthetic demand signals and estimated market behavior. Performance will improve as more real booking, occupancy, and pricing data are collected and incorporated into future training cycles.
