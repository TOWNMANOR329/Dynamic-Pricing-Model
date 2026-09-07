"""
Stage 1: Base price model.

Trains on REAL data only:
  - target_price: each property's actual current listed price (real).
  - Static/intrinsic features: things that don't change day-to-day
    (location, size, amenities, guest capacity, nearby conveniences).

Deliberately EXCLUDES every date-driven feature (checkin_day/month,
is_weekend, is_major_festival, weather, event, competitor/occupancy
signals) -- those have no real relationship to a property's current
fixed price, and mixing them in is what diluted the original model's
accuracy. They belong in the Stage 2 dynamic_multiplier.py layer
instead, applied on top of this model's output.

Because there's no genuine date-driven signal in this feature set (a
property's base price doesn't logically vary with today vs. next
week), validation uses a plain K-fold split rather than a time-based
one -- there's no real temporal structure here to hold out against.

Usage:
    python train_base_price_model.py [path_to_excel_file]
"""

import sys
import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    sys.exit("❌ lightgbm is not installed. Install it with:\n    pip install lightgbm --break-system-packages\n")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    sys.exit("❌ optuna is not installed. Install it with:\n    pip install optuna --break-system-packages\n")

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================
TEST_SIZE = 0.2
N_CV_SPLITS = 5
N_OPTUNA_TRIALS = 40
RANDOM_STATE = 42
OUTPUT_DIR = Path(".")
ONNX_OPSET = 13

TARGET_COL = "target_price"
ID_COL = "property_id"

# Everything NOT in this list, other than TARGET_COL/ID_COL, is
# assumed to be a dynamic/date-driven column and is dropped here --
# see dynamic_multiplier.py for how those are used instead.
STATIC_FEATURE_COLS = [
    "city_tier", "property_type", "property_category", "guest_capacity",
    "has_powerbackup", "has_ac_heater", "has_wifi", "has_tv", "has_parking",
    "has_kitchen", "has_pool", "has_gym",
    "is_couple_friendly", "pets_allowed", "events_allowed", "family_allowed", "bachelors_allowed",
    "area", "bedrooms", "bathrooms", "latitude", "longitude",
    "taxi_nearby", "parking_nearby", "restaurants_nearby", "grocery_nearby",
]
CATEGORICAL_COLS = ["city_tier", "property_type", "property_category"]
CATEGORY_MAPPING_FILE = "category_mappings.json"


def find_data_file(cli_arg):
    if cli_arg:
        return Path(cli_arg)
    candidates = sorted(glob.glob("LightGBM_Ready_Data_*.xlsx"), key=lambda p: Path(p).stat().st_mtime)
    if not candidates:
        sys.exit("❌ No LightGBM_Ready_Data_*.xlsx file found. Pass the path explicitly.")
    return Path(candidates[-1])


def load_and_prepare(path):
    print(f"Loading {path} ...")
    df = pd.read_excel(path)

    if TARGET_COL not in df.columns:
        sys.exit(f"❌ Expected target column '{TARGET_COL}' not found in {path}")

    missing = [c for c in STATIC_FEATURE_COLS if c not in df.columns]
    if missing:
        sys.exit(f"❌ Missing expected static feature columns: {missing}")

    before = len(df)
    df = df[df[TARGET_COL] > 0].reset_index(drop=True)
    if before - len(df):
        print(f"⚠️ Dropped {before - len(df)} rows with target_price <= 0")

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    return df


def tune_hyperparameters(train_df, n_trials):
    X = train_df[STATIC_FEATURE_COLS]
    y = train_df[TARGET_COL]
    kf = KFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "seed": RANDOM_STATE,
        }

        fold_scores = []
        for fold_train_idx, fold_val_idx in kf.split(X):
            X_tr, X_val = X.iloc[fold_train_idx], X.iloc[fold_val_idx]
            y_tr, y_val = y.iloc[fold_train_idx], y.iloc[fold_val_idx]

            train_set = lgb.Dataset(X_tr, label=y_tr, categorical_feature=CATEGORICAL_COLS, free_raw_data=False)
            val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=CATEGORICAL_COLS, reference=train_set, free_raw_data=False)

            model = lgb.train(
                params, train_set, num_boost_round=2000, valid_sets=[val_set],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )
            preds = model.predict(X_val, num_iteration=model.best_iteration)
            fold_scores.append(np.sqrt(mean_squared_error(y_val, preds)))

        return float(np.mean(fold_scores))

    print(f"\nRunning Optuna tuning ({n_trials} trials, {N_CV_SPLITS}-fold CV)...")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"Best CV RMSE: {study.best_value:.2f}")
    print(f"Best params: {study.best_params}")
    return study.best_params


def train_final_model(train_df, best_params):
    fit_df, es_val_df = train_test_split(train_df, test_size=0.15, random_state=RANDOM_STATE)

    params = {"objective": "regression", "metric": "rmse", "verbosity": -1, "boosting_type": "gbdt", "seed": RANDOM_STATE, **best_params}

    train_set = lgb.Dataset(fit_df[STATIC_FEATURE_COLS], label=fit_df[TARGET_COL], categorical_feature=CATEGORICAL_COLS, free_raw_data=False)
    es_val_set = lgb.Dataset(es_val_df[STATIC_FEATURE_COLS], label=es_val_df[TARGET_COL], categorical_feature=CATEGORICAL_COLS, reference=train_set, free_raw_data=False)

    model = lgb.train(
        params, train_set, num_boost_round=3000, valid_sets=[es_val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=75, verbose=False), lgb.log_evaluation(period=0)],
    )
    print(f"Final model trained. Best iteration: {model.best_iteration}")
    return model


def finalize_model_to_best_iteration(model, output_dir, timestamp):
    """Truncates the booster to best_iteration so the saved file, ONNX
    export, and scoring are all consistent with what early stopping
    actually picked (see the longer comment in earlier versions of
    this pipeline for why this step matters)."""
    model_path = output_dir / f"base_price_model_{timestamp}.txt"
    model.save_model(str(model_path), num_iteration=model.best_iteration)
    return lgb.Booster(model_file=str(model_path)), model_path


def evaluate(model, test_df):
    X_test = test_df[STATIC_FEATURE_COLS]
    y_test = test_df[TARGET_COL]
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = float(np.mean(np.abs((y_test - preds) / y_test.replace(0, np.nan))) * 100)
    r2 = r2_score(y_test, preds)

    return {"test_rows": len(test_df), "mae": round(float(mae), 2), "rmse": round(float(rmse), 2),
            "mape_pct": round(mape, 2), "r2": round(float(r2), 4)}


def plot_feature_importance(model, output_path):
    importance = model.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({"feature": STATIC_FEATURE_COLS, "importance": importance}).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(6, len(STATIC_FEATURE_COLS) * 0.3)))
    ax.barh(imp_df["feature"], imp_df["importance"], color="#1565C0")
    ax.set_xlabel("Gain-based importance")
    ax.set_title("Base Price Model — Feature Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"💾 Saved feature importance chart to {output_path}")


def export_onnx(model, output_path):
    try:
        from onnxmltools.convert import convert_lightgbm
        from onnxconverter_common.data_types import FloatTensorType
        import onnx
    except ImportError:
        print(
            "\n⚠️ Skipping ONNX export -- required packages not installed.\n"
            "   pip install onnxmltools onnxconverter-common onnx --break-system-packages\n"
        )
        return None
    initial_type = [("float_input", FloatTensorType([None, len(STATIC_FEATURE_COLS)]))]
    onnx_model = convert_lightgbm(model, initial_types=initial_type, target_opset=ONNX_OPSET)
    onnx.save_model(onnx_model, str(output_path))
    print(f"💾 Saved ONNX model to {output_path}")
    return output_path


def save_feature_schema(output_path):
    category_mappings = {}
    if Path(CATEGORY_MAPPING_FILE).exists():
        with open(CATEGORY_MAPPING_FILE) as f:
            all_mappings = json.load(f)
        category_mappings = {c: all_mappings.get(c, {}) for c in CATEGORICAL_COLS}

    schema = {
        "model_stage": "base_price (Stage 1 of 2 -- combine with dynamic_multiplier.py for the final recommendation)",
        "onnx_input_tensor_name": "float_input",
        "feature_order": STATIC_FEATURE_COLS,
        "categorical_features": CATEGORICAL_COLS,
        "categorical_value_to_code": category_mappings,
        "target_column_meaning": "current listed price (this model predicts a fair BASE price, not a date-adjusted one)",
        "note": (
            "Build the input vector in EXACTLY this feature_order, as float32. "
            "For categorical_features, map the raw string to its code via "
            "categorical_value_to_code (case-insensitive, matches "
            "dynamic_pricing.py's encode_categorical_column). This model's output "
            "is the BASE price -- feed it into dynamic_multiplier.py's "
            "recommend_price() along with the property's dynamic/date-driven "
            "fields to get the final bounded recommendation."
        ),
    }
    with open(output_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"💾 Saved feature schema to {output_path}")


def main():
    cli_arg = sys.argv[1] if len(sys.argv) > 1 else None
    data_path = find_data_file(cli_arg)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    df = load_and_prepare(data_path)
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"Split: train={len(train_df)} rows, test={len(test_df)} rows")

    best_params = tune_hyperparameters(train_df, N_OPTUNA_TRIALS)
    model = train_final_model(train_df, best_params)
    model, model_path = finalize_model_to_best_iteration(model, OUTPUT_DIR, timestamp)

    metrics = evaluate(model, test_df)
    metrics_path = OUTPUT_DIR / f"base_price_metrics_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump({"best_params": best_params, "metrics": metrics}, f, indent=2)

    chart_path = OUTPUT_DIR / f"base_price_feature_importance_{timestamp}.png"
    plot_feature_importance(model, chart_path)

    onnx_path = OUTPUT_DIR / f"base_price_model_{timestamp}.onnx"
    export_onnx(model, onnx_path)

    schema_path = OUTPUT_DIR / f"base_price_feature_schema_{timestamp}.json"
    save_feature_schema(schema_path)

    print("\n================ BASE PRICE MODEL RESULTS ================")
    print(f"MAE:  {metrics['mae']}")
    print(f"RMSE: {metrics['rmse']}")
    print(f"MAPE: {metrics['mape_pct']}%")
    print(f"R2:   {metrics['r2']}")
    print(f"\n💾 Model (native): {model_path}")
    print(f"💾 Model (ONNX):   {onnx_path if onnx_path and onnx_path.exists() else 'SKIPPED — see warning above'}")
    print(f"💾 Feature schema: {schema_path}")
    print(f"💾 Metrics:        {metrics_path}")
    print(f"💾 Chart:          {chart_path}")
    print(
        "\nNext step: combine this model's predictions with dynamic_multiplier.py "
        "(Stage 2, rule-based) to get the final bounded price recommendation per property."
    )


if __name__ == "__main__":
    main()
