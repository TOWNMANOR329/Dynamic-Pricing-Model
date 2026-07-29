"""
LightGBM training pipeline for the dynamic pricing time-series dataset.

- Time-based train/test split (sorted by `target_date`).
- Hyperparameters tuned with Optuna.
- Exports trained model file, a metrics report (JSON), feature importance (PNG),
  AND the ONNX production model for the Node.js backend.
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
    sys.exit(
        "❌ lightgbm is not installed. Install it with:\n"
        "    pip install lightgbm --break-system-packages\n"
    )

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    sys.exit(
        "❌ optuna is not installed. Install it with:\n"
        "    pip install optuna --break-system-packages\n"
    )

try:
    import onnxmltools
    from skl2onnx.common.data_types import FloatTensorType
except ImportError:
    print("⚠️ onnxmltools or skl2onnx not found. ONNX conversion will be skipped.")
    print("   Install with: pip install onnxmltools skl2onnx --break-system-packages")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# CONFIG -- adjust these as needed
# ============================================================
TEST_SIZE = 0.2          # fraction of most-recent rows held out as the test set
N_CV_SPLITS = 4           # TimeSeriesSplit folds used during Optuna tuning
N_OPTUNA_TRIALS = 40       # number of Optuna trials
RANDOM_STATE = 42
OUTPUT_DIR = Path(".")

# UPDATED: We now target the dynamic elastic price and the new calendar date
TARGET_COL = "dynamic_target_price"
ID_COL = "property_id"
DATE_COL = "target_date"

CATEGORICAL_COLS = ["city_tier", "property_type", "property_category"]

def find_data_file(cli_arg):
    if cli_arg:
        return Path(cli_arg)
    # UPDATED: Search for the new TimeSeries output file
    candidates = sorted(glob.glob("Airbnb_TimeSeries_Data*.xlsx"), key=lambda p: Path(p).stat().st_mtime)
    if not candidates:
        sys.exit(
            "❌ No Airbnb_TimeSeries_Data*.xlsx file found in the current directory.\n"
            "   Pass the path explicitly: python train_lightgbm.py path/to/file.xlsx"
        )
    return Path(candidates[-1])


def load_and_prepare(path):
    print(f"Loading {path} ...")
    df = pd.read_excel(path)

    if TARGET_COL not in df.columns:
        sys.exit(f"❌ Expected target column '{TARGET_COL}' not found in {path}")
    if DATE_COL not in df.columns:
        sys.exit(f"❌ Expected date column '{DATE_COL}' not found in {path}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # Drop rows with no usable target
    before = len(df)
    df = df[df[TARGET_COL] > 0].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"⚠️ Dropped {dropped} rows with {TARGET_COL} <= 0")

    # CRITICAL UPDATE: Exclude 'base_price' so the model doesn't cheat!
    cols_to_exclude = [TARGET_COL, ID_COL, DATE_COL, 'base_price']
    feature_cols = [c for c in df.columns if c not in cols_to_exclude]

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df, feature_cols


def time_based_split(df, test_size):
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    print(
        f"Time-based split: train={len(train_df)} rows "
        f"({train_df[DATE_COL].min().date()} -> {train_df[DATE_COL].max().date()}), "
        f"test={len(test_df)} rows "
        f"({test_df[DATE_COL].min().date()} -> {test_df[DATE_COL].max().date()})"
    )
    return train_df, test_df

def tune_hyperparameters(train_df, feature_cols, categorical_cols, n_trials):
    """
    Optuna objective evaluated via expanding-window TimeSeriesSplit
    """
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    X = train_df[feature_cols]
    y = train_df[TARGET_COL]

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
        for fold_train_idx, fold_val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[fold_train_idx], X.iloc[fold_val_idx]
            y_tr, y_val = y.iloc[fold_train_idx], y.iloc[fold_val_idx]

            train_set = lgb.Dataset(X_tr, label=y_tr, categorical_feature=categorical_cols, free_raw_data=False)
            val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_cols, reference=train_set, free_raw_data=False)

            model = lgb.train(
                params,
                train_set,
                num_boost_round=2000,
                valid_sets=[val_set],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )
            preds = model.predict(X_val, num_iteration=model.best_iteration)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            fold_scores.append(rmse)

        return float(np.mean(fold_scores))

    print(f"\nRunning Optuna tuning ({n_trials} trials, {N_CV_SPLITS}-fold expanding-window CV)...")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"Best CV RMSE: {study.best_value:.2f}")
    print(f"Best params: {study.best_params}")
    return study.best_params

def train_final_model(train_df, feature_cols, categorical_cols, best_params):
    """
    Trains on the full training portion, using the last 15% as validation.
    """
    val_cutoff = int(len(train_df) * 0.85)
    fit_df = train_df.iloc[:val_cutoff]
    es_val_df = train_df.iloc[val_cutoff:]

    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "seed": RANDOM_STATE,
        **best_params,
    }

    train_set = lgb.Dataset(fit_df[feature_cols], label=fit_df[TARGET_COL], categorical_feature=categorical_cols, free_raw_data=False)
    es_val_set = lgb.Dataset(es_val_df[feature_cols], label=es_val_df[TARGET_COL], categorical_feature=categorical_cols, reference=train_set, free_raw_data=False)

    model = lgb.train(
        params,
        train_set,
        num_boost_round=3000,
        valid_sets=[es_val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=75, verbose=False), lgb.log_evaluation(period=0)],
    )
    print(f"Final model trained. Best iteration: {model.best_iteration}")
    return model

def evaluate(model, test_df, feature_cols):
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]
    preds = model.predict(X_test, num_iteration=model.best_iteration)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = float(np.mean(np.abs((y_test - preds) / y_test.replace(0, np.nan))) * 100)
    r2 = r2_score(y_test, preds)

    metrics = {
        "test_rows": len(test_df),
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "mape_pct": round(mape, 2),
        "r2": round(float(r2), 4),
    }
    return metrics, preds


def plot_feature_importance(model, feature_cols, output_path):
    importance = model.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({"feature": feature_cols, "importance": importance})
    imp_df = imp_df.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(6, len(feature_cols) * 0.28)))
    ax.barh(imp_df["feature"], imp_df["importance"], color="#2E7D32")
    ax.set_xlabel("Gain-based importance")
    ax.set_title("LightGBM Feature Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"💾 Saved feature importance chart to {output_path}")

def main():
    cli_arg = sys.argv[1] if len(sys.argv) > 1 else None
    data_path = find_data_file(cli_arg)

    df, feature_cols = load_and_prepare(data_path)
    categorical_cols = [c for c in CATEGORICAL_COLS if c in feature_cols]

    train_df, test_df = time_based_split(df, TEST_SIZE)

    best_params = tune_hyperparameters(train_df, feature_cols, categorical_cols, N_OPTUNA_TRIALS)
    model = train_final_model(train_df, feature_cols, categorical_cols, best_params)

    metrics, preds = evaluate(model, test_df, feature_cols)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = OUTPUT_DIR / f"lightgbm_pricing_model_{timestamp}.txt"
    metrics_path = OUTPUT_DIR / f"lightgbm_pricing_metrics_{timestamp}.json"
    chart_path = OUTPUT_DIR / f"lightgbm_feature_importance_{timestamp}.png"
    onnx_path = OUTPUT_DIR / f"lightgbm_pricing_model_{timestamp}.onnx"

    model.save_model(str(model_path))
    with open(metrics_path, "w") as f:
        json.dump({"best_params": best_params, "metrics": metrics}, f, indent=2)
    plot_feature_importance(model, feature_cols, chart_path)

    print("\n📦 Converting final LightGBM model to ONNX runtime binary...")
    if 'onnxmltools' in sys.modules:
        try:
            initial_types = [('input', FloatTensorType([None, len(feature_cols)]))]
            onnx_model = onnxmltools.convert_lightgbm(model, initial_types=initial_types, target_opset=15)
            onnxmltools.utils.save_model(onnx_model, str(onnx_path))
            print(f"💾 ONNX Production Model Saved to {onnx_path}")
        except Exception as e:
            print(f"⚠️ Failed to convert to ONNX: {e}")
    else:
        print("⚠️ ONNX skipped (dependencies missing).")

    print("\n================ RESULTS ================")
    print(f"MAE:  {metrics['mae']}")
    print(f"RMSE: {metrics['rmse']}")
    print(f"MAPE: {metrics['mape_pct']}%")
    print(f"R2:   {metrics['r2']}")
    print(f"\n💾 Model saved to {model_path}")
    print(f"💾 Metrics saved to {metrics_path}")
    print(f"💾 Chart saved to {chart_path}")


if __name__ == "__main__":
    main()