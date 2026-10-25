"""Hybrid Random Forest + XGBoost models for the WQI forecasting pipeline.

Two hybrids are provided, both leakage-safe (fit only on the train partition,
evaluated on validation/test the same way as the rest of the benchmark):

1. Weighted-average blend
   pred = w * RF(x) + (1 - w) * XGB(x)
   The weight w is chosen on the VALIDATION partition (never on test), by a
   simple grid search that minimizes RMSE.

2. Stacked blend (meta-model)
   A small Ridge regressor learns how to combine RF's and XGB's predictions,
   trained on out-of-fold predictions from earlier target-years only, exactly
   like `chronological_oof_predictions` / `stacked_ensemble` already do for
   all four base models in model.py -- this version just restricts the base
   learners to RF + XGB, which is closer to what "RF+XGB hybrid" usually means.

Usage (run from the same directory as model.py, or adjust the import):

    from hybrid_model import run_hybrid_pipeline
    results = run_hybrid_pipeline("data/dataset.csv")
    print_hybrid_results(results)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from model import (
    RANDOM_SEED,
    FEATURE_COLUMNS,
    aggregate_station_years,
    apply_feature_engineering,
    build_forecast_pairs,
    load_dataset,
    make_preprocessor,
    regression_metrics,
    split_by_target_year,
)
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def build_rf_xgb_pair() -> dict[str, object]:
    """Fresh RF + XGBoost instances, matching the hyperparameters in model.py."""
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }


def _fit_pair(train: pd.DataFrame):
    preprocessor = make_preprocessor()
    features = preprocessor.fit_transform(train[FEATURE_COLUMNS])
    target = train["target_wqi"]
    models = build_rf_xgb_pair()
    for model in models.values():
        model.fit(features, target)
    return preprocessor, models


def _predict_pair(preprocessor, models, partition: pd.DataFrame) -> dict[str, np.ndarray]:
    features = preprocessor.transform(partition[FEATURE_COLUMNS])
    return {name: model.predict(features) for name, model in models.items()}


# --------------------------------------------------------------------------
# 1. Weighted-average blend, weight tuned on the validation partition
# --------------------------------------------------------------------------

def best_blend_weight(
    rf_val_pred: np.ndarray, xgb_val_pred: np.ndarray, val_target: pd.Series
) -> float:
    """Grid-search w in [0, 1] that minimizes RMSE on validation for
    pred = w * rf + (1 - w) * xgb. Only ever looks at validation, not test.
    """
    best_w, best_rmse = 0.5, np.inf
    for w in np.linspace(0.0, 1.0, 101):
        blended = w * rf_val_pred + (1 - w) * xgb_val_pred
        rmse = np.sqrt(np.mean((val_target.to_numpy() - blended) ** 2))
        if rmse < best_rmse:
            best_w, best_rmse = w, rmse
    return best_w


# --------------------------------------------------------------------------
# 2. Stacked blend: Ridge meta-model on chronological out-of-fold predictions
# --------------------------------------------------------------------------

def chronological_oof_pair(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Same idea as model.chronological_oof_predictions, restricted to RF+XGB."""
    predictions: list[pd.DataFrame] = []
    targets: list[pd.Series] = []
    years = sorted(train["target_year"].unique())
    for year in years[1:]:
        fold_train = train.loc[train["target_year"] < year]
        fold_validation = train.loc[train["target_year"] == year]
        if fold_train.empty or fold_validation.empty:
            continue
        preprocessor, models = _fit_pair(fold_train)
        predictions.append(pd.DataFrame(_predict_pair(preprocessor, models, fold_validation)))
        targets.append(fold_validation["target_wqi"].reset_index(drop=True))
    if not predictions:
        raise ValueError("Not enough chronological training years for the stacked hybrid.")
    return pd.concat(predictions, ignore_index=True), pd.concat(targets, ignore_index=True)


# --------------------------------------------------------------------------
# End-to-end pipeline
# --------------------------------------------------------------------------

def run_hybrid_pipeline(csv_path: str = "data/dataset.csv") -> dict:
    """Fit both hybrids and return metrics per partition, alongside the
    individual RF and XGB results so you can see whether hybridizing helped.
    """
    station_years = aggregate_station_years(load_dataset(csv_path))
    pairs = build_forecast_pairs(apply_feature_engineering(station_years))
    partitions = split_by_target_year(pairs)
    if any(partition.empty for partition in partitions.values()):
        raise ValueError("Required chronological partitions are empty.")

    preprocessor, models = _fit_pair(partitions["train"])

    # ---- weighted-average blend: tune weight on validation ----
    val_preds = _predict_pair(preprocessor, models, partitions["validation"])
    weight = best_blend_weight(
        val_preds["random_forest"], val_preds["xgboost"], partitions["validation"]["target_wqi"]
    )

    # ---- stacked blend: Ridge meta-model on chronological OOF preds ----
    oof_features, oof_target = chronological_oof_pair(partitions["train"])
    meta_model = Ridge(alpha=10.0).fit(oof_features, oof_target)

    results: dict[str, dict[str, dict[str, float]]] = {}
    for partition_name, partition in partitions.items():
        target = partition["target_wqi"]
        preds = _predict_pair(preprocessor, models, partition)

        preds["weighted_blend"] = weight * preds["random_forest"] + (1 - weight) * preds["xgboost"]

        base_frame = pd.DataFrame(
            {"random_forest": preds["random_forest"], "xgboost": preds["xgboost"]}
        )
        preds["stacked_blend"] = meta_model.predict(base_frame)

        results[partition_name] = {
            model_name: regression_metrics(target, predicted)
            for model_name, predicted in preds.items()
        }

    results["_meta"] = {
        "weighted_blend_weight_on_rf": weight,
        "stacked_blend_coefficients": dict(zip(["random_forest", "xgboost"], meta_model.coef_)),
        "stacked_blend_intercept": meta_model.intercept_,
    }
    return results


def print_hybrid_results(results: dict) -> None:
    """Print metrics in the same compact style as model.print_results."""
    for partition_name, models in results.items():
        if partition_name == "_meta":
            continue
        print(f"\n{partition_name.title()} metrics")
        for model_name, metrics in models.items():
            print(
                f"  {model_name:16} R2={metrics['r2']:.4f}  MAE={metrics['mae']:.4f}  "
                f"RMSE={metrics['rmse']:.4f}  MSE={metrics['mse']:.4f}"
            )
    meta = results["_meta"]
    print(f"\nWeighted blend weight on RF: {meta['weighted_blend_weight_on_rf']:.2f} "
          f"(XGB gets {1 - meta['weighted_blend_weight_on_rf']:.2f})")
    print(f"Stacked blend coefficients: {meta['stacked_blend_coefficients']}, "
          f"intercept={meta['stacked_blend_intercept']:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dataset.csv", help="Path to the CSV dataset.")
    args = parser.parse_args()
    print_hybrid_results(run_hybrid_pipeline(args.data))