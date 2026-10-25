"""Run the training pipeline and persist the Random-Forest + preprocessor.

Usage:
    python save_models.py [--data data/dataset.csv]

Outputs:
    models/rf.joblib
    models/preprocessor.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from src.model import (
    FEATURE_COLUMNS,
    aggregate_station_years,
    apply_feature_engineering,
    build_forecast_pairs,
    build_models,
    load_dataset,
    make_preprocessor,
    split_by_target_year,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dataset.csv")
    args = parser.parse_args()

    print("Loading and processing dataset...")
    station_years = aggregate_station_years(load_dataset(args.data))
    pairs = build_forecast_pairs(apply_feature_engineering(station_years))
    train = split_by_target_year(pairs)["train"]

    print(f"Training on {len(train)} samples...")
    preprocessor = make_preprocessor()
    X_train = preprocessor.fit_transform(train[FEATURE_COLUMNS])
    y_train = train["target_wqi"]

    rf = build_models()["random_forest"]
    rf.fit(X_train, y_train)

    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)

    joblib.dump(rf, out_dir / "rf.joblib")
    joblib.dump(preprocessor, out_dir / "preprocessor.joblib")
    print(f"Saved  models/rf.joblib  and  models/preprocessor.joblib")


if __name__ == "__main__":
    main()
