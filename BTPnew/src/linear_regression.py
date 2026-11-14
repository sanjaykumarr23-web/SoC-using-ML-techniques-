
"""
Linear Regression for Battery SoC Estimation
--------------------------------------------

Training datasets:
00001, 00003, 00005, 00006, 00007, 00008

Testing datasets:
00009, 00010

Input:
20-sample sliding windows (80 flattened features)

Preprocessing:
StandardScaler fitted on training data only.

Evaluation:
MAE, RMSE, R2
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from .feature_engineering import (
    load_reference_data,
    create_sliding_windows,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

WINDOW_SIZE = 20

TRAIN_DATASETS = [
    "00001", "00003", "00005",
    "00006", "00007", "00008"
]

TEST_DATASETS = [
    "00009", "00010"
]

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("\n" + "=" * 70)
    print("LINEAR REGRESSION SoC ESTIMATION")
    print("=" * 70)

    # Load reference data
    df = load_reference_data()

    df["dataset_id"] = (
        df["dataset_id"]
        .astype(str)
        .str.zfill(5)
    )

    # Dataset-level split
    train_df = df[
        df["dataset_id"].isin(TRAIN_DATASETS)
    ].copy()

    test_df = df[
        df["dataset_id"].isin(TEST_DATASETS)
    ].copy()

    # Create sliding windows
    (
        X_train_seq,
        X_train,
        y_train,
        train_ids
    ) = create_sliding_windows(
        train_df,
        window_size=WINDOW_SIZE
    )

    (
        X_test_seq,
        X_test,
        y_test,
        test_ids
    ) = create_sliding_windows(
        test_df,
        window_size=WINDOW_SIZE
    )

    print("\nTraining data:")
    print("X shape:", X_train.shape)
    print("y shape:", y_train.shape)

    print("\nTesting data:")
    print("X shape:", X_test.shape)
    print("y shape:", y_test.shape)

    # Check dataset separation
    assert set(np.unique(train_ids)).isdisjoint(
        set(np.unique(test_ids))
    ), "Dataset leakage detected!"

    # -----------------------------------------------------
    # Create model pipeline
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING LINEAR REGRESSION")
    print("=" * 70)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])

    print("\nModel configuration:")
    print("Algorithm       : Ordinary Least Squares")
    print("Preprocessing   : StandardScaler")
    print("Intercept       : Enabled")
    print("Window size     : 20")

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    print("\nTraining...")

    start_time = time.time()

    model.fit(
        X_train,
        y_train
    )

    training_time = time.time() - start_time

    print("Training completed.")
    print(f"Training time: {training_time:.4f} seconds")

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    print("\nGenerating predictions...")

    prediction_start = time.time()

    y_pred = model.predict(X_test)

    prediction_time = time.time() - prediction_start

    # Keep SoC within physical limits
    y_pred = np.clip(
        y_pred,
        0.0,
        1.0
    )

    print(f"Prediction time: {prediction_time:.6f} seconds")

    # -----------------------------------------------------
    # Save predictions
    # -----------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = pd.DataFrame({
        "dataset_id": test_ids,
        "true_soc": y_test,
        "predicted_soc": y_pred
    })

    output_path = (
        RESULTS_DIR / "linear_regression_predictions.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print(f"\nPredictions saved: {output_path}")

    # -----------------------------------------------------
    # Overall evaluation
    # -----------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    print("\n" + "=" * 70)
    print("LINEAR REGRESSION OVERALL RESULTS")
    print("=" * 70)

    print(f"MAE  : {mae:.6f} ({mae * 100:.4f}% SoC)")
    print(f"RMSE : {rmse:.6f} ({rmse * 100:.4f}% SoC)")
    print(f"R2   : {r2:.6f}")

    # -----------------------------------------------------
    # Per-dataset evaluation
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("PER-DATASET PERFORMANCE")
    print("=" * 70)

    for dataset_id in np.unique(test_ids):

        mask = test_ids == dataset_id

        actual = y_test[mask]
        predicted = y_pred[mask]

        dataset_mae = mean_absolute_error(
            actual, predicted
        )

        dataset_rmse = np.sqrt(
            mean_squared_error(
                actual, predicted
            )
        )

        dataset_r2 = r2_score(
            actual, predicted
        )

        print(f"\nDataset {dataset_id}")
        print(f"Samples : {mask.sum()}")
        print(f"MAE     : {dataset_mae * 100:.4f}%")
        print(f"RMSE    : {dataset_rmse * 100:.4f}%")
        print(f"R2      : {dataset_r2:.6f}")

    # -----------------------------------------------------
    # Save summary metrics
    # -----------------------------------------------------

    summary = pd.DataFrame([{
        "model": "Linear Regression",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAE_percent": mae * 100,
        "RMSE_percent": rmse * 100,
        "training_time_seconds": training_time,
        "prediction_time_seconds": prediction_time
    }])

    summary.to_csv(
        RESULTS_DIR / "linear_regression_metrics.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("LINEAR REGRESSION EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
