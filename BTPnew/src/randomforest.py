"""
Random Forest Baseline for Battery SoC Estimation
--------------------------------------------------

Uses:
    Voltage
    Current
    Temperature
    Time

with 20-sample sliding windows.

Training datasets:
    00001, 00003, 00005, 00006, 00007, 00008

Testing datasets:
    00009, 00010

Evaluation:
    MAE
    RMSE
    R2
"""

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from .feature_engineering import (
    load_reference_data,
    create_sliding_windows,
)


WINDOW_SIZE = 20

TRAIN_DATASETS = [
    "00001",
    "00003",
    "00005",
    "00006",
    "00007",
    "00008",
]

TEST_DATASETS = [
    "00009",
    "00010",
]


def main():

    print("\n" + "=" * 70)
    print("RANDOM FOREST SoC ESTIMATION")
    print("=" * 70)

    # -----------------------------------------------------
    # Load reference dataset
    # -----------------------------------------------------

    df = load_reference_data()

    df["dataset_id"] = (
        df["dataset_id"]
        .astype(str)
        .str.zfill(5)
    )

    # -----------------------------------------------------
    # Select training and testing datasets
    # -----------------------------------------------------

    train_df = df[
        df["dataset_id"].isin(
            TRAIN_DATASETS
        )
    ].copy()

    test_df = df[
        df["dataset_id"].isin(
            TEST_DATASETS
        )
    ].copy()

    # -----------------------------------------------------
    # Create sliding windows
    # -----------------------------------------------------

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
    print(f"  X shape: {X_train.shape}")
    print(f"  y shape: {y_train.shape}")

    print("\nTesting data:")
    print(f"  X shape: {X_test.shape}")
    print(f"  y shape: {y_test.shape}")

    # -----------------------------------------------------
    # Create Random Forest
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING RANDOM FOREST")
    print("=" * 70)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    print("\nModel configuration:")
    print("  n_estimators     : 200")
    print("  max_depth        : 15")
    print("  min_samples_leaf : 2")
    print("  random_state     : 42")

    print("\nTraining...")

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    print("\nGenerating predictions...")

    y_pred = model.predict(
        X_test
    )

    # Keep predictions physically valid.
    y_pred = np.clip(
        y_pred,
        0.0,
        1.0
    )
       # -----------------------------------------------------
    # Save predictions
    # -----------------------------------------------------

    import os
    import pandas as pd

    os.makedirs("results", exist_ok=True)

    rf_results = pd.DataFrame({
        "dataset_id": test_ids,
        "true_soc": y_test,
        "predicted_soc": y_pred
    })

    rf_results.to_csv(
        "results/random_forest_predictions.csv",
        index=False
    )

    print(
        "\nSaved predictions to "
        "results/random_forest_predictions.csv"
    )
    # -----------------------------------------------------
    # Metrics
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

    # -----------------------------------------------------
    # Overall results
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("RANDOM FOREST RESULTS")
    print("=" * 70)

    print(
        f"\nMAE  : {mae:.6f}"
    )

    print(
        f"RMSE : {rmse:.6f}"
    )

    print(
        f"R²   : {r2:.6f}"
    )

    print(
        f"\nMAE  : {mae * 100:.2f}% SoC"
    )

    print(
        f"RMSE : {rmse * 100:.2f}% SoC"
    )

    # -----------------------------------------------------
    # Per-dataset evaluation
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("PER-DATASET TEST PERFORMANCE")
    print("=" * 70)

    unique_test_ids = np.unique(
        test_ids
    )

    for dataset_id in unique_test_ids:

        mask = (
            test_ids == dataset_id
        )

        dataset_true = y_test[mask]
        dataset_pred = y_pred[mask]

        dataset_mae = (
            mean_absolute_error(
                dataset_true,
                dataset_pred
            )
        )

        dataset_rmse = np.sqrt(
            mean_squared_error(
                dataset_true,
                dataset_pred
            )
        )

        dataset_r2 = r2_score(
            dataset_true,
            dataset_pred
        )

        print(
            f"\nDataset {dataset_id}:"
        )

        print(
            f"  Samples : {mask.sum()}"
        )

        print(
            f"  MAE     : "
            f"{dataset_mae:.6f} "
            f"({dataset_mae * 100:.2f}%)"
        )

        print(
            f"  RMSE    : "
            f"{dataset_rmse:.6f} "
            f"({dataset_rmse * 100:.2f}%)"
        )

        print(
            f"  R²      : "
            f"{dataset_r2:.6f}"
        )

    # -----------------------------------------------------
    # Example predictions
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTIONS")
    print("=" * 70)

    print(
        "\nTrue SoC (%)    Predicted SoC (%)"
    )

    for i in range(
        min(10, len(y_test))
    ):

        print(
            f"{y_test[i] * 100:10.2f}"
            f"    "
            f"{y_pred[i] * 100:15.2f}"
        )

    print("\nRandom Forest evaluation complete.")


if __name__ == "__main__":
    main()