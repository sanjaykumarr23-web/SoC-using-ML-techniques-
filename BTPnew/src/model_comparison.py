"""
Model Comparison for Battery SoC Estimation
--------------------------------------------

Compares:
    Random Forest
    LSTM

Using predictions generated on the same test datasets.

Evaluation:
    MAE
    RMSE
    R²
"""

import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


RF_FILE = "results/random_forest_predictions.csv"
LSTM_FILE = "results/lstm_predictions.csv"


def calculate_metrics(true_soc, predicted_soc):

    mae = mean_absolute_error(
        true_soc,
        predicted_soc
    )

    rmse = np.sqrt(
        mean_squared_error(
            true_soc,
            predicted_soc
        )
    )

    r2 = r2_score(
        true_soc,
        predicted_soc
    )

    return mae, rmse, r2


def main():

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    # --------------------------------------------------
    # Load predictions
    # --------------------------------------------------

    rf = pd.read_csv(RF_FILE)
    lstm = pd.read_csv(LSTM_FILE)

    print("\nPrediction files loaded.")

    print(
        f"Random Forest predictions : {len(rf)}"
    )

    print(
        f"LSTM predictions          : {len(lstm)}"
    )

    # --------------------------------------------------
    # Check datasets
    # --------------------------------------------------

    print("\nTest datasets:")

    print(
        "Random Forest:",
        sorted(rf["dataset_id"].unique())
    )

    print(
        "LSTM:",
        sorted(lstm["dataset_id"].unique())
    )

    # --------------------------------------------------
    # Overall metrics
    # --------------------------------------------------

    rf_mae, rf_rmse, rf_r2 = calculate_metrics(
        rf["true_soc"],
        rf["predicted_soc"]
    )

    lstm_mae, lstm_rmse, lstm_r2 = calculate_metrics(
        lstm["true_soc"],
        lstm["predicted_soc"]
    )

    # --------------------------------------------------
    # Comparison table
    # --------------------------------------------------

    comparison = pd.DataFrame({

        "Model": [
            "Random Forest",
            "LSTM"
        ],

        "MAE": [
            rf_mae,
            lstm_mae
        ],

        "RMSE": [
            rf_rmse,
            lstm_rmse
        ],

        "R2": [
            rf_r2,
            lstm_r2
        ]
    })

    print("\n" + "=" * 70)
    print("OVERALL MODEL COMPARISON")
    print("=" * 70)

    print(
        comparison.to_string(
            index=False,
            formatters={
                "MAE": "{:.6f}".format,
                "RMSE": "{:.6f}".format,
                "R2": "{:.6f}".format,
            }
        )
    )

    # --------------------------------------------------
    # Percentage representation
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("ERROR IN SoC PERCENTAGE")
    print("=" * 70)

    print(
        f"\nRandom Forest:"
        f"\n  MAE  : {rf_mae * 100:.2f}%"
        f"\n  RMSE : {rf_rmse * 100:.2f}%"
    )

    print(
        f"\nLSTM:"
        f"\n  MAE  : {lstm_mae * 100:.2f}%"
        f"\n  RMSE : {lstm_rmse * 100:.2f}%"
    )

    # --------------------------------------------------
    # Determine best model
    # --------------------------------------------------

    if rf_mae < lstm_mae:
        best_mae = "Random Forest"
    else:
        best_mae = "LSTM"

    if rf_rmse < lstm_rmse:
        best_rmse = "Random Forest"
    else:
        best_rmse = "LSTM"

    if rf_r2 > lstm_r2:
        best_r2 = "Random Forest"
    else:
        best_r2 = "LSTM"

    print("\n" + "=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(
        f"\nBest MAE  : {best_mae}"
    )

    print(
        f"Best RMSE : {best_rmse}"
    )

    print(
        f"Best R²   : {best_r2}"
    )

    # --------------------------------------------------
    # Per-dataset comparison
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("PER-DATASET COMPARISON")
    print("=" * 70)

    datasets = sorted(
        set(rf["dataset_id"])
        & set(lstm["dataset_id"])
    )

    rows = []

    for dataset_id in datasets:

        rf_data = rf[
            rf["dataset_id"] == dataset_id
        ]

        lstm_data = lstm[
            lstm["dataset_id"] == dataset_id
        ]

        rf_mae_d, rf_rmse_d, rf_r2_d = calculate_metrics(
            rf_data["true_soc"],
            rf_data["predicted_soc"]
        )

        lstm_mae_d, lstm_rmse_d, lstm_r2_d = calculate_metrics(
            lstm_data["true_soc"],
            lstm_data["predicted_soc"]
        )

        rows.append({

            "dataset_id": dataset_id,

            "RF_MAE_%": rf_mae_d * 100,
            "RF_RMSE_%": rf_rmse_d * 100,
            "RF_R2": rf_r2_d,

            "LSTM_MAE_%": lstm_mae_d * 100,
            "LSTM_RMSE_%": lstm_rmse_d * 100,
            "LSTM_R2": lstm_r2_d,
        })

    dataset_comparison = pd.DataFrame(rows)

    print(
        dataset_comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    comparison.to_csv(
        "results/model_comparison.csv",
        index=False
    )

    dataset_comparison.to_csv(
        "results/per_dataset_comparison.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(
        "\nSaved:"
        "\n  results/model_comparison.csv"
        "\n  results/per_dataset_comparison.csv"
    )

    print(
        "\nModel comparison complete."
    )


if __name__ == "__main__":
    main()