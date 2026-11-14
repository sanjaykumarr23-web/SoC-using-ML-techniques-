"""
Final Evaluation of Battery SoC Estimation Models
--------------------------------------------------

This script DOES NOT train any model.

It evaluates the already generated prediction files:

    results/random_forest_predictions.csv
    results/lstm_predictions.csv

Metrics:
    MAE
    RMSE
    R²

Also generates:
    results/final_evaluation.csv
    results/final_per_dataset_evaluation.csv
"""


import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


RESULTS_DIR = "results"

RF_FILE = os.path.join(
    RESULTS_DIR,
    "random_forest_predictions.csv"
)

LSTM_FILE = os.path.join(
    RESULTS_DIR,
    "lstm_predictions.csv"
)


def calculate_metrics(
    y_true,
    y_pred
):
    """
    Calculate MAE, RMSE and R².
    """

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return mae, rmse, r2


def evaluate_model(
    df,
    model_name
):
    """
    Calculate overall model performance.
    """

    y_true = df["true_soc"].values
    y_pred = df["predicted_soc"].values

    mae, rmse, r2 = calculate_metrics(
        y_true,
        y_pred
    )

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAE_%": mae * 100,
        "RMSE_%": rmse * 100,
    }


def main():

    print("\n" + "=" * 70)
    print("FINAL SoC MODEL EVALUATION")
    print("=" * 70)

    # -----------------------------------------------------
    # Check prediction files
    # -----------------------------------------------------

    if not os.path.exists(RF_FILE):

        print(
            f"\nERROR: File not found:"
            f"\n{RF_FILE}"
        )

        return

    if not os.path.exists(LSTM_FILE):

        print(
            f"\nERROR: File not found:"
            f"\n{LSTM_FILE}"
        )

        return

    # -----------------------------------------------------
    # Load predictions
    # -----------------------------------------------------

    rf = pd.read_csv(
        RF_FILE
    )

    lstm = pd.read_csv(
        LSTM_FILE
    )

    print("\nPrediction files loaded.")

    print(
        f"Random Forest predictions : {len(rf)}"
    )

    print(
        f"LSTM predictions          : {len(lstm)}"
    )

    # -----------------------------------------------------
    # Validate columns
    # -----------------------------------------------------

    required_columns = [
        "dataset_id",
        "true_soc",
        "predicted_soc",
    ]

    for column in required_columns:

        if column not in rf.columns:

            print(
                f"\nERROR: RF missing column:"
                f" {column}"
            )

            return

        if column not in lstm.columns:

            print(
                f"\nERROR: LSTM missing column:"
                f" {column}"
            )

            return

    # -----------------------------------------------------
    # Dataset information
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST DATASET INFORMATION")
    print("=" * 70)

    print(
        "\nRandom Forest datasets:"
    )

    print(
        sorted(
            rf["dataset_id"]
            .unique()
            .tolist()
        )
    )

    print(
        "\nLSTM datasets:"
    )

    print(
        sorted(
            lstm["dataset_id"]
            .unique()
            .tolist()
        )
    )

    # -----------------------------------------------------
    # Overall evaluation
    # -----------------------------------------------------

    rf_result = evaluate_model(
        rf,
        "Random Forest"
    )

    lstm_result = evaluate_model(
        lstm,
        "LSTM"
    )

    comparison = pd.DataFrame(
        [
            rf_result,
            lstm_result
        ]
    )

    print("\n" + "=" * 70)
    print("OVERALL MODEL PERFORMANCE")
    print("=" * 70)

    print(
        comparison[
            [
                "Model",
                "MAE",
                "RMSE",
                "R2",
                "MAE_%",
                "RMSE_%"
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # -----------------------------------------------------
    # Percentage error summary
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("SoC ERROR IN PERCENTAGE")
    print("=" * 70)

    print(
        "\nRandom Forest:"
    )

    print(
        f"  MAE  : "
        f"{rf_result['MAE_%']:.2f}%"
    )

    print(
        f"  RMSE : "
        f"{rf_result['RMSE_%']:.2f}%"
    )

    print(
        f"  R²   : "
        f"{rf_result['R2']:.6f}"
    )

    print(
        "\nLSTM:"
    )

    print(
        f"  MAE  : "
        f"{lstm_result['MAE_%']:.2f}%"
    )

    print(
        f"  RMSE : "
        f"{lstm_result['RMSE_%']:.2f}%"
    )

    print(
        f"  R²   : "
        f"{lstm_result['R2']:.6f}"
    )

    # -----------------------------------------------------
    # Best models
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL SELECTION")
    print("=" * 70)

    if rf_result["MAE"] < lstm_result["MAE"]:

        print(
            "\nBest MAE  : Random Forest"
        )

    else:

        print(
            "\nBest MAE  : LSTM"
        )

    if rf_result["RMSE"] < lstm_result["RMSE"]:

        print(
            "Best RMSE : Random Forest"
        )

    else:

        print(
            "Best RMSE : LSTM"
        )

    if rf_result["R2"] > lstm_result["R2"]:

        print(
            "Best R²   : Random Forest"
        )

    else:

        print(
            "Best R²   : LSTM"
        )

    # -----------------------------------------------------
    # Per-dataset evaluation
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("PER-DATASET EVALUATION")
    print("=" * 70)

    dataset_results = []

    all_datasets = sorted(
        set(
            rf["dataset_id"].unique()
        ).union(
            set(
                lstm["dataset_id"].unique()
            )
        )
    )

    for dataset_id in all_datasets:

        # -----------------------------
        # Random Forest
        # -----------------------------

        rf_data = rf[
            rf["dataset_id"] == dataset_id
        ]

        if len(rf_data) > 0:

            rf_mae, rf_rmse, rf_r2 = (
                calculate_metrics(
                    rf_data["true_soc"].values,
                    rf_data["predicted_soc"].values
                )
            )

        else:

            rf_mae = np.nan
            rf_rmse = np.nan
            rf_r2 = np.nan

        # -----------------------------
        # LSTM
        # -----------------------------

        lstm_data = lstm[
            lstm["dataset_id"] == dataset_id
        ]

        if len(lstm_data) > 0:

            lstm_mae, lstm_rmse, lstm_r2 = (
                calculate_metrics(
                    lstm_data["true_soc"].values,
                    lstm_data["predicted_soc"].values
                )
            )

        else:

            lstm_mae = np.nan
            lstm_rmse = np.nan
            lstm_r2 = np.nan

        dataset_results.append(
            {
                "dataset_id": dataset_id,

                "RF_MAE_%":
                    rf_mae * 100,

                "RF_RMSE_%":
                    rf_rmse * 100,

                "RF_R2":
                    rf_r2,

                "LSTM_MAE_%":
                    lstm_mae * 100,

                "LSTM_RMSE_%":
                    lstm_rmse * 100,

                "LSTM_R2":
                    lstm_r2,
            }
        )

    per_dataset = pd.DataFrame(
        dataset_results
    )

    print(
        per_dataset.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # -----------------------------------------------------
    # Save final evaluation
    # -----------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    comparison.to_csv(
        os.path.join(
            RESULTS_DIR,
            "final_evaluation.csv"
        ),
        index=False
    )

    per_dataset.to_csv(
        os.path.join(
            RESULTS_DIR,
            "final_per_dataset_evaluation.csv"
        ),
        index=False
    )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        "\nresults/final_evaluation.csv"
    )

    print(
        "results/final_per_dataset_evaluation.csv"
    )

    print("\n" + "=" * 70)
    print("FINAL EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()