"""
Visualization for Battery SoC Estimation
-----------------------------------------

Generates:
1. True vs Predicted SoC
2. SoC tracking over test samples
3. MAE / RMSE model comparison
4. Per-dataset MAE comparison
"""

import os

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = "results"

RF_FILE = os.path.join(
    RESULTS_DIR,
    "random_forest_predictions.csv"
)

LSTM_FILE = os.path.join(
    RESULTS_DIR,
    "lstm_predictions.csv"
)


def main():

    print("\n" + "=" * 70)
    print("BATTERY SoC VISUALIZATION")
    print("=" * 70)

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------
    # Load prediction files
    # --------------------------------------------------

    rf = pd.read_csv(RF_FILE)
    lstm = pd.read_csv(LSTM_FILE)

    print("\nPrediction files loaded.")
    print(f"Random Forest : {len(rf)} samples")
    print(f"LSTM          : {len(lstm)} samples")

    # ==================================================
    # GRAPH 1 — TRUE vs PREDICTED SoC
    # ==================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        rf["true_soc"] * 100,
        label="True SoC",
        linewidth=2
    )

    plt.plot(
        rf["predicted_soc"] * 100,
        label="Random Forest",
        alpha=0.8
    )

    plt.plot(
        lstm["predicted_soc"] * 100,
        label="LSTM",
        alpha=0.8
    )

    plt.xlabel("Test Sample")
    plt.ylabel("SoC (%)")
    plt.title("True vs Predicted Battery SoC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "soc_prediction_comparison.png"
        ),
        dpi=300
    )

    plt.close()

    print(
        "\nSaved:"
        "\n  soc_prediction_comparison.png"
    )

    # ==================================================
    # GRAPH 2 — DATASET 00009
    # ==================================================

    rf_00009 = rf[
        rf["dataset_id"] == 9
    ].reset_index(drop=True)

    lstm_00009 = lstm[
        lstm["dataset_id"] == 9
    ].reset_index(drop=True)

    plt.figure(figsize=(10, 6))

    plt.plot(
        rf_00009["true_soc"] * 100,
        label="Reference SoC",
        linewidth=2
    )

    plt.plot(
        rf_00009["predicted_soc"] * 100,
        label="Random Forest"
    )

    plt.plot(
        lstm_00009["predicted_soc"] * 100,
        label="LSTM"
    )

    plt.xlabel("Test Sample")
    plt.ylabel("SoC (%)")
    plt.title("SoC Estimation — Dataset 00009")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "dataset_00009_soc.png"
        ),
        dpi=300
    )

    plt.close()

    print(
        "  dataset_00009_soc.png"
    )

    # ==================================================
    # GRAPH 3 — DATASET 00010
    # ==================================================

    rf_00010 = rf[
        rf["dataset_id"] == 10
    ].reset_index(drop=True)

    lstm_00010 = lstm[
        lstm["dataset_id"] == 10
    ].reset_index(drop=True)

    plt.figure(figsize=(10, 6))

    plt.plot(
        rf_00010["true_soc"] * 100,
        label="Reference SoC",
        linewidth=2
    )

    plt.plot(
        rf_00010["predicted_soc"] * 100,
        label="Random Forest"
    )

    plt.plot(
        lstm_00010["predicted_soc"] * 100,
        label="LSTM"
    )

    plt.xlabel("Test Sample")
    plt.ylabel("SoC (%)")
    plt.title("SoC Estimation — Dataset 00010")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "dataset_00010_soc.png"
        ),
        dpi=300
    )

    plt.close()

    print(
        "  dataset_00010_soc.png"
    )

    # ==================================================
    # GRAPH 4 — MODEL ERROR COMPARISON
    # ==================================================

    models = [
        "Random Forest",
        "LSTM"
    ]

    mae = [
        0.005175 * 100,
        0.007804 * 100
    ]

    rmse = [
        0.011556 * 100,
        0.009416 * 100
    ]

    x = range(len(models))

    plt.figure(figsize=(8, 6))

    width = 0.35

    plt.bar(
        [i - width / 2 for i in x],
        mae,
        width,
        label="MAE"
    )

    plt.bar(
        [i + width / 2 for i in x],
        rmse,
        width,
        label="RMSE"
    )

    plt.xticks(
        list(x),
        models
    )

    plt.ylabel("Error (%)")
    plt.title("Model Error Comparison")
    plt.legend()
    plt.grid(
        axis="y"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "model_error_comparison.png"
        ),
        dpi=300
    )

    plt.close()

    print(
        "  model_error_comparison.png"
    )

    # ==================================================
    # GRAPH 5 — PER DATASET MAE
    # ==================================================

    rf_mae = []
    lstm_mae = []

    datasets = [
        "00009",
        "00010"
    ]

    for dataset in datasets:

        rf_data = rf[
            rf["dataset_id"] == int(dataset)
        ]

        lstm_data = lstm[
            lstm["dataset_id"] == int(dataset)
        ]

        rf_error = (
            abs(
                rf_data["true_soc"]
                - rf_data["predicted_soc"]
            ).mean()
            * 100
        )

        lstm_error = (
            abs(
                lstm_data["true_soc"]
                - lstm_data["predicted_soc"]
            ).mean()
            * 100
        )

        rf_mae.append(
            rf_error
        )

        lstm_mae.append(
            lstm_error
        )

    x = range(len(datasets))

    plt.figure(figsize=(8, 6))

    width = 0.35

    plt.bar(
        [i - width / 2 for i in x],
        rf_mae,
        width,
        label="Random Forest"
    )

    plt.bar(
        [i + width / 2 for i in x],
        lstm_mae,
        width,
        label="LSTM"
    )

    plt.xticks(
        list(x),
        datasets
    )

    plt.xlabel("Test Dataset")
    plt.ylabel("MAE (%)")
    plt.title("Per-Dataset MAE Comparison")
    plt.legend()
    plt.grid(
        axis="y"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "per_dataset_mae.png"
        ),
        dpi=300
    )

    plt.close()

    print(
        "  per_dataset_mae.png"
    )

    # --------------------------------------------------
    # Complete
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)

    print("\nGraphs saved in:")
    print("  results/")


if __name__ == "__main__":
    main()