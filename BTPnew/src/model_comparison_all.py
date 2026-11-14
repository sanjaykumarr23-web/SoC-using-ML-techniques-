from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_FILES = {
    "Random Forest": "random_forest_predictions.csv",
    "LSTM": "lstm_predictions.csv",
    "CatBoost": "catboost_predictions.csv",
    "Linear Regression": "linear_regression_predictions.csv",
    "SVR": "svr_predictions.csv",
    "GRU": "gru_predictions.csv",
    "MLP": "mlp_predictions.csv",
}


def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAE_percent": mae * 100,
        "RMSE_percent": rmse * 100,
    }


def main():

    print("\n" + "=" * 85)
    print("COMPARATIVE EVALUATION OF ALL SoC ESTIMATION MODELS")
    print("=" * 85)

    overall_results = []
    dataset_results = []

    for model_name, filename in MODEL_FILES.items():

        file_path = RESULTS_DIR / filename

        if not file_path.exists():
            print(f"\nWARNING: Missing file for {model_name}: {filename}")
            continue

        df = pd.read_csv(file_path)

        required_columns = [
            "dataset_id",
            "true_soc",
            "predicted_soc",
        ]

        if not all(col in df.columns for col in required_columns):
            print(f"\nWARNING: Invalid columns in {filename}")
            continue

        actual = df["true_soc"].to_numpy()
        predicted = df["predicted_soc"].to_numpy()

        metrics = calculate_metrics(actual, predicted)

        overall_results.append({
            "Model": model_name,
            "Samples": len(df),
            **metrics,
        })

        for dataset_id, group in df.groupby("dataset_id"):

            dataset_metrics = calculate_metrics(
                group["true_soc"].to_numpy(),
                group["predicted_soc"].to_numpy(),
            )

            dataset_results.append({
                "Model": model_name,
                "Dataset": dataset_id,
                "Samples": len(group),
                **dataset_metrics,
            })

    if not overall_results:
        print("\nNo valid prediction files found.")
        return

    overall_df = pd.DataFrame(overall_results)
    per_dataset_df = pd.DataFrame(dataset_results)

    # Sort by RMSE for the main leaderboard.
    overall_df = overall_df.sort_values("RMSE")

    # Rank each model separately on each metric.
    overall_df["MAE_Rank"] = overall_df["MAE"].rank(
        ascending=True, method="min"
    ).astype(int)

    overall_df["RMSE_Rank"] = overall_df["RMSE"].rank(
        ascending=True, method="min"
    ).astype(int)

    overall_df["R2_Rank"] = overall_df["R2"].rank(
        ascending=False, method="min"
    ).astype(int)

    overall_df["Average_Rank"] = overall_df[
        ["MAE_Rank", "RMSE_Rank", "R2_Rank"]
    ].mean(axis=1)

    overall_df = overall_df.sort_values("Average_Rank")

    # Save comparison tables.
    overall_df.to_csv(
        RESULTS_DIR / "all_models_comparison.csv",
        index=False,
    )

    per_dataset_df.to_csv(
        RESULTS_DIR / "all_models_per_dataset_comparison.csv",
        index=False,
    )

    # Display overall leaderboard.
    print("\n" + "=" * 85)
    print("OVERALL MODEL LEADERBOARD")
    print("=" * 85)

    print(
        overall_df[
            [
                "Model",
                "MAE_percent",
                "RMSE_percent",
                "R2",
                "Average_Rank",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )

    # Display best model for each metric.
    best_mae = overall_df.loc[overall_df["MAE"].idxmin()]
    best_rmse = overall_df.loc[overall_df["RMSE"].idxmin()]
    best_r2 = overall_df.loc[overall_df["R2"].idxmax()]

    print("\n" + "=" * 85)
    print("BEST MODELS BY METRIC")
    print("=" * 85)

    print(f"Lowest MAE  : {best_mae['Model']} ({best_mae['MAE_percent']:.4f}%)")
    print(f"Lowest RMSE : {best_rmse['Model']} ({best_rmse['RMSE_percent']:.4f}%)")
    print(f"Highest R2  : {best_r2['Model']} ({best_r2['R2']:.6f})")

    print("\nBest average rank:", overall_df.iloc[0]["Model"])

    # -------------------------------------------------
    # Plot comparison
    # -------------------------------------------------

    plot_df = overall_df.sort_values("RMSE")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].bar(
        plot_df["Model"],
        plot_df["MAE_percent"],
    )
    axes[0].set_title("Model Comparison - MAE")
    axes[0].set_ylabel("MAE (% SoC)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(
        plot_df["Model"],
        plot_df["RMSE_percent"],
    )
    axes[1].set_title("Model Comparison - RMSE")
    axes[1].set_ylabel("RMSE (% SoC)")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "all_models_error_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    print("\nSaved:")
    print("results/all_models_comparison.csv")
    print("results/all_models_per_dataset_comparison.csv")
    print("results/all_models_error_comparison.png")

    print("\nCOMPARATIVE EVALUATION COMPLETED.")


if __name__ == "__main__":
    main()