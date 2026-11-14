"""
Battery Capacity Analysis
-------------------------
Estimates discharged capacity from current integration.

For discharge datasets:
    Q(Ah) = integral(|I| dt) / 3600

Time is assumed to be in seconds.
Current is assumed to be in amperes.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from src.data_loader import CSVBatteryDataLoader


def calculate_capacity(df):
    """
    Calculate discharged capacity using trapezoidal integration.

    Parameters
    ----------
    df : pandas.DataFrame
        Battery dataset containing time and current_measured.

    Returns
    -------
    float
        Capacity in Ah.
    """

    df = df.sort_values("time").copy()

    time_seconds = df["time"].to_numpy(dtype=float)
    current = df["current_measured"].to_numpy(dtype=float)

    # For discharge, current is negative.
    # Absolute value gives the amount of charge removed.
    current_abs = np.abs(current)

    charge_coulombs = np.trapezoid(
        current_abs,
        time_seconds
    )

    capacity_ah = charge_coulombs / 3600.0

    return capacity_ah


def main():

    loader = CSVBatteryDataLoader()

    df = loader.load_selected_datasets()

    print("\n" + "=" * 70)
    print("BATTERY CAPACITY ANALYSIS")
    print("=" * 70)

    results = []

    # Only analyze discharge datasets
    discharge_data = df[
        df["cycle_type"] == "discharge"
    ]

    for dataset_id, group in discharge_data.groupby(
        "dataset_id"
    ):

        capacity = calculate_capacity(group)

        results.append({
            "dataset_id": dataset_id,
            "samples": len(group),
            "time_seconds": group["time"].max(),
            "capacity_Ah": capacity
        })

        print(
            f"\nDataset {dataset_id}"
        )

        print(
            f"  Samples       : {len(group)}"
        )

        print(
            f"  Duration      : {group['time'].max():.2f} s"
        )

        print(
            f"  Capacity      : {capacity:.4f} Ah"
        )

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("CAPACITY SUMMARY")
    print("=" * 70)

    print(
        results_df.to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)

    mean_capacity = results_df["capacity_Ah"].mean()
    std_capacity = results_df["capacity_Ah"].std()
    min_capacity = results_df["capacity_Ah"].min()
    max_capacity = results_df["capacity_Ah"].max()

    print(
        f"Mean capacity : {mean_capacity:.4f} Ah"
    )

    print(
        f"Std deviation : {std_capacity:.4f} Ah"
    )

    print(
        f"Minimum       : {min_capacity:.4f} Ah"
    )

    print(
        f"Maximum       : {max_capacity:.4f} Ah"
    )

    print("\n" + "=" * 70)
    print("RECOMMENDED REFERENCE CAPACITY")
    print("=" * 70)

    print(
        f"Recommended Q_nom = {mean_capacity:.4f} Ah"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()