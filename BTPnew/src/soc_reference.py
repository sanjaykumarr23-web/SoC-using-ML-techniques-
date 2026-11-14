"""
Reference SoC Generation using Coulomb Counting
------------------------------------------------

Generates a Coulomb-counting reference SoC profile for
both charge and discharge battery telemetry.

Discharge:
    SoC = 1 - cumulative_removed_charge / total_removed_charge

Charge:
    SoC = cumulative_charged_charge / total_charged_charge

The resulting reference dataset is saved to:

    BTPnew/data/reference_soc.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import CSVBatteryDataLoader


OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "reference_soc.csv"
)


def cumulative_integral(time, current):
    """
    Calculate cumulative current integration using
    the trapezoidal rule.

    Parameters
    ----------
    time : numpy.ndarray
        Time in seconds.

    current : numpy.ndarray
        Current in amperes.

    Returns
    -------
    numpy.ndarray
        Cumulative charge in Ah.
    """

    time = np.asarray(time, dtype=float)
    current = np.asarray(current, dtype=float)

    cumulative_coulombs = np.zeros(len(time))

    for i in range(1, len(time)):

        dt = time[i] - time[i - 1]

        cumulative_coulombs[i] = (
            cumulative_coulombs[i - 1]
            + 0.5
            * (
                current[i]
                + current[i - 1]
            )
            * dt
        )

    return cumulative_coulombs / 3600.0


def generate_discharge_soc(df):
    """
    Generate reference SoC for a discharge record.

    Negative current represents battery discharge.
    """

    df = (
        df.sort_values("time")
        .reset_index(drop=True)
        .copy()
    )

    time = df["time"].to_numpy(dtype=float)
    current = df["current_measured"].to_numpy(dtype=float)

    # Only discharge current contributes to removed charge.
    discharge_current = np.maximum(-current, 0.0)

    removed_capacity = cumulative_integral(
        time,
        discharge_current
    )

    total_capacity = removed_capacity[-1]

    if total_capacity <= 0:
        raise ValueError(
            "Invalid discharge capacity."
        )

    soc = (
        1.0
        - removed_capacity / total_capacity
    )

    soc = np.clip(soc, 0.0, 1.0)

    df["capacity_change_Ah"] = -removed_capacity

    df["reference_soc"] = soc

    return df


def generate_charge_soc(df):
    """
    Generate reference SoC for a charge record.

    Positive current represents battery charging.
    """

    df = (
        df.sort_values("time")
        .reset_index(drop=True)
        .copy()
    )

    time = df["time"].to_numpy(dtype=float)
    current = df["current_measured"].to_numpy(dtype=float)

    # Only positive current contributes to charging.
    charge_current = np.maximum(current, 0.0)

    charged_capacity = cumulative_integral(
        time,
        charge_current
    )

    total_capacity = charged_capacity[-1]

    if total_capacity <= 0:
        raise ValueError(
            "Invalid charge capacity."
        )

    soc = (
        charged_capacity / total_capacity
    )

    soc = np.clip(soc, 0.0, 1.0)

    df["capacity_change_Ah"] = charged_capacity

    df["reference_soc"] = soc

    return df


def generate_reference_soc(df):
    """
    Generate reference SoC based on cycle type.
    """

    all_results = []

    for dataset_id, group in df.groupby(
        "dataset_id"
    ):

        cycle_type = group[
            "cycle_type"
        ].iloc[0]

        if cycle_type == "discharge":

            result = generate_discharge_soc(
                group
            )

        elif cycle_type == "charge":

            result = generate_charge_soc(
                group
            )

        else:
            continue

        all_results.append(result)

    if not all_results:
        raise ValueError(
            "No valid charge/discharge datasets found."
        )

    return pd.concat(
        all_results,
        ignore_index=True
    )


def main():

    loader = CSVBatteryDataLoader()

    df = loader.load_selected_datasets()

    print("\n" + "=" * 70)
    print("COULOMB-COUNTING REFERENCE SOC")
    print("=" * 70)

    reference_df = generate_reference_soc(df)

    print("\nDataset results:")

    summary = []

    for dataset_id, group in reference_df.groupby(
        "dataset_id"
    ):

        cycle_type = group[
            "cycle_type"
        ].iloc[0]

        capacity = abs(
            group[
                "capacity_change_Ah"
            ].iloc[-1]
        )

        initial_soc = (
            group["reference_soc"].iloc[0]
            * 100
        )

        final_soc = (
            group["reference_soc"].iloc[-1]
            * 100
        )

        summary.append({
            "dataset_id": dataset_id,
            "type": cycle_type,
            "samples": len(group),
            "capacity_Ah": capacity,
            "initial_SoC_%": initial_soc,
            "final_SoC_%": final_soc
        })

    summary_df = pd.DataFrame(summary)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("CHECKING ALL 8 DATASETS")
    print("=" * 70)

    print(
        f"Datasets processed : "
        f"{reference_df['dataset_id'].nunique()}"
    )

    print(
        f"Total samples      : "
        f"{len(reference_df):,}"
    )

    print(
        f"SoC minimum        : "
        f"{reference_df['reference_soc'].min() * 100:.2f}%"
    )

    print(
        f"SoC maximum        : "
        f"{reference_df['reference_soc'].max() * 100:.2f}%"
    )

    # -----------------------------------------------------
    # Save reference dataset
    # -----------------------------------------------------

    reference_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("REFERENCE DATASET SAVED")
    print("=" * 70)

    print(
        f"File: {OUTPUT_FILE}"
    )

    print(
        f"Rows: {len(reference_df):,}"
    )

    print("\nReference SoC generation complete.")


if __name__ == "__main__":
    main()