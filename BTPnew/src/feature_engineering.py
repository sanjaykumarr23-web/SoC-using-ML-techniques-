"""
Feature Engineering and Sliding Window Creation
------------------------------------------------

Input:
    data/reference_soc.csv

Features:
    voltage_measured
    current_measured
    temperature_measured
    time

Target:
    reference_soc

The same windowed data will later be used for:
    1. Random Forest
    2. LSTM
"""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "reference_soc.csv"
)


FEATURE_COLUMNS = [
    "voltage_measured",
    "current_measured",
    "temperature_measured",
    "time",
]

TARGET_COLUMN = "reference_soc"


def load_reference_data():
    """Load the Coulomb-counting reference dataset."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Reference dataset not found:\n{DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = [
        "dataset_id",
        "cycle_type",
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


def create_sliding_windows(
    df,
    window_size=20,
):
    """
    Create sliding windows separately for every battery dataset.

    This prevents a window from crossing from one battery
    dataset into another.

    Returns
    -------
    X_sequence : ndarray
        Shape: (samples, window_size, features)

    X_flat : ndarray
        Shape: (samples, window_size * features)

    y : ndarray
        Target SoC at the end of each window.

    dataset_ids : ndarray
        Dataset associated with each sample.
    """

    X_sequence = []
    y = []
    dataset_ids = []

    for dataset_id, group in df.groupby(
        "dataset_id",
        sort=True
    ):

        group = (
            group.sort_values("time")
            .reset_index(drop=True)
        )

        features = group[
            FEATURE_COLUMNS
        ].to_numpy(dtype=np.float32)

        target = group[
            TARGET_COLUMN
        ].to_numpy(dtype=np.float32)

        if len(group) <= window_size:
            print(
                f"Skipping {dataset_id}: "
                f"only {len(group)} rows."
            )
            continue

        for i in range(
            window_size,
            len(group)
        ):

            window = features[
                i - window_size:i
            ]

            target_value = target[i]

            X_sequence.append(window)
            y.append(target_value)
            dataset_ids.append(dataset_id)

    X_sequence = np.asarray(
        X_sequence,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.float32
    )

    dataset_ids = np.asarray(
        dataset_ids
    )

    # Random Forest needs 2-D input.
    n_samples = X_sequence.shape[0]
    n_features = X_sequence.shape[1]
    n_channels = X_sequence.shape[2]

    X_flat = X_sequence.reshape(
        n_samples,
        n_features * n_channels
    )

    return (
        X_sequence,
        X_flat,
        y,
        dataset_ids
    )


def print_dataset_distribution(
    dataset_ids,
    y
):
    """Print number of ML samples per dataset."""

    print("\n" + "=" * 70)
    print("WINDOWED DATASET DISTRIBUTION")
    print("=" * 70)

    distribution = []

    for dataset_id in np.unique(
        dataset_ids
    ):

        mask = (
            dataset_ids == dataset_id
        )

        distribution.append({
            "dataset_id": dataset_id,
            "samples": int(mask.sum()),
            "min_soc_%": float(
                y[mask].min() * 100
            ),
            "max_soc_%": float(
                y[mask].max() * 100
            )
        })

    distribution_df = pd.DataFrame(
        distribution
    )

    print(
        distribution_df.to_string(
            index=False
        )
    )

    print("=" * 70)


def main():

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)

    df = load_reference_data()

    print(
        f"\nRaw reference data: "
        f"{df.shape}"
    )

    print(
        f"Datasets: "
        f"{df['dataset_id'].nunique()}"
    )

    # -----------------------------------------------------
    # Create sliding windows
    # -----------------------------------------------------

    WINDOW_SIZE = 20

    print(
        f"\nCreating sliding windows..."
    )

    print(
        f"Window size: {WINDOW_SIZE}"
    )

    (
        X_sequence,
        X_flat,
        y,
        dataset_ids
    ) = create_sliding_windows(
        df,
        window_size=WINDOW_SIZE
    )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING RESULTS")
    print("=" * 70)

    print(
        f"Sequence shape : {X_sequence.shape}"
    )

    print(
        f"Flat shape     : {X_flat.shape}"
    )

    print(
        f"Target shape   : {y.shape}"
    )

    print(
        f"Feature count  : {len(FEATURE_COLUMNS)}"
    )

    print(
        f"Window size    : {WINDOW_SIZE}"
    )

    print(
        f"SoC range      : "
        f"{y.min() * 100:.2f}% - "
        f"{y.max() * 100:.2f}%"
    )

    print_dataset_distribution(
        dataset_ids,
        y
    )

    # -----------------------------------------------------
    # Example
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("EXAMPLE WINDOW")
    print("=" * 70)

    print(
        "\nFeatures:"
    )

    print(
        FEATURE_COLUMNS
    )

    print(
        "\nFirst window:"
    )

    print(
        X_sequence[0]
    )

    print(
        "\nTarget SoC:"
    )

    print(
        f"{y[0] * 100:.2f}%"
    )

    print(
        "\nDataset:"
    )

    print(
        dataset_ids[0]
    )

    print("\nFeature engineering complete.")


if __name__ == "__main__":
    main()