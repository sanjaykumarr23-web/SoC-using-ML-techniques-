"""
Dataset-Level Train/Test Split
------------------------------

Creates a leakage-safe split where complete battery
datasets are assigned to either training or testing.

Training datasets:
    00001, 00003, 00005, 00006, 00007, 00008

Testing datasets:
    00009, 00010
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_engineering import (
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
    print("DATASET-LEVEL TRAIN / TEST SPLIT")
    print("=" * 70)

    df = load_reference_data()

    # Make dataset IDs strings so they match our lists.
    df["dataset_id"] = (
        df["dataset_id"]
        .astype(str)
        .str.zfill(5)
    )

    print(
        f"\nTotal raw samples: {len(df):,}"
    )

    print(
        f"Total datasets: "
        f"{df['dataset_id'].nunique()}"
    )

    # -----------------------------------------------------
    # Check requested datasets
    # -----------------------------------------------------

    available = set(
        df["dataset_id"].unique()
    )

    requested = set(
        TRAIN_DATASETS + TEST_DATASETS
    )

    missing = requested - available

    if missing:

        raise ValueError(
            f"Datasets not found: {missing}"
        )

    # -----------------------------------------------------
    # Separate raw data
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
    # Create windows independently
    # -----------------------------------------------------

    (
        X_train_seq,
        X_train_flat,
        y_train,
        train_ids
    ) = create_sliding_windows(
        train_df,
        window_size=WINDOW_SIZE
    )

    (
        X_test_seq,
        X_test_flat,
        y_test,
        test_ids
    ) = create_sliding_windows(
        test_df,
        window_size=WINDOW_SIZE
    )

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING DATA")
    print("=" * 70)

    print(
        f"Datasets: {TRAIN_DATASETS}"
    )

    print(
        f"Raw samples: "
        f"{len(train_df):,}"
    )

    print(
        f"Sequence shape: "
        f"{X_train_seq.shape}"
    )

    print(
        f"Random Forest shape: "
        f"{X_train_flat.shape}"
    )

    print(
        f"Target shape: "
        f"{y_train.shape}"
    )

    print("\n" + "=" * 70)
    print("TEST DATA")
    print("=" * 70)

    print(
        f"Datasets: {TEST_DATASETS}"
    )

    print(
        f"Raw samples: "
        f"{len(test_df):,}"
    )

    print(
        f"Sequence shape: "
        f"{X_test_seq.shape}"
    )

    print(
        f"Random Forest shape: "
        f"{X_test_flat.shape}"
    )

    print(
        f"Target shape: "
        f"{y_test.shape}"
    )

    # -----------------------------------------------------
    # Verify no dataset leakage
    # -----------------------------------------------------

    train_set = set(
        train_ids
    )

    test_set = set(
        test_ids
    )

    overlap = (
        train_set & test_set
    )

    print("\n" + "=" * 70)
    print("LEAKAGE CHECK")
    print("=" * 70)

    print(
        f"Training datasets: "
        f"{sorted(train_set)}"
    )

    print(
        f"Testing datasets: "
        f"{sorted(test_set)}"
    )

    print(
        f"Dataset overlap: "
        f"{overlap}"
    )

    if overlap:

        raise ValueError(
            "DATA LEAKAGE DETECTED!"
        )

    print(
        "✓ No dataset overlap detected."
    )

    print("\n" + "=" * 70)
    print("SPLIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()