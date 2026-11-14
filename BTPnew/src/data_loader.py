"""
CSV Battery Dataset Loader
---------------------------
Loads battery charge/discharge CSV files containing:
    - Voltage_measured
    - Current_measured
    - Temperature_measured
    - Time

Impedance-only files are automatically skipped.

Designed for the BTP SoC estimation project.
"""

from pathlib import Path
from typing import Optional, List
import logging

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Files selected for the main 8-dataset experiment.
# We can change this later if our dataset analysis suggests
# a better selection.
SELECTED_DATASETS = [
    "00001.csv",
    "00003.csv",
    "00005.csv",
    "00006.csv",
    "00007.csv",
    "00008.csv",
    "00009.csv",
    "00010.csv",
]


# ---------------------------------------------------------
# CSV Battery Data Loader
# ---------------------------------------------------------

class CSVBatteryDataLoader:
    """
    Loads and standardizes battery CSV datasets.

    Expected core columns:

        Voltage_measured
        Current_measured
        Temperature_measured
        Time

    Charge files may additionally contain:

        Current_charge
        Voltage_charge

    Discharge files may additionally contain:

        Current_load
        Voltage_load

    Impedance-only files are skipped.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        selected_files: Optional[List[str]] = None
    ):
        if data_dir is None:
            # BTPnew/src/data_loader.py
            # data directory is BTPnew/data/
            self.data_dir = Path(__file__).resolve().parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)

        self.selected_files = (
            selected_files
            if selected_files is not None
            else SELECTED_DATASETS
        )

        logger.info("Data directory: %s", self.data_dir)

    # -----------------------------------------------------
    # List CSV files
    # -----------------------------------------------------

    def list_available_files(self) -> List[str]:
        """Return all CSV files available in the data directory."""

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory does not exist: {self.data_dir}"
            )

        files = sorted(
            [file.name for file in self.data_dir.glob("*.csv")]
        )

        logger.info("Found %d CSV files.", len(files))

        return files

    # -----------------------------------------------------
    # Detect cycle type
    # -----------------------------------------------------

    @staticmethod
    def detect_cycle_type(df: pd.DataFrame) -> str:
        """
        Determine whether a CSV contains charge, discharge,
        or unsupported impedance data.
        """

        columns = {str(col).strip().lower() for col in df.columns}

        # Impedance files generally do not contain the required
        # continuous voltage/current/temperature/time telemetry.
        has_core_telemetry = {
            "voltage_measured",
            "current_measured",
            "temperature_measured",
            "time",
        }.issubset(columns)

        if not has_core_telemetry:
            return "impedance"

        has_charge_columns = (
            "current_charge" in columns
            or "voltage_charge" in columns
        )

        has_discharge_columns = (
            "current_load" in columns
            or "voltage_load" in columns
        )

        if has_charge_columns:
            return "charge"

        if has_discharge_columns:
            return "discharge"

        # If core telemetry exists but there is no explicit
        # charge/load column, keep it as telemetry rather than
        # incorrectly calling it impedance.
        return "telemetry"

    # -----------------------------------------------------
    # Standardize column names
    # -----------------------------------------------------

    @staticmethod
    def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names while preserving the original
        information needed by the project.
        """

        rename_map = {}

        for column in df.columns:
            clean_name = str(column).strip()

            normalized = clean_name.lower().replace(" ", "_")

            if normalized == "voltage_measured":
                rename_map[column] = "voltage_measured"

            elif normalized == "current_measured":
                rename_map[column] = "current_measured"

            elif normalized == "temperature_measured":
                rename_map[column] = "temperature_measured"

            elif normalized == "time":
                rename_map[column] = "time"

            elif normalized == "current_charge":
                rename_map[column] = "current_charge"

            elif normalized == "voltage_charge":
                rename_map[column] = "voltage_charge"

            elif normalized == "current_load":
                rename_map[column] = "current_load"

            elif normalized == "voltage_load":
                rename_map[column] = "voltage_load"

        return df.rename(columns=rename_map)

    # -----------------------------------------------------
    # Load one CSV
    # -----------------------------------------------------

    def load_file(self, filename: str) -> pd.DataFrame:
        """
        Load and validate one CSV file.
        """

        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(
                f"CSV file not found: {filepath}"
            )

        logger.info("Loading: %s", filename)

        df = pd.read_csv(filepath)

        df = self.standardize_columns(df)

        cycle_type = self.detect_cycle_type(df)

        logger.info(
            "%s -> type=%s, rows=%d",
            filename,
            cycle_type,
            len(df)
        )

        # -------------------------------------------------
        # Required telemetry
        # -------------------------------------------------

        required_columns = [
            "voltage_measured",
            "current_measured",
            "temperature_measured",
            "time",
        ]

        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            logger.warning(
                "Skipping %s. Missing columns: %s",
                filename,
                missing
            )

            return pd.DataFrame()

        # -------------------------------------------------
        # Reject impedance-only data
        # -------------------------------------------------

        if cycle_type == "impedance":
            logger.info(
                "Skipping impedance-only dataset: %s",
                filename
            )

            return pd.DataFrame()

        # -------------------------------------------------
        # Keep required columns + useful optional columns
        # -------------------------------------------------

        useful_columns = [
            "time",
            "voltage_measured",
            "current_measured",
            "temperature_measured",
            "current_charge",
            "voltage_charge",
            "current_load",
            "voltage_load",
        ]

        existing_columns = [
            col for col in useful_columns
            if col in df.columns
        ]

        df = df[existing_columns].copy()

        # -------------------------------------------------
        # Convert numerical columns
        # -------------------------------------------------

        for column in existing_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        # -------------------------------------------------
        # Remove invalid rows
        # -------------------------------------------------

        df = df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        df = df.dropna(
            subset=[
                "time",
                "voltage_measured",
                "current_measured",
                "temperature_measured",
            ]
        )

        # -------------------------------------------------
        # Sort by time
        # -------------------------------------------------

        df = df.sort_values("time").reset_index(drop=True)

        # -------------------------------------------------
        # Add dataset metadata
        # -------------------------------------------------

        dataset_id = Path(filename).stem

        df["dataset_id"] = dataset_id
        df["cycle_type"] = cycle_type

        # Useful ordering for later processing
        column_order = [
            "dataset_id",
            "cycle_type",
            "time",
            "voltage_measured",
            "current_measured",
            "temperature_measured",
        ]

        optional_columns = [
            col
            for col in [
                "current_charge",
                "voltage_charge",
                "current_load",
                "voltage_load",
            ]
            if col in df.columns
        ]

        df = df[
            column_order + optional_columns
        ]

        logger.info(
            "Loaded %s successfully: %d rows",
            filename,
            len(df)
        )

        return df

    # -----------------------------------------------------
    # Load selected datasets
    # -----------------------------------------------------

    def load_selected_datasets(self) -> pd.DataFrame:
        """
        Load the selected eight datasets and combine them.
        """

        available_files = set(
            self.list_available_files()
        )

        all_data = []

        for filename in self.selected_files:

            if filename not in available_files:
                logger.warning(
                    "Selected file not found: %s",
                    filename
                )
                continue

            try:
                df = self.load_file(filename)

                if not df.empty:
                    all_data.append(df)

            except Exception as exc:
                logger.error(
                    "Error loading %s: %s",
                    filename,
                    exc
                )

        if not all_data:
            raise ValueError(
                "No valid battery datasets were loaded."
            )

        combined_df = pd.concat(
            all_data,
            ignore_index=True
        )

        logger.info(
            "Combined dataset shape: %s",
            combined_df.shape
        )

        return combined_df

    # -----------------------------------------------------
    # Load all compatible datasets
    # -----------------------------------------------------

    def load_all_compatible_datasets(self) -> pd.DataFrame:
        """
        Scan every CSV in the data directory and automatically
        keep only compatible charge/discharge telemetry files.

        This is useful for checking how many usable datasets
        are available.
        """

        available_files = self.list_available_files()

        all_data = []

        for filename in available_files:

            try:
                df = self.load_file(filename)

                if not df.empty:
                    all_data.append(df)

            except Exception as exc:
                logger.error(
                    "Error loading %s: %s",
                    filename,
                    exc
                )

        if not all_data:
            raise ValueError(
                "No compatible telemetry datasets found."
            )

        combined_df = pd.concat(
            all_data,
            ignore_index=True
        )

        return combined_df

    # -----------------------------------------------------
    # Dataset summary
    # -----------------------------------------------------

    def print_dataset_summary(
        self,
        df: pd.DataFrame
    ) -> None:
        """Print a useful summary for verification."""

        print("\n" + "=" * 70)
        print("BATTERY DATASET SUMMARY")
        print("=" * 70)

        print(
            f"Total records : {len(df):,}"
        )

        print(
            f"Datasets      : {df['dataset_id'].nunique()}"
        )

        print("\nDataset breakdown:")

        summary = (
            df.groupby(
                ["dataset_id", "cycle_type"]
            )
            .size()
            .reset_index(
                name="samples"
            )
        )

        print(summary.to_string(index=False))

        print("\nColumns:")
        print(list(df.columns))

        print("=" * 70)


# ---------------------------------------------------------
# Convenience function
# ---------------------------------------------------------

def load_battery_dataset(
    data_dir: Optional[str] = None,
    selected_files: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convenience function used by the rest of the BTP pipeline.
    """

    loader = CSVBatteryDataLoader(
        data_dir=data_dir,
        selected_files=selected_files
    )

    return loader.load_selected_datasets()


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    print("\nStarting CSV battery data loader...\n")

    loader = CSVBatteryDataLoader()

    print("Available CSV files:")
    for file in loader.list_available_files():
        print("  ", file)

    print("\nLoading selected datasets...\n")

    df = loader.load_selected_datasets()

    loader.print_dataset_summary(df)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)