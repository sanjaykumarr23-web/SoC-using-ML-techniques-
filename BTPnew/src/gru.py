
"""
GRU for Battery State of Charge (SoC) Estimation
------------------------------------------------

Input features:
    voltage_measured
    current_measured
    temperature_measured
    time

Sequence:
    20 time steps x 4 features

Training datasets:
    00001, 00003, 00005, 00006, 00007, 00008

Testing datasets:
    00009, 00010

Evaluation:
    MAE, RMSE, R2
"""

from pathlib import Path
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from .feature_engineering import (
    load_reference_data,
    create_sliding_windows,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

WINDOW_SIZE = 20
INPUT_SIZE = 4
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001

RANDOM_SEED = 42

TRAIN_DATASETS = [
    "00001", "00003", "00005",
    "00006", "00007", "00008"
]

TEST_DATASETS = [
    "00009", "00010"
]

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# GRU Model
# ---------------------------------------------------------

class SoCGRU(nn.Module):

    def __init__(
        self,
        input_size=4,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        output, hidden = self.gru(x)

        # Use the output at the final time step.
        last_output = output[:, -1, :]

        prediction = self.fc(last_output)

        return prediction.squeeze(1)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(
        actual, predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual, predicted
        )
    )

    r2 = r2_score(
        actual, predicted
    )

    return mae, rmse, r2


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    set_seed(RANDOM_SEED)

    print("\n" + "=" * 70)
    print("GRU SoC ESTIMATION")
    print("=" * 70)

    print("\nTraining device:", DEVICE)

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    df = load_reference_data()

    df["dataset_id"] = (
        df["dataset_id"]
        .astype(str)
        .str.zfill(5)
    )

    train_df = df[
        df["dataset_id"].isin(TRAIN_DATASETS)
    ].copy()

    test_df = df[
        df["dataset_id"].isin(TEST_DATASETS)
    ].copy()

    # -----------------------------------------------------
    # Create sliding windows
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

    print("\nTraining data:")
    print("X shape:", X_train_seq.shape)
    print("y shape:", y_train.shape)

    print("\nTesting data:")
    print("X shape:", X_test_seq.shape)
    print("y shape:", y_test.shape)

    # Check dataset-level separation.
    assert set(np.unique(train_ids)).isdisjoint(
        set(np.unique(test_ids))
    ), "Dataset leakage detected!"

    # -----------------------------------------------------
    # Feature scaling
    # Fit only on training data.
    # -----------------------------------------------------

    scaler = StandardScaler()

    n_features = X_train_seq.shape[2]

    X_train_2d = X_train_seq.reshape(
        -1, n_features
    )

    X_test_2d = X_test_seq.reshape(
        -1, n_features
    )

    scaler.fit(X_train_2d)

    X_train_scaled = scaler.transform(
        X_train_2d
    ).reshape(X_train_seq.shape)

    X_test_scaled = scaler.transform(
        X_test_2d
    ).reshape(X_test_seq.shape)

    # -----------------------------------------------------
    # Convert to PyTorch tensors
    # -----------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train_scaled,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.float32
    )

    X_test_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32
    )

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # -----------------------------------------------------
    # Initialize model
    # -----------------------------------------------------

    model = SoCGRU(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("\n" + "=" * 70)
    print("GRU MODEL CONFIGURATION")
    print("=" * 70)

    print("Input size     :", INPUT_SIZE)
    print("Hidden size    :", HIDDEN_SIZE)
    print("GRU layers     :", NUM_LAYERS)
    print("Dropout        :", DROPOUT)
    print("Epochs         :", EPOCHS)
    print("Batch size     :", BATCH_SIZE)
    print("Learning rate  :", LEARNING_RATE)
    print("Optimizer      : Adam")
    print("Loss function  : MSE")

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING GRU")
    print("=" * 70)

    start_time = time.time()

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            predictions = model(batch_X)

            loss = criterion(
                predictions,
                batch_y
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item() * batch_X.size(0)
            )

        epoch_loss = (
            total_loss / len(train_dataset)
        )

        if (
            (epoch + 1) % 5 == 0
            or epoch == 0
        ):

            print(
                f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
                f"Loss: {epoch_loss:.6f}"
            )

    training_time = time.time() - start_time

    print("\nTraining completed.")
    print(f"Training time: {training_time:.2f} seconds")

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    print("\nGenerating predictions...")

    model.eval()

    prediction_start = time.time()

    with torch.no_grad():

        X_test_tensor = X_test_tensor.to(DEVICE)

        y_pred = model(
            X_test_tensor
        ).cpu().numpy()

    prediction_time = (
        time.time() - prediction_start
    )

    y_pred = np.clip(
        y_pred,
        0.0,
        1.0
    )

    print(
        f"Prediction time: {prediction_time:.6f} seconds"
    )

    # -----------------------------------------------------
    # Save predictions
    # -----------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    prediction_df = pd.DataFrame({

        "dataset_id": test_ids,

        "true_soc": y_test,

        "predicted_soc": y_pred

    })

    prediction_path = (
        RESULTS_DIR / "gru_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False
    )

    print(
        "\nPredictions saved:",
        prediction_path
    )

    # -----------------------------------------------------
    # Overall evaluation
    # -----------------------------------------------------

    mae, rmse, r2 = calculate_metrics(
        y_test,
        y_pred
    )

    print("\n" + "=" * 70)
    print("GRU OVERALL RESULTS")
    print("=" * 70)

    print(
        f"MAE  : {mae:.6f} ({mae * 100:.4f}% SoC)"
    )

    print(
        f"RMSE : {rmse:.6f} ({rmse * 100:.4f}% SoC)"
    )

    print(
        f"R2   : {r2:.6f}"
    )

    # -----------------------------------------------------
    # Per-dataset evaluation
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("PER-DATASET PERFORMANCE")
    print("=" * 70)

    per_dataset_results = []

    for dataset_id in np.unique(test_ids):

        mask = test_ids == dataset_id

        actual = y_test[mask]
        predicted = y_pred[mask]

        d_mae, d_rmse, d_r2 = calculate_metrics(
            actual,
            predicted
        )

        print(f"\nDataset {dataset_id}")
        print(f"Samples : {mask.sum()}")
        print(f"MAE     : {d_mae * 100:.4f}%")
        print(f"RMSE    : {d_rmse * 100:.4f}%")
        print(f"R2      : {d_r2:.6f}")

        per_dataset_results.append({

            "model": "GRU",
            "dataset_id": dataset_id,
            "samples": int(mask.sum()),
            "MAE": d_mae,
            "RMSE": d_rmse,
            "R2": d_r2

        })

    # -----------------------------------------------------
    # Save summary metrics
    # -----------------------------------------------------

    summary = pd.DataFrame([{

        "model": "GRU",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAE_percent": mae * 100,
        "RMSE_percent": rmse * 100,
        "training_time_seconds": training_time,
        "prediction_time_seconds": prediction_time

    }])

    summary.to_csv(
        RESULTS_DIR / "gru_metrics.csv",
        index=False
    )

    pd.DataFrame(
        per_dataset_results
    ).to_csv(
        RESULTS_DIR / "gru_per_dataset_metrics.csv",
        index=False
    )

    # Save trained model.
    torch.save(
        model.state_dict(),
        RESULTS_DIR / "gru_model.pth"
    )

    print("\n" + "=" * 70)
    print("GRU EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
