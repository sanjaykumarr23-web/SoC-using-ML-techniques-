"""
LSTM-based Battery SoC Estimation
----------------------------------

Input:
    Voltage
    Current
    Temperature
    Time

Window:
    20 time steps

Training datasets:
    00001, 00003, 00005, 00006, 00007, 00008

Testing datasets:
    00009, 00010

Evaluation:
    MAE
    RMSE
    R²
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.preprocessing import StandardScaler

from .feature_engineering import (
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

EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001


class SoCLSTM(nn.Module):

    def __init__(
        self,
        input_size=4,
        hidden_size=64,
        num_layers=2,
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )

        self.fc = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        # Use the final time step
        last_output = output[:, -1, :]

        prediction = self.fc(
            last_output
        )

        return prediction.squeeze(1)


def scale_sequences(
    X_train,
    X_test
):
    """
    Scale features using only training data.

    This prevents test-data leakage.
    """

    n_train, steps, features = (
        X_train.shape
    )

    n_test = X_test.shape[0]

    scaler = StandardScaler()

    X_train_2d = X_train.reshape(
        -1,
        features
    )

    X_test_2d = X_test.reshape(
        -1,
        features
    )

    scaler.fit(
        X_train_2d
    )

    X_train_scaled = scaler.transform(
        X_train_2d
    )

    X_test_scaled = scaler.transform(
        X_test_2d
    )

    X_train_scaled = X_train_scaled.reshape(
        n_train,
        steps,
        features
    )

    X_test_scaled = X_test_scaled.reshape(
        n_test,
        steps,
        features
    )

    return (
        X_train_scaled,
        X_test_scaled,
        scaler
    )


def main():

    print("\n" + "=" * 70)
    print("LSTM BATTERY SoC ESTIMATION")
    print("=" * 70)

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
    # Create windows
    # -----------------------------------------------------

    (
        X_train,
        _,
        y_train,
        train_ids
    ) = create_sliding_windows(
        train_df,
        window_size=WINDOW_SIZE
    )

    (
        X_test,
        _,
        y_test,
        test_ids
    ) = create_sliding_windows(
        test_df,
        window_size=WINDOW_SIZE
    )

    print("\nTraining sequence:")
    print(
        f"  {X_train.shape}"
    )

    print("\nTesting sequence:")
    print(
        f"  {X_test.shape}"
    )

    # -----------------------------------------------------
    # Scale features
    # -----------------------------------------------------

    print("\nScaling features...")

    (
        X_train,
        X_test,
        scaler
    ) = scale_sequences(
        X_train,
        X_test
    )

    # -----------------------------------------------------
    # Convert to tensors
    # -----------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.float32
    )

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nTraining device: {device}"
    )

    model = SoCLSTM(
        input_size=4,
        hidden_size=64,
        num_layers=2,
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -----------------------------------------------------
    # DataLoader
    # -----------------------------------------------------

    dataset = torch.utils.data.TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING LSTM")
    print("=" * 70)

    for epoch in range(
        EPOCHS
    ):

        model.train()

        epoch_loss = 0.0

        for X_batch, y_batch in dataloader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            prediction = model(
                X_batch
            )

            loss = criterion(
                prediction,
                y_batch
            )

            loss.backward()

            optimizer.step()

            epoch_loss += (
                loss.item()
                * len(X_batch)
            )

        epoch_loss /= len(dataset)

        print(
            f"Epoch "
            f"{epoch + 1:02d}/{EPOCHS} "
            f"- Loss: "
            f"{epoch_loss:.6f}"
        )

    # -----------------------------------------------------
    # Testing
    # -----------------------------------------------------

    print("\nGenerating test predictions...")

    model.eval()

    with torch.no_grad():

        X_test_tensor = (
            X_test_tensor.to(device)
        )

        y_pred = model(
            X_test_tensor
        )

        y_pred = (
            y_pred
            .cpu()
            .numpy()
        )

    # Keep predictions physically valid
    y_pred = np.clip(
        y_pred,
        0.0,
        1.0
    )

    # -----------------------------------------------------
    # Save predictions
    # -----------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True
    )

    lstm_results = pd.DataFrame({
        "dataset_id": test_ids,
        "true_soc": y_test,
        "predicted_soc": y_pred
    })

    lstm_results.to_csv(
        "results/lstm_predictions.csv",
        index=False
    )

    print(
        "\nSaved predictions to "
        "results/lstm_predictions.csv"
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    # -----------------------------------------------------
    # Overall results
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("LSTM RESULTS")
    print("=" * 70)

    print(
        f"\nMAE  : {mae:.6f}"
    )

    print(
        f"RMSE : {rmse:.6f}"
    )

    print(
        f"R²   : {r2:.6f}"
    )

    print(
        f"\nMAE  : {mae * 100:.2f}% SoC"
    )

    print(
        f"RMSE : {rmse * 100:.2f}% SoC"
    )

    # -----------------------------------------------------
    # Per-dataset results
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("PER-DATASET TEST PERFORMANCE")
    print("=" * 70)

    for dataset_id in np.unique(
        test_ids
    ):

        mask = (
            test_ids == dataset_id
        )

        true_values = y_test[
            mask
        ]

        pred_values = y_pred[
            mask
        ]

        dataset_mae = (
            mean_absolute_error(
                true_values,
                pred_values
            )
        )

        dataset_rmse = np.sqrt(
            mean_squared_error(
                true_values,
                pred_values
            )
        )

        dataset_r2 = r2_score(
            true_values,
            pred_values
        )

        print(
            f"\nDataset {dataset_id}:"
        )

        print(
            f"  MAE  : "
            f"{dataset_mae * 100:.2f}%"
        )

        print(
            f"  RMSE : "
            f"{dataset_rmse * 100:.2f}%"
        )

        print(
            f"  R²   : "
            f"{dataset_r2:.6f}"
        )

    # -----------------------------------------------------
    # Example predictions
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTIONS")
    print("=" * 70)

    print(
        "\nTrue SoC (%)    Predicted SoC (%)"
    )

    for i in range(
        min(10, len(y_test))
    ):

        print(
            f"{y_test[i] * 100:10.2f}"
            f"    "
            f"{y_pred[i] * 100:15.2f}"
        )

    print("\nLSTM evaluation complete.")


if __name__ == "__main__":
    main()