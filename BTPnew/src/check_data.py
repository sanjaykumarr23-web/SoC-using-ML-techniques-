from src.data_loader import CSVBatteryDataLoader

loader = CSVBatteryDataLoader()

df = loader.load_selected_datasets()

print("\n" + "=" * 70)
print("DATA DIAGNOSTIC")
print("=" * 70)

for dataset_id, group in df.groupby("dataset_id"):

    print(f"\nDataset: {dataset_id}")
    print(f"Type: {group['cycle_type'].iloc[0]}")
    print(f"Samples: {len(group)}")

    print(
        f"Time: {group['time'].min():.3f} "
        f"to {group['time'].max():.3f}"
    )

    print(
        f"Voltage: {group['voltage_measured'].min():.4f} "
        f"to {group['voltage_measured'].max():.4f} V"
    )

    print(
        f"Current: {group['current_measured'].min():.4f} "
        f"to {group['current_measured'].max():.4f} A"
    )

    print(
        f"Temperature: "
        f"{group['temperature_measured'].min():.2f} "
        f"to {group['temperature_measured'].max():.2f}"
    )

print("\n" + "=" * 70)
print("OVERALL CURRENT RANGE")
print("=" * 70)

print(
    df.groupby("cycle_type")["current_measured"]
      .agg(["min", "max", "mean"])
)

print("\n" + "=" * 70)
print("TIME RANGE")
print("=" * 70)

print(
    df.groupby("dataset_id")["time"]
      .agg(["min", "max"])
)

print("\nDiagnostic complete.")