"""
Battery Capacity Degradation Plot
----------------------------------
Uses the capacity values calculated from discharge datasets.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


CAPACITY_DATA = {
    "dataset_id": ["00001", "00005", "00007", "00009"],
    "capacity_Ah": [
        1.705934,
        1.548536,
        1.532319,
        1.511623,
    ],
}


def main():

    print("\n" + "=" * 70)
    print("BATTERY CAPACITY DEGRADATION PLOT")
    print("=" * 70)

    df = pd.DataFrame(CAPACITY_DATA)

    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["dataset_id"],
        df["capacity_Ah"],
        marker="o",
        linewidth=2
    )

    plt.xlabel("Discharge Dataset")
    plt.ylabel("Capacity (Ah)")
    plt.title("Battery Capacity Degradation")

    plt.grid(True, alpha=0.3)

    for i, row in df.iterrows():

        plt.annotate(
            f'{row["capacity_Ah"]:.3f} Ah',
            (
                row["dataset_id"],
                row["capacity_Ah"]
            ),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center"
        )

    plt.tight_layout()

    output_file = (
        "results/capacity_degradation.png"
    )

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.close()

    print("\nCapacity values:")
    print(df.to_string(index=False))

    print(
        f"\nSaved: {output_file}"
    )

    print("\nCapacity degradation plot complete.")


if __name__ == "__main__":
    main()