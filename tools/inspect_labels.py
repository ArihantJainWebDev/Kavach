from pathlib import Path
from collections import Counter

import pandas as pd


ROOT = Path("data/unraveled/data/network-flows")

columns = [
    "Activity",
    "Stage",
    "DefenderResponse",
    "Signature",
]

files = sorted(ROOT.rglob("*.csv"))

print(f"CSV files found: {len(files)}")

counts = {column: Counter() for column in columns}
total_rows = 0

for index, file in enumerate(files, start=1):
    print(f"[{index}/{len(files)}] {file.name}")

    try:
        for chunk in pd.read_csv(
            file,
            usecols=columns,
            engine="python",
            on_bad_lines="skip",
            chunksize=100_000,
        ):
            total_rows += len(chunk)

            for column in columns:
                values = chunk[column].dropna().astype(str)
                counts[column].update(values)

    except Exception as exc:
        print(f"  ERROR: {exc}")

print()
print("=" * 70)
print(f"ROWS PROCESSED: {total_rows:,}")
print("=" * 70)

for column in columns:
    print()
    print(f"{column}")
    print("-" * 40)

    for value, count in counts[column].most_common():
        print(f"{value:40} {count:,}")