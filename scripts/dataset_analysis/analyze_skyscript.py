from pathlib import Path
import pandas as pd
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
FILE = ROOT / "SkyScript_train_unfiltered_5M.csv"
CHUNK_SIZE = 100_000

total = 0
multi_object = 0
single_object = 0

source_counts = Counter()
year_counts = Counter()

print("Starting analysis...\n")

for i, chunk in enumerate(pd.read_csv(FILE, chunksize=CHUNK_SIZE)):

    total += len(chunk)

    # Multi-object captions
    multi = (
        chunk["title_multi_objects"].fillna("")
        != chunk["title"].fillna("")
    )

    multi_object += multi.sum()
    single_object += (~multi).sum()

    # Image source + year from filename
    for path in chunk["filepath"].dropna():

        filename = path.split("/")[-1]
        parts = filename.replace(".jpg", "").split("_")

        # Example:
        # a242763481_CH_18
        if len(parts) >= 3:
            source_counts[parts[-2]] += 1
            year_counts[parts[-1]] += 1

    if (i + 1) % 5 == 0:
        print(f"Processed {total:,} rows...")

print("\n========== RESULTS ==========")

print(f"Total rows:          {total:,}")
print(f"Single-object:       {single_object:,}")
print(f"Multi-object:        {multi_object:,}")

print("\nTop image sources:")
for source, count in source_counts.most_common(20):
    print(f"{source:10s} {count:,}")

print("\nYears:")
for year, count in sorted(year_counts.items()):
    print(f"{year}: {count:,}")