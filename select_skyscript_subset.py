import math
import random
from collections import Counter, defaultdict

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

METADATA_FILE = "metadata_index.csv"
TRAIN_FILE = "SkyScript_train_unfiltered_5M.csv"

OUTPUT_CANDIDATES = "selected_candidates_v2.csv"
OUTPUT_FINAL = "selected_200k_v2.csv"

TARGET = 200_000
CANDIDATE_TARGET = 220_000

CHUNK_SIZE = 200_000
SEED = 42

random.seed(SEED)


# ============================================================
# SEMANTIC CATEGORY
# ============================================================

def classify(tags):

    if not isinstance(tags, str) or not tags:
        return "other"

    keys = {
        item.split("=", 1)[0]
        for item in tags.split(";")
        if "=" in item
    }

    if keys.intersection({
        "highway",
        "railway",
        "aeroway",
        "route",
        "bridge",
        "tunnel",
        "junction"
    }):
        return "transportation"

    if keys.intersection({
        "building",
        "office",
        "shop",
        "craft",
        "tourism"
    }):
        return "buildings"

    if keys.intersection({
        "natural",
        "waterway"
    }):
        return "natural"

    if "landuse" in keys:
        return "landuse"

    if "power" in keys:
        return "power_utilities"

    if keys.intersection({
        "leisure",
        "sport"
    }):
        return "leisure"

    if "religion" in keys:
        return "religion"

    return "other"


# ============================================================
# GEOGRAPHIC CELL
# ============================================================

def geo_cell(lat, lon):

    lat_cell = math.floor(lat / 5) * 5
    lon_cell = math.floor(lon / 5) * 5

    return f"{lat_cell}_{lon_cell}"


# ============================================================
# PASS 1
# Get source counts + source quotas
# ============================================================

print("=" * 60)
print("PASS 1: SOURCE DISTRIBUTION")
print("=" * 60)

source_counts = Counter()
total = 0

for chunk in pd.read_csv(
    METADATA_FILE,
    usecols=["source"],
    chunksize=CHUNK_SIZE
):

    source_counts.update(chunk["source"].dropna())
    total += len(chunk)

print(f"Metadata rows: {total:,}")

print("\nOriginal source distribution:")

for source, count in source_counts.most_common():

    print(
        f"{source:5s} "
        f"{count:10,} "
        f"{count / total * 100:6.2f}%"
    )


# ------------------------------------------------------------
# Source quotas
#
# US is capped at 45%.
# Remaining 55% follows the relative distribution of
# all non-US sources.
# ------------------------------------------------------------

US_CAP = 0.45

non_us_total = total - source_counts["US"]

source_quotas = {}

source_quotas["US"] = int(CANDIDATE_TARGET * US_CAP)

remaining = CANDIDATE_TARGET - source_quotas["US"]

for source, count in source_counts.items():

    if source == "US":
        continue

    proportion = count / non_us_total

    source_quotas[source] = round(
        remaining * proportion
    )


# Fix rounding
difference = CANDIDATE_TARGET - sum(source_quotas.values())

if difference != 0:

    source_quotas["CH"] += difference


print("\nSource quotas for candidate pool:")

for source, quota in source_quotas.items():

    print(
        f"{source:5s} "
        f"{quota:10,} "
        f"{quota / CANDIDATE_TARGET * 100:6.2f}%"
    )


# ============================================================
# PASS 2
# Count source + geographic + semantic strata
# ============================================================

print("\n" + "=" * 60)
print("PASS 2: GEOGRAPHIC / SEMANTIC STRATA")
print("=" * 60)

strata_counts = Counter()

usecols = [
    "filename",
    "source",
    "year",
    "latitude",
    "longitude",
    "center_tags",
    "surrounding_count"
]

total = 0

for chunk in pd.read_csv(
    METADATA_FILE,
    usecols=usecols,
    chunksize=CHUNK_SIZE
):

    total += len(chunk)

    chunk["category"] = chunk["center_tags"].map(
        classify
    )

    chunk["geo_cell"] = [
        geo_cell(lat, lon)
        for lat, lon in zip(
            chunk["latitude"],
            chunk["longitude"]
        )
    ]

    grouped = chunk.groupby(
        ["source", "geo_cell", "category"]
    ).size()

    for key, count in grouped.items():
        strata_counts[key] += int(count)

    if total % 1_000_000 == 0:
        print(f"Processed {total:,}")


print(f"\nTotal rows: {total:,}")
print(f"Strata:     {len(strata_counts):,}")


# ============================================================
# Allocate source quotas among geographic/semantic strata
#
# Within each source we use sqrt(count), which gives
# smaller geographic/semantic strata more representation.
# ============================================================

strata_by_source = defaultdict(list)

for key, count in strata_counts.items():

    source = key[0]

    if source in source_quotas:
        strata_by_source[source].append(
            (key, count)
        )


quotas = {}

for source, strata in strata_by_source.items():

    target = source_quotas[source]

    weights = {
        key: math.sqrt(count)
        for key, count in strata
    }

    total_weight = sum(weights.values())

    allocated = 0

    for key, count in strata:

        quota = round(
            target *
            weights[key] /
            total_weight
        )

        quota = min(quota, count)

        quotas[key] = quota
        allocated += quota

    # Redistribute rounding difference
    difference = target - allocated

    if difference > 0:

        candidates = sorted(
            strata,
            key=lambda x: x[1] - quotas[x[0]],
            reverse=True
        )

        for key, count in candidates:

            if difference <= 0:
                break

            available = count - quotas[key]

            add = min(available, difference)

            quotas[key] += add
            difference -= add


# ============================================================
# PASS 3
# Reservoir sampling
# ============================================================

print("\n" + "=" * 60)
print("PASS 3: RESERVOIR SAMPLING")
print("=" * 60)

reservoirs = defaultdict(list)
seen = Counter()

total = 0

for chunk in pd.read_csv(
    METADATA_FILE,
    usecols=usecols,
    chunksize=CHUNK_SIZE
):

    total += len(chunk)

    chunk["category"] = chunk["center_tags"].map(
        classify
    )

    chunk["geo_cell"] = [
        geo_cell(lat, lon)
        for lat, lon in zip(
            chunk["latitude"],
            chunk["longitude"]
        )
    ]

    for row in chunk.to_dict("records"):

        key = (
            row["source"],
            row["geo_cell"],
            row["category"]
        )

        if key not in quotas:
            continue

        limit = quotas[key]

        if limit <= 0:
            continue

        seen[key] += 1

        if len(reservoirs[key]) < limit:

            reservoirs[key].append(row)

        else:

            j = random.randint(
                0,
                seen[key] - 1
            )

            if j < limit:
                reservoirs[key][j] = row

    if total % 1_000_000 == 0:
        print(f"Processed {total:,}")


# ============================================================
# BUILD CANDIDATE DATASET
# ============================================================

candidates = []

for rows in reservoirs.values():
    candidates.extend(rows)

random.shuffle(candidates)

candidate_df = pd.DataFrame(candidates)

candidate_df.to_csv(
    OUTPUT_CANDIDATES,
    index=False
)

print(
    f"\nCandidate rows: "
    f"{len(candidate_df):,}"
)

print(
    f"Saved: {OUTPUT_CANDIDATES}"
)


# ============================================================
# MATCH AGAINST TRAINING CSV
# ============================================================

print("\n" + "=" * 60)
print("PASS 4: MATCH TRAINING CSV")
print("=" * 60)

candidate_names = set(
    candidate_df["filename"].astype(str)
)

matches = []

scanned = 0

for chunk in pd.read_csv(
    TRAIN_FILE,
    chunksize=CHUNK_SIZE
):

    scanned += len(chunk)

    chunk["filename"] = (
        chunk["filepath"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .str.split("/")
        .str[-1]
    )

    matched = chunk[
        chunk["filename"].isin(candidate_names)
    ]

    if len(matched):
        matches.append(matched)

    if scanned % 1_000_000 == 0:
        print(
            f"Scanned {scanned:,}"
        )


if not matches:
    raise RuntimeError(
        "No candidates matched training CSV."
    )


final_df = pd.concat(
    matches,
    ignore_index=True
)

final_df = final_df.drop_duplicates(
    subset=["filename"]
)


# ============================================================
# Add metadata
# ============================================================

metadata_columns = [
    "filename",
    "source",
    "year",
    "latitude",
    "longitude",
    "center_tags",
    "surrounding_count",
    "category",
    "geo_cell"
]

metadata_lookup = candidate_df[
    metadata_columns
].drop_duplicates(
    subset=["filename"]
)


final_df = final_df.merge(
    metadata_lookup,
    on="filename",
    how="left"
)


# ============================================================
# FINAL SHUFFLE + 200K
# ============================================================

final_df = final_df.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)

final_df = final_df.head(TARGET)


final_df.to_csv(
    OUTPUT_FINAL,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 60)
print("FINAL V2 DATASET")
print("=" * 60)

print(
    f"Training pairs: "
    f"{len(final_df):,}"
)

print(
    f"Saved: {OUTPUT_FINAL}"
)


print("\nSOURCE DISTRIBUTION")

print(
    final_df["source"]
    .value_counts()
    .to_string()
)


print("\nCATEGORY DISTRIBUTION")

print(
    final_df["category"]
    .value_counts()
    .to_string()
)


print("\nYEAR DISTRIBUTION")

print(
    final_df["year"]
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)


multi = (
    final_df["title_multi_objects"].fillna("")
    != final_df["title"].fillna("")
)

print("\nMULTI-OBJECT")

print(
    f"{multi.sum():,} / "
    f"{len(final_df):,} "
    f"= {multi.mean() * 100:.2f}%"
)


print("\nGEOGRAPHIC CELLS")

print(
    final_df["geo_cell"].nunique()
)


print("\nDONE.")