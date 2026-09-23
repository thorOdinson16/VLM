import pandas as pd

CANDIDATES = "selected_candidates_v2.csv"

VAL = "SkyScript_val_5K_filtered_by_CLIP_laion_RS.csv"
TEST = "SkyScript_test_30K_filtered_by_CLIP_laion_RS.csv"

OUTPUT = "selected_200k_final.csv"

TARGET = 200_000


# ============================================================
# Load candidates
# ============================================================

print("=" * 60)
print("FINALIZING SKYScript TRAINING SET")
print("=" * 60)

candidates = pd.read_csv(CANDIDATES)

print(
    f"Candidate pool: {len(candidates):,}"
)


# ============================================================
# Load evaluation filenames
# ============================================================

print("\nLoading validation/test filenames...")

val = pd.read_csv(
    VAL,
    usecols=["filepath"]
)

test = pd.read_csv(
    TEST,
    usecols=["filepath"]
)


def get_filename(series):

    return (
        series
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .str.split("/")
        .str[-1]
    )


val_names = set(
    get_filename(val["filepath"])
)

test_names = set(
    get_filename(test["filepath"])
)

reserved = val_names | test_names

print(
    f"Validation images: {len(val_names):,}"
)

print(
    f"Test images:       {len(test_names):,}"
)

print(
    f"Reserved images:   {len(reserved):,}"
)


# ============================================================
# Remove evaluation leakage
# ============================================================

candidates["filename"] = (
    candidates["filename"]
    .astype(str)
)

before = len(candidates)

candidates = candidates[
    ~candidates["filename"].isin(reserved)
].copy()

removed = before - len(candidates)

print(
    f"\nRemoved benchmark overlap: {removed:,}"
)

print(
    f"Remaining candidates:      {len(candidates):,}"
)


# ============================================================
# Load training CSV
# ============================================================

print("\nMatching against training CSV...")

matches = []

for chunk in pd.read_csv(
    "SkyScript_train_unfiltered_5M.csv",
    chunksize=200_000
):

    chunk["filename"] = (
        chunk["filepath"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .str.split("/")
        .str[-1]
    )

    matched = chunk[
        chunk["filename"].isin(
            set(candidates["filename"])
        )
    ]

    if len(matched):
        matches.append(matched)

    print(
        f"Scanned {len(chunk):,} rows..."
    )


train = pd.concat(
    matches,
    ignore_index=True
)

train = train.drop_duplicates(
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

metadata = candidates[
    metadata_columns
].drop_duplicates(
    subset=["filename"]
)

final = train.merge(
    metadata,
    on="filename",
    how="inner"
)


# ============================================================
# Remove evaluation leakage AGAIN
# ============================================================

final = final[
    ~final["filename"].isin(reserved)
].copy()


# ============================================================
# Deterministic final selection
# ============================================================

final = final.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

if len(final) < TARGET:

    raise RuntimeError(
        f"Only {len(final):,} "
        f"leakage-free training images available; "
        f"need {TARGET:,}."
    )

final = final.head(TARGET)


# ============================================================
# Save
# ============================================================

final.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# Report
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATASET")
print("=" * 60)

print(
    f"Training pairs: {len(final):,}"
)

print(
    f"Unique images:  {final['filename'].nunique():,}"
)

print(
    f"Saved: {OUTPUT}"
)


print("\nSOURCE DISTRIBUTION")

print(
    final["source"]
    .value_counts()
    .to_string()
)


print("\nCATEGORY DISTRIBUTION")

print(
    final["category"]
    .value_counts()
    .to_string()
)


print("\nGEOGRAPHIC CELLS")

print(
    final["geo_cell"].nunique()
)


multi = (
    final["title_multi_objects"].fillna("")
    != final["title"].fillna("")
)

print("\nMULTI-OBJECT")

print(
    f"{multi.sum():,} / "
    f"{len(final):,} "
    f"= {multi.mean() * 100:.2f}%"
)

print("\nDONE.")