import pandas as pd

TRAIN = "SkyScript_train_unfiltered_5M.csv"
VAL = "SkyScript_val_5K_filtered_by_CLIP_laion_RS.csv"
TEST = "SkyScript_test_30K_filtered_by_CLIP_laion_RS.csv"
SELECTED = "selected_200k_final.csv"


print("=" * 60)
print("VERIFYING SELECTED DATASET")
print("=" * 60)


# ------------------------------------------------------------
# Selected dataset
# ------------------------------------------------------------

selected = pd.read_csv(SELECTED)

print(f"\nSelected rows: {len(selected):,}")
print(f"Unique filenames: {selected['filename'].nunique():,}")
print(f"Unique filepaths: {selected['filepath'].nunique():,}")


# ------------------------------------------------------------
# Check image archive distribution
# ------------------------------------------------------------

print("\nIMAGE ARCHIVE DISTRIBUTION")

archive = (
    selected["filepath"]
    .astype(str)
    .str.replace("\\", "/", regex=False)
    .str.split("/")
    .str[0]
)

print(archive.value_counts().sort_index().to_string())


# ------------------------------------------------------------
# Load validation/test filenames
# ------------------------------------------------------------

print("\nLoading validation/test sets...")

val = pd.read_csv(VAL, usecols=["filepath"])
test = pd.read_csv(TEST, usecols=["filepath"])

val_names = set(
    val["filepath"]
    .astype(str)
    .str.replace("\\", "/", regex=False)
    .str.split("/")
    .str[-1]
)

test_names = set(
    test["filepath"]
    .astype(str)
    .str.replace("\\", "/", regex=False)
    .str.split("/")
    .str[-1]
)

selected_names = set(
    selected["filename"].astype(str)
)


# ------------------------------------------------------------
# Overlap
# ------------------------------------------------------------

val_overlap = selected_names & val_names
test_overlap = selected_names & test_names

print("\nSPLIT OVERLAP")

print(
    f"Validation overlap: {len(val_overlap):,}"
)

print(
    f"Test overlap:       {len(test_overlap):,}"
)


# ------------------------------------------------------------
# Missing archive prefix
# ------------------------------------------------------------

valid_prefixes = {
    "images2",
    "images3",
    "images4",
    "images5",
    "images6",
    "images7"
}

invalid = archive[
    ~archive.isin(valid_prefixes)
]

print("\nINVALID IMAGE PREFIXES")
print(len(invalid))


# ------------------------------------------------------------
# Final
# ------------------------------------------------------------

print("\n" + "=" * 60)

if (
    len(selected) == 200_000
    and selected["filename"].nunique() == 200_000
    and len(val_overlap) == 0
    and len(test_overlap) == 0
    and len(invalid) == 0
):
    print("DATASET CHECK PASSED")
else:
    print("DATASET CHECK FAILED")

print("=" * 60)