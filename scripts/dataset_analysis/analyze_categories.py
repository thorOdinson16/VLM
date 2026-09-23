from pathlib import Path
import os
import pickle
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
META_DIRS = [ROOT / f"meta{i}" for i in range(2, 8)]

# Broad semantic categories
CATEGORY_RULES = {
    "transportation": {
        "highway", "railway", "route", "aeroway",
        "bridge", "tunnel", "junction"
    },

    "buildings": {
        "building", "office", "shop", "amenity",
        "craft", "tourism"
    },

    "natural": {
        "natural", "waterway"
    },

    "landuse": {
        "landuse"
    },

    "power_utilities": {
        "power", "utility"
    },

    "leisure": {
        "leisure", "sport"
    },

    "agriculture": {
        "crop", "farm", "orchard", "vineyard"
    },

    "religion": {
        "religion"
    },

    "other": set()
}


def classify(tags):

    keys = set(tags.keys())

    for category, rule_keys in CATEGORY_RULES.items():

        if category == "other":
            continue

        if keys.intersection(rule_keys):
            return category

    return "other"


counts = Counter()
total = 0

print("Analyzing semantic categories...\n")

for meta_dir in META_DIRS:

    print(f"Processing {meta_dir}...")

    for filename in os.listdir(meta_dir):

        if not filename.endswith(".pickle"):
            continue

        path = os.path.join(meta_dir, filename)

        try:

            with open(path, "rb") as f:
                data = pickle.load(f)

            tags = data.get("center_tags", {})

            category = classify(tags)

            counts[category] += 1
            total += 1

        except Exception:
            pass


print("\n==============================")
print("SEMANTIC CATEGORY DISTRIBUTION")
print("==============================")

for category, count in counts.most_common():

    percentage = count / total * 100

    print(
        f"{category:20s} "
        f"{count:10,} "
        f"{percentage:6.2f}%"
    )

print("\nTotal:", f"{total:,}")