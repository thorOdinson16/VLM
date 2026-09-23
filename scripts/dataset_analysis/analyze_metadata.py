from pathlib import Path
import os
import pickle
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
META_DIRS = [ROOT / f"meta{i}" for i in range(2, 8)]

total = 0
valid = 0
invalid = 0

sources = Counter()
years = Counter()

center_objects = Counter()
surrounding_objects = Counter()
all_objects = Counter()

lat_bins = Counter()
lon_bins = Counter()

print("Starting metadata analysis...\n")

for meta_dir in META_DIRS:

    print(f"Processing {meta_dir}...")

    for filename in os.listdir(meta_dir):

        if not filename.endswith(".pickle"):
            continue

        total += 1

        path = os.path.join(meta_dir, filename)

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            valid += 1

            # -------------------------
            # Filename information
            # -------------------------

            name = filename.replace(".pickle", "")
            parts = name.split("_")

            if len(parts) >= 3:
                source = parts[-2]
                year = parts[-1]

                sources[source] += 1
                years[year] += 1

            # -------------------------
            # Geographic information
            # -------------------------

            bbox = data.get("bbox")

            if bbox and len(bbox) == 4:

                west, south, east, north = map(float, bbox)

                center_lat = (south + north) / 2
                center_lon = (west + east) / 2

                # 5-degree geographic bins
                lat_bin = int(center_lat // 5) * 5
                lon_bin = int(center_lon // 5) * 5

                lat_bins[lat_bin] += 1
                lon_bins[lon_bin] += 1

            # -------------------------
            # Semantic tags
            # -------------------------

            center_tags = data.get("center_tags", {})
            surrounding_tags = data.get("surrounding_tags", [])

            for key, value in center_tags.items():
                center_objects[f"{key}={value}"] += 1
                all_objects[key] += 1

            for obj in surrounding_tags:

                if isinstance(obj, dict):

                    for key, value in obj.items():
                        surrounding_objects[f"{key}={value}"] += 1
                        all_objects[key] += 1

        except Exception as e:

            invalid += 1

            if invalid <= 10:
                print("ERROR:", path, e)

    print(f"  processed: {total:,}")

print("\n==============================")
print("FINAL METADATA STATISTICS")
print("==============================")

print(f"\nTotal metadata files: {total:,}")
print(f"Valid:                 {valid:,}")
print(f"Invalid:               {invalid:,}")

print("\n------------------------------")
print("IMAGE SOURCES")
print("------------------------------")

for source, count in sources.most_common():
    print(f"{source:8s} {count:10,}")

print("\n------------------------------")
print("YEARS")
print("------------------------------")

for year, count in sorted(years.items()):
    print(f"{year:8s} {count:10,}")

print("\n------------------------------")
print("TOP CENTER OBJECT TAGS")
print("------------------------------")

for tag, count in center_objects.most_common(30):
    print(f"{tag:40s} {count:10,}")

print("\n------------------------------")
print("TOP SURROUNDING OBJECT TAGS")
print("------------------------------")

for tag, count in surrounding_objects.most_common(30):
    print(f"{tag:40s} {count:10,}")

print("\n------------------------------")
print("LATITUDE DISTRIBUTION")
print("------------------------------")

for lat, count in sorted(lat_bins.items()):
    print(f"{lat:5d} to {lat + 5:5d}: {count:10,}")

print("\n------------------------------")
print("LONGITUDE DISTRIBUTION")
print("------------------------------")

for lon, count in sorted(lon_bins.items()):
    print(f"{lon:5d} to {lon + 5:5d}: {count:10,}")

print("\nDone.")