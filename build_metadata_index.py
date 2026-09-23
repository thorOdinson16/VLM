import os
import csv
import pickle

META_DIRS = [f"meta{i}" for i in range(2, 8)]

OUTPUT = "metadata_index.csv"

print("Building metadata index...")
print(f"Output: {OUTPUT}\n")

with open(OUTPUT, "w", newline="", encoding="utf-8") as out:

    writer = csv.writer(out)

    writer.writerow([
        "filename",
        "source",
        "year",
        "latitude",
        "longitude",
        "center_tags",
        "surrounding_count"
    ])

    total = 0
    written = 0
    errors = 0

    for meta_dir in META_DIRS:

        print(f"Processing {meta_dir}...")

        for filename in os.listdir(meta_dir):

            if not filename.endswith(".pickle"):
                continue

            total += 1

            path = os.path.join(meta_dir, filename)

            try:

                # --------------------------------
                # Filename information
                # --------------------------------

                name = filename[:-7]  # remove .pickle
                parts = name.split("_")

                source = parts[-2] if len(parts) >= 2 else ""
                year = parts[-1] if len(parts) >= 3 else ""

                # Corresponding image path
                image_filename = name + ".jpg"

                # --------------------------------
                # Load metadata
                # --------------------------------

                with open(path, "rb") as f:
                    data = pickle.load(f)

                # --------------------------------
                # Geographic information
                # --------------------------------

                bbox = data.get("bbox")

                latitude = ""
                longitude = ""

                if bbox and len(bbox) == 4:

                    west, south, east, north = map(float, bbox)

                    latitude = (south + north) / 2
                    longitude = (west + east) / 2

                # --------------------------------
                # Semantic information
                # --------------------------------

                center_tags = data.get("center_tags", {})
                surrounding_tags = data.get(
                    "surrounding_tags", []
                )

                # Store center tag keys/values compactly
                tag_string = ";".join(
                    f"{k}={v}"
                    for k, v in center_tags.items()
                )

                surrounding_count = len(surrounding_tags)

                writer.writerow([
                    image_filename,
                    source,
                    year,
                    latitude,
                    longitude,
                    tag_string,
                    surrounding_count
                ])

                written += 1

            except Exception as e:

                errors += 1

                if errors <= 10:
                    print("ERROR:", path, e)

            if total % 100000 == 0:
                print(
                    f"  Processed: {total:,} | "
                    f"Written: {written:,} | "
                    f"Errors: {errors:,}"
                )

print("\n==============================")
print("DONE")
print("==============================")

print(f"Processed: {total:,}")
print(f"Written:   {written:,}")
print(f"Errors:    {errors:,}")