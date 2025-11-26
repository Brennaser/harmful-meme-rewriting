import json
import csv

input_jsonl = "img/train190_subset.jsonl"
gemma_csv = "memes_gemma_basic.csv"      # your existing csv
output_csv = "memes_gemma_basic_with_ids.csv"

# 1) Load original items (with ids)
orig_items = []
with open(input_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        orig_items.append(json.loads(line))

# 2) Load old Gemma rows
gemma_rows = []
with open(gemma_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        gemma_rows.append(row)

assert len(orig_items) == len(gemma_rows), "Row count mismatch!"

# 3) Merge row by row
merged_rows = []
for orig, gen in zip(orig_items, gemma_rows):
    merged = {
        "id": orig.get("id", ""),
        "img": orig.get("img", ""),
        "label": orig.get("label", ""),
        "text": orig.get("text", ""),
    }
    # add all gemma columns too
    merged.update(gen)
    merged_rows.append(merged)

# 4) Save merged CSV
fieldnames = list(merged_rows[0].keys())
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(merged_rows)

print("Saved merged file to", output_csv)
