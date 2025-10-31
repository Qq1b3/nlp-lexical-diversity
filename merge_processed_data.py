import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import statistics

# --- Paths ---
processed_dir = Path("processed_data")
output_combined = processed_dir / "processed_data_all.csv"
output_stats = processed_dir / "dataset_statistics_combined.json"

# --- Step 1: Merge all *-chatgpt.csv files ---
csv_files = list(processed_dir.glob("*-chatgpt.csv"))
if not csv_files:
    print("❌ No processed CSVs found. Make sure they are inside processed_data/.")
    exit()

print(f"Found {len(csv_files)} files to merge:")
for f in csv_files:
    print("  -", f.name)

df_list = [pd.read_csv(f) for f in csv_files]
merged = pd.concat(df_list, ignore_index=True)
merged.to_csv(output_combined, index=False)
print(f"\n✅ Combined CSV saved: {output_combined}")
print("   Total rows:", len(merged))

# --- Step 2: Recompute statistics ---
stats = {
    "total_articles": len(merged),
    "by_language": {},
    "by_period": {"pre-chatgpt": 0, "post-chatgpt": 0},
    "by_year": {},
    "date_range": {"earliest": None, "latest": None},
    "token_statistics": {"mean": 0, "median": 0, "min": 0, "max": 0}
}

token_counts = []
dates = []

for _, row in merged.iterrows():
    lang = row["language"]
    stats["by_language"][lang] = stats["by_language"].get(lang, 0) + 1
    period = row["period"]
    stats["by_period"][period] = stats["by_period"].get(period, 0) + 1
    year = int(row["year"])
    stats["by_year"][year] = stats["by_year"].get(year, 0) + 1
    # Parse date safely
    try:
        d = datetime.strptime(str(row["publication_date"])[:10], "%Y-%m-%d")
        dates.append(d)
    except Exception:
        pass
    token_counts.append(int(row["token_count"]))

# Token stats
if token_counts:
    stats["token_statistics"]["mean"] = round(statistics.mean(token_counts), 2)
    stats["token_statistics"]["median"] = statistics.median(token_counts)
    stats["token_statistics"]["min"] = min(token_counts)
    stats["token_statistics"]["max"] = max(token_counts)

# Date range
if dates:
    stats["date_range"]["earliest"] = min(dates).strftime("%Y-%m-%d")
    stats["date_range"]["latest"] = max(dates).strftime("%Y-%m-%d")

with open(output_stats, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

print(f"\n✅ Combined statistics saved: {output_stats}")
print("\n📊 Summary:")
for lang, count in stats["by_language"].items():
    print(f"  {lang.title()}: {count:,}")
print(f"  Total: {stats['total_articles']:,}")
print(f"  Pre-ChatGPT: {stats['by_period']['pre-chatgpt']:,}")
print(f"  Post-ChatGPT: {stats['by_period']['post-chatgpt']:,}")
print(f"  Date Range: {stats['date_range']['earliest']} → {stats['date_range']['latest']}")
