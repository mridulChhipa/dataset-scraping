import os
import sys
import pandas as pd
from pathlib import Path
import shutil

# Assuming fibis_scraper.py is in the same directory
from fibis_scraper import FIBISScraper, Dataset

print("Imports successful!")
print(f"Working directory: {os.getcwd()}")

# Initialize scraper with output directory
OUTPUT_DIR = "datasets"
scraper = FIBISScraper(output_dir=OUTPUT_DIR)

print(f"Scraper initialized.")
print(f"Output directory: {scraper.output_dir.absolute()}")
print(f"Rate limit: {scraper.REQUEST_DELAY}s between requests")
print(f"Max retries: {scraper.MAX_RETRIES}")

# Clean up old datasets folder before re-running (optional)
datasets_path = Path("datasets")
if datasets_path.exists():
    print("Removing old datasets folder...")
    shutil.rmtree(datasets_path)
    print("Old datasets folder removed.")

scraper = FIBISScraper()

# Fetch all available datasets from FIBIS
all_datasets = scraper.get_available_datasets()

print(f"\nFound {len(all_datasets)} publicly available datasets")
print("\nSample datasets with categories:")
for i, ds in enumerate(all_datasets[:10], 1):
    print(f"  {i}. {ds.name}")
    print(f"     Category: {ds.category if ds.category else 'N/A'}")
    print(f"     Records: {ds.record_count:,} | ID: {ds.dataset_id}")

# Select the first 10 datasets directly (they are already unique by ID)
# No need to match by name - just take the dataset objects directly
# I created a category for every dataset so that organising in folder makes it easier
# This was done to ensure that datasets with same name don't overwrite each other

selected_datasets = all_datasets[:10]

print(f"{'='*80}")
print("SELECTED DATASETS FOR SCRAPING")
print(f"{'='*80}")
total_records = 0
for i, ds in enumerate(selected_datasets, 1):
    print(f"\n{i:2}. {ds.name}")
    print(f"    Category: {ds.category if ds.category else 'N/A'}")
    print(f"    Records: {ds.record_count:,} | ID: {ds.dataset_id}")
    total_records += ds.record_count

print(f"\n{'='*80}")
print(f"Total expected records: {total_records:,}")
print(f"{'='*80}")

# Run the scraping pipeline
print("Starting scraping pipeline...")
print(f"Estimated time: {len(selected_datasets) * 2}-{len(selected_datasets) * 5} minutes\n")

results = scraper.run_pipeline(datasets_to_scrape=selected_datasets)

print("\n" + "="*60)
print("SCRAPING COMPLETE!")
print("="*60)

# Display results summary
print("\nScraping Results:")
print("-" * 60)

success_count = 0
for name, path in results.items():
    if path.startswith("ERROR"):
        print(f"❌ {name}: {path}")
    else:
        success_count += 1
        print(f"✅ {name}")
        print(f"   └── {path}")

print(f"\nSuccessfully scraped: {success_count}/{len(results)} datasets")

# Verify folder structure
print("\nOutput Folder Structure:")
print("=" * 60)

output_path = Path(OUTPUT_DIR)

def print_tree(path, prefix=""):
    """Recursively print folder tree structure"""
    items = sorted(path.iterdir())
    dirs = [item for item in items if item.is_dir()]
    files = [item for item in items if item.is_file()]
    
    for file in files:
        size = file.stat().st_size
        if size > 1024:
            size_str = f"{size/1024:.1f} KB"
        else:
            size_str = f"{size} bytes"
        print(f"{prefix}📄 {file.name} ({size_str})")
    
    for i, folder in enumerate(dirs):
        is_last = (i == len(dirs) - 1)
        print(f"{prefix}📁 {folder.name}/")
        
        new_prefix = prefix + "    "
        print_tree(folder, new_prefix)

if output_path.exists():
    print_tree(output_path)
else:
    print(f"Output directory not found: {output_path}")

print("\n" + "=" * 60)
csv_files = list(output_path.rglob("*.csv")) if output_path.exists() else []
print(f"Total datasets saved: {len(csv_files)}")

print("\nSample Data Preview:")
print("=" * 60)

output_path = Path(OUTPUT_DIR)
sample_shown = False

csv_files = list(output_path.rglob("*.csv"))

if csv_files:
    csv_file = csv_files[0]
    try:
        df = pd.read_csv(csv_file)
        if len(df) > 0:
            relative_path = csv_file.relative_to(output_path)
            print(f"\nDataset: {csv_file.stem}")
            print(f"Path: {relative_path}")
            print(f"Records: {len(df)}, Columns: {len(df.columns)}")
            print(f"Columns: {list(df.columns)}")
            print("\nFirst 5 rows:")
            print(df.head())
            sample_shown = True
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")

if not sample_shown:
    print("No data to preview.")

print("\nFinal Summary:")
print("=" * 60)

total_files = 0
total_records_scraped = 0
dataset_stats = []

output_path = Path(OUTPUT_DIR)

csv_files = list(output_path.rglob("*.csv"))

for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        total_files += 1
        total_records_scraped += len(df)
        
        relative_path = csv_file.relative_to(output_path)
        category_parts = list(relative_path.parts[:-2])
        category = " > ".join(category_parts) if category_parts else "N/A"
        
        dataset_stats.append({
            'Dataset': csv_file.stem,
            'Category': category,
            'Records': len(df),
            'Columns': len(df.columns)
        })
    except Exception as e:
        pass

print(f"Total datasets scraped: {total_files}")
print(f"Total records extracted: {total_records_scraped:,}")

if dataset_stats:
    stats_df = pd.DataFrame(dataset_stats)
    print(f"\nPer-Dataset Statistics:")
    print(stats_df)
