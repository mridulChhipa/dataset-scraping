# FIBIS Archival Records Web Scraper

## Task 1: Web Scraping (FIBIS Archival Records)

**Website:** https://search.fibis.org/bin/recordslist.php

This project implements a scalable, fully automated web scraping pipeline for extracting datasets from the FIBIS (Families In British India Society) archival records portal.

---

## 📊 Selected Datasets (10 Total)

The following 10 datasets were scraped, organized by their FIBIS category hierarchy:

| # | Dataset Name | Records | Category |
|---|-------------|---------|----------|
| 1 | Jhansi Lychgate Burial Register | 1,698 | Bengal Burials |
| 2 | Saharanpur Burials | 368 | Bengal Burials |
| 3 | St John the Baptist Armenian Apostolic Church, Rangoon - Burial Register | 426 | Bengal Burials |
| 4 | Register of Burials at Cinnamara, Assam and Outstations 1939-1959 | 22 | Bengal Burials |
| 5 | Transcription of assorted entries from the Burial Indexes 1800-1947 | 1,312 | Bengal Burials |
| 6 | Transcription of assorted entries from the Burial Indexes 1800-1947 | 569 | Bombay Burials |
| 7 | Cape Town, St. George's Cathedral (1796 - 1830) | 49 | Burials Outside India |
| 8 | Chandernagore Civil Death Indexes (1831-1864) | 1,008 | Chandernagore Civil |
| 9 | Civil Registration of Deaths in Chandernagore (1865-1899) | 403 | Chandernagore Civil |
| 10 | All Souls Church Coimbatore (1872-2015) Burial Register - Index of names | 207 | Madras Burials |

**Total Records Scraped: 6,062**

> **Note:** Datasets #5 and #6 have the same name but are from different categories (Bengal vs Bombay Burials). The scraper correctly saves them in separate folders based on their category hierarchy.

---

## 🔧 How the Scraper Works

### Pagination Handling

FIBIS uses a `st=` (start) parameter for pagination, displaying 30 records per page:

| Feature | Implementation |
|---------|---------------|
| **URL Parameter** | Uses `st=` parameter (e.g., `?st=0`, `?st=30`, `?st=60`) |
| **Records Per Page** | 30 records per page |
| **Next Page Detection** | Parses HTML for navigation links containing higher `st=` values |
| **Continuation Logic** | Continues fetching until no more "next" pages are found |
| **Safety Limit** | Maximum 1,000 pages per dataset to prevent infinite loops |

### Rate Limiting

To be respectful to the FIBIS server and avoid being blocked:

- **Request Delay:** 1.5 seconds between each HTTP request
- **Polite Scraping:** Mimics human browsing behavior with realistic User-Agent headers

### Error Handling

The scraper implements robust error handling with retry logic:

| Error Type | Handling Strategy |
|-----------|------------------|
| **Network Timeout** | 30-second timeout per request |
| **HTTP Errors (4xx, 5xx)** | Retry up to 3 times with exponential backoff |
| **Connection Errors** | Wait and retry with increasing delays (5s, 10s, 15s) |
| **Parse Errors** | Log warning, skip problematic record, continue with next |
| **Empty Data** | Skip empty records, log for review |
| **Dataset Failure** | Log error, continue with next dataset (graceful degradation) |

---

## 📁 Output Folder Structure

The scraper organizes output using the FIBIS category hierarchy to prevent overwrites:

```
datasets/
└── Birth Marriage & Deaths/
    └── Deaths & Burials/
        ├── Bengal Burials/
        │   ├── Jhansi Lychgate Burial Register/
        │   │   ├── Jhansi Lychgate Burial Register.csv
        │   │   └── metadata.json
        │   ├── Saharanpur Burials/
        │   ├── St John the Baptist Armenian.../
        │   ├── Register of Burials at Cinnamara.../
        │   └── Transcription of assorted entries.../
        ├── Bombay Burials/
        │   └── Transcription of assorted entries.../  (different from Bengal!)
        ├── Burials Outside India/
        │   └── Cape Town, St. George's Cathedral.../
        ├── Chandernagore Civil/
        │   ├── Chandernagore Civil Death Indexes.../
        │   └── Civil Registration of Deaths.../
        └── Madras Burials/
            └── All Souls Church Coimbatore.../
```

Each dataset folder contains:
- `[Dataset Name].csv` - The scraped data
- `metadata.json` - Scraping metadata (timestamps, record counts, URLs)

---

## 🚀 How to Run the Pipeline End-to-End

### Prerequisites

1. **Python 3.8+** installed
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Option 1: Run via Jupyter Notebook (Recommended)

1. Open the notebook:
   ```bash
   jupyter notebook FIBIS_Scraper.ipynb
   ```

2. Run all cells in order:
   - **Cell 1-2:** Import libraries and configure settings
   - **Cell 3-4:** Initialize scraper and discover datasets
   - **Cell 5:** View all available datasets
   - **Cell 6:** Select 10 datasets for scraping
   - **Cell 7:** Run the scraping pipeline
   - **Cell 8-10:** Verify output and view results

### Option 2: Run Programmatically

```python
from fibis_scraper import FIBISScraper

# Initialize scraper
scraper = FIBISScraper(output_dir="datasets")

# Discover all available datasets
all_datasets = scraper.get_available_datasets()

# Select first 10 datasets
selected = all_datasets[:10]

# Run the pipeline
results = scraper.run_pipeline(datasets_to_scrape=selected)

# Check results
for name, path in results.items():
    print(f"{name}: {path}")
```

---

## 📋 Project Files

| File | Description |
|------|-------------|
| `fibis_scraper.py` | Main scraper module with all scraping logic |
| `FIBIS_Scraper.ipynb` | Jupyter notebook to run the pipeline interactively |
| `requirements.txt` | Python package dependencies |
| `datasets/` | Output folder with scraped data (organized by category) |

---

## ⚙️ Configuration Options

The scraper can be configured with these parameters:

```python
scraper = FIBISScraper(
    output_dir="datasets",      # Output directory path
)

# Pipeline options
scraper.run_pipeline(
    datasets_to_scrape=selected,  # List of Dataset objects
    max_records_per_dataset=None  # Optional limit per dataset
)
```

### Constants (in fibis_scraper.py)

| Constant | Default | Description |
|----------|---------|-------------|
| `REQUEST_DELAY` | 1.5s | Delay between requests |
| `TIMEOUT` | 30s | Request timeout |
| `MAX_RETRIES` | 3 | Retry attempts on failure |
| `RETRY_DELAY` | 5s | Base delay for exponential backoff |

---

## 📝 Notes

- The scraper respects the website's structure and rate limits
- All dates and timestamps are recorded in metadata.json
- The category hierarchy ensures datasets with the same name but different categories are saved separately
- Scraping 10 datasets takes approximately 10-30 minutes depending on dataset sizes
