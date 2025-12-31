"""
FIBIS Archival Records Scraper
==============================
Website: https://search.fibis.org/bin/recordslist.php
"""

import os
import re
import csv
import json
import time
import logging
import requests
from typing import List, Dict, Optional, Tuple, Any
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    """Represents a dataset available on FIBIS"""
    name: str
    url: str
    record_count: int
    index_access: str
    dataset_access: str
    category: str = ""
    dataset_id: str = ""  # The dataset ID from URL


class FIBISScraper:
    """
    Scraper for FIBIS archival records portal.
    
    Handles:
    - Dataset discovery from the main records list
    - Individual record extraction with pagination
    - Error handling and retries
    - Rate limiting
    - Output organization
    """
    
    BASE_URL = "https://search.fibis.org"
    RECORDS_LIST_URL = "https://search.fibis.org/bin/recordslist.php"
    
    # Rate limiting settings
    REQUEST_DELAY = 1.5  # seconds between requests
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds before retry
    TIMEOUT = 30  # request timeout in seconds
    
    def __init__(self, output_dir: str = "datasets"):
        """
        Initialize the scraper.
        
        Args:
            output_dir: Base directory for saving scraped datasets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        })
        
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: dict = None) -> Optional[requests.Response]:
        """
        Make an HTTP request with retry logic.
        
        Args:
            url: URL to request
            params: Optional query parameters
            
        Returns:
            Response object or None if all retries failed
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                self._rate_limit()
                response = self.session.get(url, params=params, timeout=self.TIMEOUT)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.MAX_RETRIES}): {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"All retries failed for URL: {url}")
                    return None
        return None
    
    def get_available_datasets(self) -> List[Dataset]:
        """
        Scrape the main records list page to get all available datasets.
        
        Returns:
            List of Dataset objects with metadata
        """
        logger.info("Fetching available datasets from FIBIS...")
        
        response = self._make_request(self.RECORDS_LIST_URL)
        if not response:
            raise Exception("Failed to fetch records list page")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        datasets = []
        
        # Build a map of categories from the page structure
        # FIBIS displays categories as section headers with browse_classes links
        # Category rows contain links like "Birth Marriage & Deaths > Deaths & Burials > Bengal Burials"
        # We need to track which category each dataset belongs to
        category_map = {}  # Maps dataset_id -> category path
        
        # Get all rows and iterate to track current category
        current_category = ""
        all_rows = soup.find_all('tr')
        
        for row in all_rows:
            # Check if this is a category row (has browse_classes links)
            class_links = row.find_all('a', href=lambda h: h and 'browse_classes' in h if h else False)
            if class_links:
                # Build category from link texts (each link is a level in the hierarchy)
                category_parts = [link.get_text(strip=True) for link in class_links]
                current_category = ' > '.join(category_parts)
            
            # Check if this row has a dataset link
            dataset_link = row.find('a', href=lambda h: h and 'browse_dataset' in h if h else False)
            if dataset_link and current_category:
                href = dataset_link.get('href', '')
                id_match = re.search(r'id=(\d+)', href)
                if id_match:
                    dataset_id = id_match.group(1)
                    category_map[dataset_id] = current_category
        
        logger.info(f"Mapped {len(category_map)} datasets to categories")
        
        # FIBIS uses links with "mode=browse_dataset" in href
        # The dataset name is in the parent element (td), not in the link text
        dataset_links = soup.find_all('a', href=re.compile(r'browse_dataset'))
        
        logger.info(f"Found {len(dataset_links)} dataset links to process")
        
        for link in dataset_links:
            try:
                href = link.get('href', '')
                
                # Get the dataset URL
                dataset_url = urljoin(self.BASE_URL + "/bin/", href)
                
                # Extract dataset ID from URL
                id_match = re.search(r'id=(\d+)', href)
                dataset_id = id_match.group(1) if id_match else ""
                
                # Skip if no valid dataset ID (required for unique identification)
                if not dataset_id:
                    continue
                
                # Get the parent cell/element which contains the dataset name
                parent = link.parent
                if parent:
                    dataset_name = parent.get_text(strip=True)
                else:
                    dataset_name = link.get_text(strip=True)
                
                # Clean up the name - remove record count and extra text
                dataset_name = re.sub(r'\d{1,3}(?:,\d{3})*\s*records?.*$', '', dataset_name, flags=re.I).strip()
                dataset_name = re.sub(r'Index access:.*$', '', dataset_name, flags=re.I).strip()
                
                # Skip if no name found
                if not dataset_name:
                    continue
                
                # Get record count from the next sibling cell (td)
                record_count = 0
                parent_td = link.find_parent('td')
                if parent_td:
                    next_td = parent_td.find_next_sibling('td')
                    if next_td:
                        cell_text = next_td.get_text(strip=True)
                        record_match = re.search(r'(\d{1,3}(?:,\d{3})*|\d+)\s*records?\b', cell_text, re.I)
                        if record_match:
                            record_str = record_match.group(1).replace(',', '')
                            record_count = int(record_str)
                
                # Get the row for additional context (access info)
                row = link.find_parent('tr')
                row_text = row.get_text(strip=True) if row else ""
                
                # Extract access information
                index_access = "publicly available" if "publicly available" in row_text.lower() else "members only"
                dataset_access = "publicly available" if "dataset access: publicly available" in row_text.lower() else "publicly available"  # Default to public for this site
                
                # Skip if members only
                if "members only" in row_text.lower() and "dataset access:" in row_text.lower():
                    dataset_access = "members only"
                    continue
                
                # Get category from the category map we built earlier
                category = category_map.get(dataset_id, "")
                
                datasets.append(Dataset(
                    name=dataset_name,
                    url=dataset_url,
                    record_count=record_count,
                    index_access=index_access,
                    dataset_access=dataset_access,
                    category=category,
                    dataset_id=dataset_id
                ))
                
            except Exception as e:
                logger.warning(f"Error parsing dataset link: {e}")
                continue
        
        # Remove duplicates (same dataset_id)
        seen_ids = set()
        unique_datasets = []
        for ds in datasets:
            if ds.dataset_id not in seen_ids:
                seen_ids.add(ds.dataset_id)
                unique_datasets.append(ds)
        
        datasets = unique_datasets
        logger.info(f"Found {len(datasets)} unique publicly available datasets")
        return datasets
    
    def _sanitize_filename(self, name: str) -> str:
        """
        Sanitize a string to be safe for use as a filename.
        Preserves spaces but removes/replaces invalid characters.
        
        Args:
            name: Original filename
            
        Returns:
            Sanitized filename
        """
        # Replace invalid characters but keep spaces
        invalid_chars = '<>:"/\\|?*'
        sanitized = name
        for char in invalid_chars:
            sanitized = sanitized.replace(char, '_')
        # Remove leading/trailing spaces and dots
        sanitized = sanitized.strip('. ')
        
        return sanitized
    
    def scrape_dataset(self, dataset: Dataset) -> Tuple[List[Dict], List[str]]:
        """
        Scrape all records from a single dataset, handling pagination.
        
        Args:
            dataset: Dataset object with URL and metadata
            
        Returns:
            Tuple of (list of record dictionaries, list of column headers)
        """
        logger.info(f"Scraping dataset: {dataset.name} ({dataset.record_count} records)")
        
        all_records = []
        headers = []
        page = 1
        records_per_page = 30  # FIBIS uses 30 records per page
        
        while True:
            # Construct URL with pagination
            # FIBIS uses 'st' parameter for pagination (start offset)
            start_offset = (page - 1) * records_per_page
            
            # Build the full URL with parameters
            base_url = dataset.url
            if 'st=' in base_url:
                # Remove existing st parameter
                base_url = re.sub(r'st=\d+', '', base_url)
                base_url = re.sub(r'&&', '&', base_url)  # Clean up double ampersands
            
            if '?' in base_url:
                page_url = f"{base_url}&st={start_offset}" if start_offset > 0 else base_url
            else:
                page_url = f"{base_url}?st={start_offset}" if start_offset > 0 else base_url
            
            response = self._make_request(page_url)
            if not response:
                logger.error(f"Failed to fetch page {page} of dataset: {dataset.name}")
                break
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try multiple parsing strategies
            page_records, page_headers = self._parse_records_from_page(soup, headers)
            
            if page_headers and not headers:
                headers = page_headers
            
            if not page_records:
                if page == 1:
                    logger.warning(f"No records found for dataset: {dataset.name}")
                break
            
            all_records.extend(page_records)
            logger.info(f"  Page {page}: {len(page_records)} records (total: {len(all_records)})")
            
            if len(page_records) < records_per_page:
                # Last page (fewer records than expected)
                break
            
            if not self._has_next_page(soup, page, len(all_records)):
                break
            
            page += 1
            
            # Safety limit
            if page > 1000:
                logger.warning("Reached page limit (1000), stopping")
                break
        
        logger.info(f"Completed scraping {dataset.name}: {len(all_records)} records extracted")
        return all_records, headers
    
    def _parse_records_from_page(self, soup: BeautifulSoup, existing_headers: List[str]) -> Tuple[List[Dict], List[str]]:
        """
        Parse records from a page using multiple strategies.
        
        Args:
            soup: BeautifulSoup object of the page
            existing_headers: Previously found headers
            
        Returns:
            Tuple of (records list, headers list)
        """
        records = []
        headers = existing_headers.copy() if existing_headers else []
        
        # This strategy is unique to the FIBIS website
        # I inspect the HTML structure and found that all the dataset names and records
        # are contained within HTML tables.
        tables = soup.find_all('table')
        
        # Find the best candidate table
        best_table = None
        best_score = 0
        
        for table in tables:
            rows = table.find_all('tr')
            if len(rows) < 3:
                continue
            
            # Skip tables that contain nested tables (layout tables)
            if table.find('table'):
                continue
            
            # Check first row for header-like content
            first_row = rows[0]
            header_cells = first_row.find_all(['th', 'td'])
            if not header_cells:
                continue
            
            # Score the table based on header-like content
            header_texts = [c.get_text(strip=True).lower() for c in header_cells]
            score = 0
            
            # FIBIS common headers
            fibis_headers = ['surname', 'name', 'names', 'date', 'place', 'view', 'cemetery', 
                           'rank', 'regiment', 'year', 'birth', 'death', 'marriage', 'age',
                           'first name', 'father', 'mother', 'spouse', 'occupation', 'ship',
                           'arrival', 'departure', 'reference', 'page', 'volume']
            
            for header in header_texts:
                if any(h in header for h in fibis_headers):
                    score += 10
            
            # Prefer tables with more rows (more data)
            score += len(rows)
            
            # Prefer tables where data rows have similar cell count to header
            num_header_cols = len(header_cells)
            if len(rows) > 1:
                data_row = rows[1]
                data_cells = data_row.find_all('td')
                if len(data_cells) == num_header_cols:
                    score += 20
            
            if score > best_score:
                best_score = score
                best_table = table
        
        if best_table:
            records, headers = self._parse_table(best_table, existing_headers)
            if records:
                return records, headers
        
        # No suitable table found or no records parsed, try alternative strategies
        # This was added because I can't guarantee all datasets will use tables
        # So alternative strategies suggested by LLMs are implemented below
        # Strategy 2: Look for list-based records (dl/dt/dd or div-based)
        definition_lists = soup.find_all('dl')
        for dl in definition_lists:
            record = {}
            dts = dl.find_all('dt')
            dds = dl.find_all('dd')
            for dt, dd in zip(dts, dds):
                key = dt.get_text(strip=True).rstrip(':')
                val = dd.get_text(strip=True)
                if key:
                    record[key] = val
                    if key not in headers:
                        headers.append(key)
            if record:
                records.append(record)
        
        if records:
            return records, headers
        
        # Strategy 3: Look for div-based record containers
        record_divs = soup.find_all('div', class_=re.compile(r'record|result|entry|item', re.I))
        for div in record_divs:
            record = self._parse_record_div(div)
            if record:
                records.append(record)
                for key in record.keys():
                    if key not in headers:
                        headers.append(key)
        
        # Strategy 4: Look for spans with data-field attributes or similar
        if not records:
            data_spans = soup.find_all(['span', 'div'], attrs={'data-field': True})
            if data_spans:
                record = {}
                for span in data_spans:
                    key = span.get('data-field', '')
                    val = span.get_text(strip=True)
                    if key:
                        record[key] = val
                        if key not in headers:
                            headers.append(key)
                if record:
                    records.append(record)
        
        return records, headers
    
    def _parse_table(self, table: Any, existing_headers: List[str]) -> Tuple[List[Dict], List[str]]:
        """
        Parse records from an HTML table.
        
        Args:
            table: BeautifulSoup table element
            existing_headers: Previously found headers
            
        Returns:
            Tuple of (records list, headers list)
        """
        records = []
        headers = existing_headers.copy() if existing_headers else []
        rows = table.find_all('tr')
        
        if not rows:
            return records, headers
        
        # Determine header row
        first_row = rows[0]
        header_cells = first_row.find_all(['th', 'td'])
        
        # Check if first row is actually headers (contains th or looks like headers)
        is_header_row = bool(first_row.find_all('th'))
        if not is_header_row:
            # Heuristic: if first row cells are short text and repeated pattern, it's likely headers
            cell_texts = [c.get_text(strip=True) for c in header_cells]
            if all(len(t) < 50 for t in cell_texts) and len(set(cell_texts)) == len(cell_texts):
                is_header_row = True
        
        if not headers:
            if is_header_row:
                headers = [c.get_text(strip=True) or f"Column_{i}" for i, c in enumerate(header_cells)]
                rows = rows[1:]  # Skip header row
            else:
                # Generate default headers
                headers = [f"Column_{i}" for i in range(len(header_cells))]
        else:
            # Skip first row if it matches existing headers
            first_row_texts = [c.get_text(strip=True) for c in header_cells]
            if first_row_texts == headers[:len(first_row_texts)]:
                rows = rows[1:]
        
        # Parse data rows
        for row in rows:
            cells = row.find_all('td')
            if not cells:
                continue
            
            record = {}
            for i, cell in enumerate(cells):
                if i < len(headers):
                    # Get text content, handling links and nested elements
                    text = cell.get_text(strip=True)
                    # Also get any link href if present
                    link = cell.find('a', href=True)
                    if link and 'view' in link.get('href', '').lower():
                        # This might be a "view record" link, get the text
                        text = link.get_text(strip=True) or text
                    record[headers[i]] = text
            
            # Only add non-empty records
            if any(v.strip() for v in record.values() if v):
                records.append(record)
        
        return records, headers
    
    def _parse_record_div(self, div: Any) -> Dict:
        """
        Parse a record from a div element.
        
        Args:
            div: BeautifulSoup div element
            
        Returns:
            Record dictionary
        """
        record = {}
        
        # Try label/value pairs
        labels = div.find_all(['label', 'dt', 'span'], class_=re.compile(r'label|field|key', re.I))
        for label in labels:
            key = label.get_text(strip=True).rstrip(':')
            # Find the associated value
            value_elem = label.find_next_sibling(['span', 'dd', 'div'])
            if value_elem:
                val = value_elem.get_text(strip=True)
                if key:
                    record[key] = val
        
        return record
    
    def _has_next_page(self, soup: BeautifulSoup, current_page: int = 1, total_records: int = 0) -> bool:
        """
        Check if there's a next page of results.
        
        Args:
            soup: BeautifulSoup object of current page
            current_page: Current page number
            total_records: Total records scraped so far
            
        Returns:
            True if next page exists
        """
        records_per_page = 30
        
        # FIBIS specific: Look for "Next" link with st= parameter
        next_links = soup.find_all('a', string=re.compile(r'Next', re.I))
        for link in next_links:
            href = link.get('href', '')
            if href and 'st=' in href:
                return True
        
        # Also check for numbered pagination links with higher st values
        current_start = (current_page - 1) * records_per_page
        page_links = soup.find_all('a', href=re.compile(r'st=\d+'))
        
        for link in page_links:
            href = link.get('href', '')
            match = re.search(r'st=(\d+)', href)
            if match:
                start = int(match.group(1))
                if start > current_start:
                    return True
        
        # Check tn= parameter which shows total number of records
        tn_match = soup.find('a', href=re.compile(r'tn=\d+'))
        if tn_match:
            href = tn_match.get('href', '')
            match = re.search(r'tn=(\d+)', href)
            if match:
                total_available = int(match.group(1))
                if total_records < total_available:
                    return True
        
        return False
    
    def save_dataset(self, dataset: Dataset, records: List[Dict], headers: List[str]) -> str:
        """
        Save scraped records to a CSV file in a properly named folder.
        Uses the FIBIS category hierarchy as folder structure to prevent
        datasets with the same name but different categories from overwriting each other.
        
        Args:
            dataset: Dataset metadata
            records: List of record dictionaries
            headers: List of column headers
            
        Returns:
            Path to saved file
        """
        # This was done to prevent overwriting files when datasets have the same name
        # I observed that some datasets share names but belong to different categories
        # Like Bombay Burials and Bengal Burials under different parent categories

        # Create folder structure from category hierarchy
        # Category format: "Birth Marriage & Deaths > Deaths & Burials > Bengal Burials"
        # Convert to folder path: "Birth Marriage & Deaths/Deaths & Burials/Bengal Burials"
        if dataset.category:
            # Split category by " > " and sanitize each part for filesystem
            category_parts = [self._sanitize_filename(part.strip()) for part in dataset.category.split(' > ')]
            category_path = self.output_dir
            for part in category_parts:
                category_path = category_path / part
        else:
            # Fallback to root output directory if no category
            category_path = self.output_dir
        
        # Create folder with exact dataset name (sanitized for filesystem)
        folder_name = self._sanitize_filename(dataset.name)
        folder_path = category_path / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV with exact dataset name
        file_name = f"{folder_name}.csv"
        file_path = folder_path / file_name
        
        if not records:
            logger.warning(f"No records to save for {dataset.name}")
            # Create empty file with headers
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                if headers:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
            return str(file_path)
        
        # Determine all headers from records
        if not headers:
            headers = list(records[0].keys())
        
        # Ensure all records have all headers
        all_keys = set()
        for record in records:
            all_keys.update(record.keys())
        headers = list(all_keys)
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(records)
        
        logger.info(f"Saved {len(records)} records to {file_path}")
        
        # Also save metadata
        metadata_path = folder_path / "metadata.json"
        metadata = {
            "dataset_name": dataset.name,
            "original_url": dataset.url,
            "expected_record_count": dataset.record_count,
            "actual_record_count": len(records),
            "category": dataset.category,
            "index_access": dataset.index_access,
            "dataset_access": dataset.dataset_access,
            "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return str(file_path)
    
    def select_datasets(self, datasets: List[Dataset], count: int = 10, 
                       min_records: int = 10, max_records: int = 2000) -> List[Dataset]:
        """
        Select a subset of datasets for scraping.
        
        Args:
            datasets: List of all available datasets
            count: Number of datasets to select
            min_records: Minimum record count
            max_records: Maximum record count
            
        Returns:
            List of selected datasets
        """
        # Filter by record count
        filtered = [d for d in datasets 
                   if min_records <= d.record_count <= max_records]
        
        # Sort by record count for variety
        filtered.sort(key=lambda x: x.record_count)
        
        # Select a mix of different sizes
        if len(filtered) >= count:
            # Take some small, medium, and larger datasets
            step = len(filtered) // count
            selected = [filtered[i * step] for i in range(count)]
        else:
            selected = filtered[:count]
        
        return selected
    
    def run_pipeline(self, datasets_to_scrape: List[Dataset] = None, 
                    count: int = 10) -> Dict[str, str]:
        """
        Run the complete scraping pipeline.
        
        Args:
            datasets_to_scrape: Specific datasets to scrape (optional)
            count: Number of datasets to scrape if not specified
            
        Returns:
            Dictionary mapping dataset names to output file paths
        """
        results = {}
        
        if datasets_to_scrape is None:
            # Discover and select datasets
            all_datasets = self.get_available_datasets()
            datasets_to_scrape = self.select_datasets(all_datasets, count=count)
        
        logger.info(f"Starting pipeline with {len(datasets_to_scrape)} datasets")
        
        for i, dataset in enumerate(datasets_to_scrape, 1):
            logger.info(f"[{i}/{len(datasets_to_scrape)}] Processing: {dataset.name}")
            
            try:
                records, headers = self.scrape_dataset(dataset)
                output_path = self.save_dataset(dataset, records, headers)
                results[dataset.name] = output_path
            except Exception as e:
                logger.error(f"Failed to scrape {dataset.name}: {e}")
                results[dataset.name] = f"ERROR: {str(e)}"
        
        logger.info("Pipeline completed!")
        return results


def main():
    """Main entry point for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='FIBIS Archival Records Scraper')
    parser.add_argument('--output', '-o', default='datasets',
                       help='Output directory for scraped data')
    parser.add_argument('--count', '-n', type=int, default=10,
                       help='Number of datasets to scrape')
    parser.add_argument('--list-only', action='store_true',
                       help='Only list available datasets, do not scrape')
    
    args = parser.parse_args()
    
    scraper = FIBISScraper(output_dir=args.output)
    
    if args.list_only:
        datasets = scraper.get_available_datasets()
        print(f"\nFound {len(datasets)} publicly available datasets:\n")
        for i, d in enumerate(datasets[:50], 1):  # Show first 50
            print(f"{i:3}. {d.name} ({d.record_count:,} records)")
        if len(datasets) > 50:
            print(f"... and {len(datasets) - 50} more")
    else:
        results = scraper.run_pipeline(count=args.count)
        print("\n" + "="*60)
        print("SCRAPING RESULTS")
        print("="*60)
        for name, path in results.items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
