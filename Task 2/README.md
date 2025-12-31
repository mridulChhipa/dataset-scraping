# Documentation for Task 2: PDF-to-Dataset Extraction

## Overview
This project extracts structured service record data from a PDF file (`Service Record.pdf`) and compiles it into a CSV dataset (`extracted_dataset.csv`) that aligns with a provided sample structure (`output_format.csv`).

## Methodology
The solution uses a two-step process:
1.  **OCR (Optical Character Recognition):** PaddleOCR is used to extract raw text from the PDF pages. The PDF pages are first converted to images using `pdf2image`.
2.  **LLM Parsing:** The extracted raw text is sent to Google's Gemini 2.5 Flash model. The LLM is prompted to parse the unstructured text into a JSON format, mapping the data to the required columns.

## Assumptions
*   **PDF Structure:** The PDF contains service records where each entry generally follows a pattern of personal details followed by a list of appointments.
*   **"Do." References:** The text frequently uses "Do." or "do." to indicate "Ditto" (same as above). It is assumed that the LLM can contextually resolve these references to the previous valid value for the same person.
*   **Date Formats:** Dates in the PDF are often in `D-M-YY` format. The system assumes these belong to the 19th or 20th century based on context (e.g., birth dates vs. service dates) and standardizes them to `DD-MM-YYYY`.
*   **Column Mapping:** The columns in `output_format.csv` are the definitive target schema.

## Edge Cases Handled
*   **Rotated Text:** PaddleOCR is initialized with `use_angle_cls=True` to handle potential text rotation in the scanned images.
*   **Broken Structure:** The LLM is specifically instructed to handle the "broken structure" of the PDF, where tables might not have clear borders or alignment.
*   **Missing Data:** The prompt instructs the LLM to use null or empty strings for missing fields rather than hallucinating data.
*   **JSON Parsing:** The code includes logic to clean markdown formatting (e.g., ```json ... ```) from the LLM response before parsing it.

## Dependencies
*   `paddlepaddle-gpu` (or `paddlepaddle` for CPU)
*   `paddleocr`
*   `google-generativeai`
*   `pandas`
*   `pdf2image`
*   `poppler` (binary required for pdf2image)

### Configuration Notes
*   **CUDA/GPU Support:** The code is configured to use `device="gpu"` for PaddleOCR.
    *   If you have a compatible NVIDIA GPU, ensure you have the correct CUDA and cuDNN libraries installed.
    *   If you are running on **CPU**, you must modify the `PaddleOCR` initialization in the notebook/script to set `device="cpu"`.
*   **Poppler:** `pdf2image` requires Poppler to be installed and accessible.
    *   Windows: Download the binary, extract it, and ensure the `bin` folder is in your system PATH or pointed to by the `POPPLER_PATH` variable in the code.
    *   Linux/Mac: Install via package manager (e.g., `sudo apt-get install poppler-utils` or `brew install poppler`).

## Usage
1.  Open `extraction.ipynb`.
2.  Install dependencies (first cell).
3.  Set your Google Gemini API key in the configuration section.
4.  Run all cells to generate `extracted_dataset.csv`.
