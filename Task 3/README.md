# Task 3: Army List Dataset Extraction

## Overview
This project extracts structured tabular data from images of "Army List" records (e.g., `task3_1.png`, `task3_2.png`) and compiles it into a CSV dataset (`Army_List_Dataset.csv`).

## Methodology
The solution employs a multimodal approach:
1.  **OCR (Optical Character Recognition):** `PaddleOCR` is used to extract raw text from the images. It is configured to use GPU for performance and handles text rotation (`use_angle_cls=True`).
2.  **Multimodal LLM Parsing:** The extracted text and the original image are sent to Google's **Gemini 2.5 Flash** model. The model is prompted to:
    *   Parse the visual and textual data into a structured JSON format.
    *   Resolve contextual references like "do", "ditto", or quotes (`"`) to the value in the row above.
    *   Resolve footnote symbols (e.g., `*`, `†`) using definitions found in the image.

## Assumptions
*   **Input Files:** The script looks for `task3_1.png` and `task3_2.png` in the root of the task folder.
*   **Data Structure:** The target schema includes `Name`, `Rank`, `Rank Date`, `Corps`, and `Remarks`.
*   **Contextual Resolution:** It is assumed that the LLM can correctly interpret visual cues for grouped data (e.g., vertical brackets indicating a shared date for multiple names).

## Edge Cases Handled
*   **"Ditto" Marks:** The prompt explicitly instructs the model to resolve "do" and ditto marks to the previous valid entry.
*   **Footnotes:** Symbols like `*` are mapped to their definitions (e.g., `* = C.B.`).
*   **Vertical Grouping:** Cases where a single date applies to a bracketed group of names are handled by the LLM's visual understanding.

## Dependencies & Setup

### Python Packages
Install the required packages:
```bash
pip install paddlepaddle-gpu paddleocr langchain-google-genai pandas python-dotenv pydantic
```
*Note: If you do not have a CUDA-enabled GPU, install `paddlepaddle` instead of `paddlepaddle-gpu`.*

### External Dependencies
*   **CUDA (Optional but Recommended):** The script is configured to use `device="gpu"`.
    *   **GPU Users:** Ensure you have the correct CUDA and cuDNN versions installed for your PaddlePaddle version.
    *   **CPU Users:** If you encounter errors or lack a GPU, modify the `PaddleOCR` initialization in `image-extraction.py`:
        ```python
        ocr_engine = PaddleOCR(..., device="cpu")
        ```

### Configuration
1.  Create a `.env` file in the project root.
2.  Add your Google API key:
    ```
    GOOGLE_API_KEY=your_api_key_here
    ```

## Usage
Run the extraction script:
```bash
python image-extraction.py
```
The output will be saved to `Army_List_Dataset.csv`.
