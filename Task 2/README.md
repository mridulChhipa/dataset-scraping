# Task 2: Service Record Extraction

This directory contains the solution for extracting service record entries from a PDF into a structured CSV format. The solution leverages OCR and Large Language Models (LLMs) to handle complex layouts and data spanning across pages.

## Method Used

1.  **PDF to Image Conversion**: The PDF pages are converted into images using `pdf2image` (Poppler).
2.  **OCR (Optical Character Recognition)**: `PaddleOCR` is used to extract raw text from the page images. This text provides spatial reference and content for the LLM.
3.  **LLM Extraction with LangChain**: 
    - We use `LangChain` to interface with Google's Gemini 2.5 Flash model.
    - A **Context-Aware** approach is implemented. The extraction function for a page accepts the last record from the previous page as context. This allows the model to correctly attribute rows at the start of a new page to the correct officer if the record spans across the page break.
    - **Structured Output**: Pydantic models (`ServiceEntry`, `ServiceRecordDataset`) are used to enforce a strict schema for the extracted data, ensuring consistency with the required CSV headers.

## Assumptions & Edge Cases Handled

*   **Page Spanning Records**: It is assumed that a service record for an individual can start on one page and continue to the next. The solution handles this by passing the details of the last officer from Page `N` to the prompt for Page `N+1`.
*   **Ditto Marks**: The source text often uses "Do.", "..", or '"' to indicate repetition. The LLM is explicitly instructed to resolve these to the actual values from the row above.
*   **Date Normalization**: Dates are normalized to the `DD-MM-YYYY` format (e.g., converting "8-4-80" to "08-04-1880").
*   **Missing Headers**: If a page starts with table rows but no officer name header, the system uses the "Continuing Record" context to fill in the identity fields (`Full Name`, `Educational Qualification`, etc.).

## Limitations

*   **Dataset Scope**: The current extraction and testing are limited to **Page 8 and Page 9** of the PDF. This restriction is due to API key usage limits during development.
If you have an api key with enough usage available just change the start page `start_p` and end page `end_p` in `run_extraction()` to generate dataset from the complete pdf.
*   **Image Input**: Currently, the image input to the LLM is commented out to save on token usage/bandwidth, relying primarily on the OCR text injected into the prompt. Enabling the image input could potentially improve accuracy for very complex visual layouts.
*   **Paddle OCR Accuracy Limits**: Using MistralAI's OCR can improve accuracy by a good marging and would eliminate use of both Gemini and PaddleOCR. But Since MistralAI's OCR is paid I have not used it here. MistralAI's OCR will work in one go and do task of both gemini and paddleocr. Currently sending image to gemini directly eliminates use of PaddleOCR.

## Files

### Notebooks
*   `Context Aware Extration.ipynb`: The main notebook implementing the context-aware extraction logic (handling page breaks).
*   `Improved Extraction.ipynb`: An iteration containing improvements to the extraction prompts and logic.
*   `Service_Record_Extraction.ipynb`: The initial notebook for setting up the extraction pipeline.

### Output CSVs
*   `Page_8_9_ContextAware_Text_only.csv`: Result of the context-aware extraction on pages 8 & 9 (using text-only prompt).
*   `Page_8_9_ContextAware.csv`: Result of context-aware extraction using both image and ocr text.
*   `Page_8_9_Test.csv`: Test output for pages 8 & 9 without context awareness from previous page.
*   `Page_8_Test.csv`: Test output for page 8.
*   `output_format.csv`: Template or example of the required output format.

## Code Snippet: HumanMessage

The solution uses `HumanMessage` from LangChain to construct the prompt. Note the commented-out image input, which can be enabled for multimodal capabilities:

```python
message = HumanMessage(content=[
    {"type": "text", "text": prompt}
    # {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
])
```

## Setup & Configuration

If you are cloning this repository to run the code locally, please ensure you configure the following:

### 1. Environment Variables
Create a `.env` file in the `Task 2` directory. You need to add your Google Gemini API key:
```
GOOGLE_API_KEY=your_api_key_here
```

### 2. Poppler Setup
The project uses `pdf2image` which requires **Poppler**.
- A local version of Poppler is included in the `poppler/` folder within this directory.
- The notebooks are configured to look for the Poppler binary at:
  ```python
  POPPLER_BIN = os.path.join(BASE_DIR, "poppler", "Library", "bin")
  ```
- If you prefer to use a system-installed Poppler, you can modify the `POPPLER_BIN` path in the notebooks or remove the `poppler_path` argument in the `convert_from_path` function calls.

### 3. Path Configuration
In the notebooks (e.g., `Context Aware Extration.ipynb`), there is a `BASE_DIR` variable defined at the top:
```python
BASE_DIR = r"D:\mridul\Scraping Assessment\Task 2"
```
**You must update this path** to match the location where you have cloned/downloaded the repository on your machine. For example:
```python
BASE_DIR = os.getcwd() # If running from the directory itself
# OR
BASE_DIR = r"C:\Users\YourName\Projects\Scraping Assessment\Task 2"
```

## Requirements

*   Python 3.x
*   `langchain`, `langchain-google-genai`, `langchain-core`
*   `paddleocr`, `paddlepaddle`, `paddlepaddle-gpu`
*   `pdf2image`
*   `pydantic`
*   `python-dotenv`
