import os
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
import base64
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from langchain_core.messages import HumanMessage

# 1. Load variables from .env
load_dotenv() 

# Define absolute paths based on your provided structure
BASE_DIR = r"D:\mridul\Scraping Assessment\Task 2"
POPPLER_BIN = os.path.join(BASE_DIR, "poppler", "Library", "bin")
PDF_PATH = os.path.join(BASE_DIR, "Service Record.pdf")

# Create an output directory for PaddleOCR debug files
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Paths confirmed.\nPoppler: {POPPLER_BIN}\nPDF: {PDF_PATH}")

# 2. Initialize Gemini 2.5 Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

# 3. Define the Schema
class ServiceEntry(BaseModel):
    full_name: str = Field(description="Name of the officer")
    edu_qual: Optional[str] = Field(description="Qualifications (e.g., B.A., Oxon.)")
    title: Optional[str] = Field(description="Honorific/Title (e.g., C.B.)")
    dob: str = Field(description="Date of Birth (DD-MM-YYYY)")
    joining_date: str = Field(description="Date of Joining Service (DD-MM-YYYY)")
    arrival_date: str = Field(description="Date of Arrival (DD-MM-YYYY)")
    voted_status: str = Field(description="Voted or Non-voted")
    domicile: str = Field(description="Domicile (e.g., Non-Asiatic)")
    station: str = Field(description="Station/Location name. Handle 'Do.' as ditto.")
    subst_appointment: str = Field(description="Substantive Appointment. Handle 'Do.' as ditto.")
    subst_date: str = Field(description="Substantive Appointment Date")
    off_appointment: Optional[str] = Field(description="Officiating Appointment (if any)")
    off_date: Optional[str] = Field(description="Officiating Appointment Date")

class ServiceRecordDataset(BaseModel):
    entries: List[ServiceEntry]

structured_llm = llm.with_structured_output(ServiceRecordDataset)
print("Step 1 Complete: Environment and Schema ready.")

# 1. Initialize PaddleOCR 3.x with your specific settings + GPU
# Ensure 'device' is set to 'gpu' for your hardware acceleration
ocr_engine = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="gpu" # Explicitly use your GPU support
)

def get_ocr_and_image(pdf_path, page_index):
    # Convert specific page to image
    images = convert_from_path(
        pdf_path, 
        first_page=page_index+1, 
        last_page=page_index+1, 
        poppler_path=POPPLER_BIN
    )
    page_image = images[0]
    
    # Save image for PaddleOCR and as a source for Gemini
    img_path = f"page_{page_index}.png"
    page_image.save(img_path)

    # 2. Execute PaddleOCR 3.x predict logic
    result = ocr_engine.predict(img_path)
    
    raw_text_lines = []
    for res in result:
        # res.print() # Optional: prints to console for your tracking
        res.save_to_img("output") # Saves visualized results to output/
        res.save_to_json("output") # Saves structured JSON to output/
        
        # Access the recognized text from the result object
        # In PaddleOCR 3.x 'predict', the result structure provides convenient access
        for line in res.json['res']:
            raw_text_lines.append(line['text'])

    raw_text = "\n".join(raw_text_lines)
    
    # Encode image for Gemini Multimodal input
    with open(img_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")
        
    return raw_text, img_base64, img_path

print("Step 2 Complete: GPU-accelerated OCR function initialized.")

def run_full_extraction(pdf_path):
    # 1. Determine total pages
    # We use a small trick to just get the count without loading all images to RAM
    temp_images = convert_from_path(pdf_path, dpi=10, poppler_path=POPPLER_BIN)
    total_pages = len(temp_images)
    del temp_images
    
    final_data = []

    print(f"Starting extraction for {total_pages} pages...")

    for i in range(total_pages):
        print(f"Processing Page {i+1}/{total_pages}...")
        
        # Get OCR text and Image Base64 from our Step 2 function
        raw_text, img_base64, _ = get_ocr_and_image(pdf_path, i)

        # Call Gemini (Multimodal)
        prompt = f"""
        Extract the service record data from this page into a list of entries.
        
        CRITICAL REPLICABILITY RULES:
        1. COLUMN ALIGNMENT: Use the image layout to ensure 'Station' and 'Appointment' are separated correctly.
        2. DITTO LOGIC: Replace 'Do.', '..', or '"' with the value from the row directly above it.
        3. DATE FORMAT: Ensure all dates are DD-MM-YYYY.
        4. PERSISTENCE: Every row must have the officer's Full Name.
        
        PAGE OCR TEXT:
        {raw_text}
        """

        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
        ])

        try:
            # Structured output returns our Pydantic model
            response = structured_llm.invoke([message])
            
            # Convert Pydantic objects to dictionaries for the list
            page_entries = [entry.model_dump() for entry in response.entries]
            final_data.extend(page_entries)
            
        except Exception as e:
            print(f"Error on page {i+1}: {e}")

    # 2. Final Dataset Compilation
    df = pd.DataFrame(final_data)
    
    # Save to CSV
    output_filename = "Service_Record_Dataset.csv"
    df.to_csv(output_filename, index=False)
    
    return df

def test_page_8():
    print("Extracting Page 8...")
    
    # 1. Convert ONLY Page 8 to image (1-indexed)
    pages = convert_from_path(
        PDF_PATH, 
        dpi=300, 
        first_page=8, 
        last_page=8, 
        poppler_path=POPPLER_BIN
    )
    
    if not pages:
        print("Page 8 not found.")
        return

    page_image = pages[0]
    img_path = os.path.join(BASE_DIR, "test_page_8.png")
    page_image.save(img_path)

    # 2. Run PaddleOCR Predict (GPU-accelerated)
    result = ocr_engine.predict(img_path)
    
    raw_text_lines = []
    for res in result:
        # Save debug files as per your preference
        res.save_to_json(os.path.join(BASE_DIR, "output"))
        
        # FIX: Correctly access the text data in PaddleOCR 3.x
        # res.json is a dictionary in 3.x. We access 'res' -> 'doc_res' or 'res'
        data = res.json
        if 'res' in data and isinstance(data['res'], list):
            for item in data['res']:
                # The text is usually under the 'text' key
                if 'text' in item:
                    raw_text_lines.append(item['text'])
    
    raw_text = "\n".join(raw_text_lines)

    # 3. Process with Gemini 2.5 Flash
    import base64
    from langchain_core.messages import HumanMessage

    with open(img_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    # This prompt is tailored to your sample Abraham entries
    prompt = f"""
    Extract the service records from this page. 
    Format: CSV with the exact headers provided in the sample.
    
    SPECIAL INSTRUCTION:
    If you see "Do.", inherit the value from the row above.
    Ensure 'Abraham, Edgar Garton Furtado' is repeated for every row if it belongs to him.

    OCR TEXT FOR REFERENCE:
    {raw_text}
    """
    
    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
    ])

    response = structured_llm.invoke([message])
    
    # 4. Convert and Display
    test_df = pd.DataFrame([e.model_dump() for e in response.entries])
    print("\n--- Page 8 Extraction Results ---")
    print(test_df)
    return test_df

if __name__ == "__main__":
    # Execute
    page_8_results = test_page_8()
