import os
import base64
import pandas as pd
from dotenv import load_dotenv
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()
BASE_DIR = r"D:\mridul\Scraping Assessment\Task 2"
POPPLER_BIN = os.path.join(BASE_DIR, "poppler", "Library", "bin")
PDF_PATH = os.path.join(BASE_DIR, "Service Record.pdf")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ocr_engine = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, 
                       use_textline_orientation=False, device="gpu")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

class ServiceEntry(BaseModel):
    # Mapping Pydantic fields to your exact required CSV headers
    Full_Name: str = Field(alias="Full Name")
    Educational_Qualification: Optional[str] = Field(alias="Educational Qualification")
    Honorific_Title: Optional[str] = Field(alias="Honorific/Title")
    Date_of_Birth: str = Field(alias="Date of Birth")
    Date_of_Joining_Service: str = Field(alias="Date of Joining Service")
    Date_of_Arrival: str = Field(alias="Date of Arrival")
    Voted_Non_voted: str = Field(alias="Voted/Non-voted")
    Domicile: str = Field(alias="Domicile")
    Station: str = Field(alias="Station")
    Substantive_Appointment: str = Field(alias="Substantive Appointment")
    Subst_Date: str = Field(alias="Subst. Date")
    Officiating_Appointment: Optional[str] = Field(alias="Officiating Appointment")
    Off_Date: Optional[str] = Field(alias="Off. Date")

    model_config = ConfigDict(populate_by_name=True)

class ServiceRecordDataset(BaseModel):
    entries: List[ServiceEntry]

structured_llm = llm.with_structured_output(ServiceRecordDataset)

def get_page_ocr_text(img_path):
    """Executes PaddleOCR 3.x and extracts text with robust error handling."""
    result = ocr_engine.predict(img_path)
    raw_text_lines = []
    for res in result:
        data = res.json
        if 'res' in data and isinstance(data['res'], list):
            raw_text_lines.extend([item['text'] for item in data['res'] if 'text' in item])
    return "\n".join(raw_text_lines)

def process_single_page(page_index):
    """Encapsulates image conversion, OCR, and Gemini extraction for one page."""
    images = convert_from_path(PDF_PATH, dpi=300, first_page=page_index+1, 
                               last_page=page_index+1, poppler_path=POPPLER_BIN)
    if not images: return []
    
    img_path = os.path.join(OUTPUT_DIR, f"temp_p{page_index}.png")
    images[0].save(img_path)
    
    # Run OCR
    raw_text = get_page_ocr_text(img_path)
    
    # Prepare Multimodal Input for Gemini
    with open(img_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")
        
    prompt = f"""
    Extract service record data into the specified structured format.
    
    COLUMNS: Full Name, Educational Qualification, Honorific/Title, Date of Birth, 
             Date of Joining Service, Date of Arrival, Voted/Non-voted, Domicile, 
             Station, Substantive Appointment, Subst. Date, Officiating Appointment, Off. Date.

    RULES:
    1. REPLICABILITY: Always repeat the 'Full Name' and 'Date of Birth' for every row.
    2. DITTO LOGIC: Replace "Do.", "..", or empty cells with the value from the row directly above.
    3. DATE FORMAT: Normalize dates to DD-MM-YYYY.

    OCR TEXT:
    {raw_text}
    """
    
    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
    ])
    
    try:
        response = structured_llm.invoke([message])
        return [e.model_dump(by_alias=True) for e in response.entries]
    except Exception as e:
        print(f"Error on page {page_index+1}: {e}")
        return []

def run_extraction(start_p, end_p, output_filename="Service_Record_Dataset.csv"):
    """Orchestrates extraction across a range of pages and saves to CSV."""
    all_rows = []
    for i in range(start_p - 1, end_p):
        print(f"Processing Page {i+1}...")
        rows = process_single_page(i)
        all_rows.extend(rows)
        
    if not all_rows:
        print("No data extracted.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    
    # Ensure columns match your required header list exactly
    required_cols = [
        "Full Name", "Educational Qualification", "Honorific/Title", "Date of Birth", 
        "Date of Joining Service", "Date of Arrival", "Voted/Non-voted", "Domicile", 
        "Station", "Substantive Appointment", "Subst. Date", "Officiating Appointment", "Off. Date"
    ]
    df = df[required_cols] 
    
    df.to_csv(os.path.join(BASE_DIR, output_filename), index=False)
    print(f"Done! {len(df)} rows saved to {output_filename}")
    return df

if __name__ == "__main__":
    test_results = run_extraction(start_p=8, end_p=8, output_filename="Page_8_Test.csv")
    print(test_results.head())

    test_results = run_extraction(start_p=8, end_p=9, output_filename="Page_8_9_Test.csv")
    print(test_results.head())
