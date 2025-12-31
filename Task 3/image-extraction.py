import os
import base64
import pandas as pd
from dotenv import load_dotenv
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict
from paddleocr import PaddleOCR
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv() 

BASE_DIR = os.getenv("BASE_DIR", ".")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', device="gpu")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

class ArmyEntry(BaseModel):
    Name: str = Field(alias="Name")
    Rank: str = Field(alias="Rank")
    Rank_Date: str = Field(alias="Rank Date")
    Corps: str = Field(alias="Corps")
    Remarks: Optional[str] = Field(default="", alias="Remarks")
    
    model_config = ConfigDict(populate_by_name=True)

class ArmyRecordDataset(BaseModel):
    entries: List[ArmyEntry]

def get_page_ocr_text(img_path):
    """Executes PaddleOCR 3.x and extracts text based on detected rec_texts key."""
    # Corrected function as per your request
    result = ocr_engine.predict(img_path)
    raw_text_lines = []
    
    for res in result:
        if 'rec_texts' in res and isinstance(res['rec_texts'], list):
            raw_text_lines.extend(res['rec_texts'])
        
        res.save_to_json(OUTPUT_DIR)
            
    raw_text = "\n".join(raw_text_lines)
    print(f"Extracted {len(raw_text_lines)} lines of text.")
    return raw_text

def process_army_image(img_path):
    """Orchestrates extraction from a single image file."""
    raw_text = get_page_ocr_text(img_path)
    
    with open(img_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = f"""
    TASK: Extract the 'Army List' tabular data into a structured dataset.
    
    EXTRACTION RULES:
    1. Resolve 'do', 'ditto', or '"' by using the value from the row directly above.
    2. Resolve footnote symbols (*, †, ‡) using definitions at the bottom (e.g., * = C.B.).
    3. For 'Colonels' with a vertical bracket, apply '5 June 1829' to all associated names.
    4. Ensure every row has a 'Name', 'Rank', and 'Rank Date'.
    
    OCR TEXT FOR CONTEXT:
    {raw_text}
    """
    
    structured_llm = llm.with_structured_output(ArmyRecordDataset)
    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
    ])
    
    response = structured_llm.invoke([message])
    return [e.model_dump(by_alias=True) for e in response.entries]

image_paths = ["task3_1.png", "task3_2.png"]
all_extracted_rows = []

for path in image_paths:
    if os.path.exists(path):
        print(f"Processing: {path}")
        rows = process_army_image(path)
        all_extracted_rows.extend(rows)

df = pd.DataFrame(all_extracted_rows)
csv_output = os.path.join(BASE_DIR, "Army_List_Dataset.csv")
df.to_csv(csv_output, index=False)
print(f"Extraction Complete. Saved to {csv_output}")