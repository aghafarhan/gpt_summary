# backend.py
"""
FastAPI backend for summarizing quotations from PDF, DOCX, and TXT files.
It extracts text, summarizes it using GPT, and provides downloadable Excel summaries.
"""
import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import logging
import traceback
from openai import OpenAI
from summarize_doc import (
    extract_text_from_pdf, summarize_text_with_gpt,
    markdown_table_to_df, save_summary_to_excel
)
from supplier_summary import generate_supplier_summary_excel
from models import ProcurementRequest



TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI()

# ─────────────────────────────────────────────────────────────
# CORS: Allow only the ERP frontend
# ─────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ─────────────────────────────────────────────────────────────
# Health Check Endpoint
# ─────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok"}
    
  
class ChatQuery(BaseModel):
    query: str
    context: str
    
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")            
)
    
    
# ─────────────────────────────────────────────────────────────
# Main Endpoint
# ─────────────────────────────────────────────────────────────

@app.post("/summarize-quotations/")
async def summarize_quotations(files: list[UploadFile] = File(...)):
    combined_text = ""
    logger.info(f"📥 Received {len(files)} file(s)")

    for uploaded_file in files:
        temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_{uploaded_file.filename}")
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(uploaded_file.file, f)

        try:
            ext = os.path.splitext(uploaded_file.filename)[1].lower()
            if ext == ".pdf":
                txt = extract_text_from_pdf(temp_path)
            combined_text += f"\n\n--- FILE: {uploaded_file.filename} ---\n\n{txt}"
        finally:
            os.remove(temp_path)

    if not combined_text.strip():
        return JSONResponse({"error": "No text extracted."}, status_code=400)

    logger.info("🧠 Sending text to GPT summarizer")
    md_summary = summarize_text_with_gpt(combined_text)

    # Save Excel summary
    excel_filename = f"quotation_summary_{uuid.uuid4().hex[:8]}.xlsx"
    excel_filepath = os.path.join(TEMP_DIR, excel_filename)
    save_summary_to_excel(md_summary, output_path=excel_filepath)

    # Convert markdown tables to JSON
    summary_blocks = [b for b in md_summary.split("\n\n") if "|" in b and "-" in b]
    tables_json = []
    for block in summary_blocks:
        df = markdown_table_to_df(block)
        tables_json.append(df.fillna("").to_dict(orient="records"))
    
    logger.info(f"✅ Returning {len(tables_json)} summary table(s)")

    return {
        "summary_tables": tables_json,
        "excel_download_path": f"/download/{excel_filename}"
    }

# ─────────────────────────────────────────────────────────────
# Download Endpoint
# ─────────────────────────────────────────────────────────────

@app.get("/download/{filename}")
async def download_excel(filename: str):
    file_path = os.path.join(TEMP_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse({"error": "File not found."}, status_code=404)
    return FileResponse(path=file_path, filename="quotation_summary.xlsx")
    
    
@app.post("/generate-supplier-summary/")
async def generate_supplier_summary(data: ProcurementRequest):
    items = [{"item_name": item.item_name} for item in data.items]


    os.makedirs("temp_files", exist_ok=True)
    filename = f"supplier_summary_{uuid.uuid4().hex[:8]}.xlsx"
    filepath = os.path.join("temp_files", filename)

    try:
        generate_supplier_summary_excel(items, output_file=filepath)
    except Exception as e:
        traceback.print_exc()  # 🔥 this will print the full error to the terminal
        return JSONResponse(status_code=500, content={"error": str(e)})

    return FileResponse(
        filepath,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="supplier_summary.xlsx"
    )
    
@app.post("/chat-about-quotation/")
async def chat_about_quotation(payload: ChatQuery):
    prompt = f"""You are a intelligent Purchase Officer assistant. Given this quotation summary:\n\n{payload.context}\n\nAnswer this question:\n{payload.query}"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return {"answer": response.choices[0].message.content.strip()}
    except Exception as e:
        return {"error": str(e)}   
