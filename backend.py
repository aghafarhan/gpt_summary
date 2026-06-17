# backend.py
"""
FastAPI backend for summarizing quotations from PDF, DOCX, and TXT files.
It extracts text and summarizes it using an OpenAI-compatible model.
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
from llm_client import get_chat_model, get_llm_client, unwrap_llm_text
from summarize_doc import (
    extract_text_from_file, summarize_text_with_gpt, markdown_table_to_df, 
)
from supplier_summary import generate_supplier_summary_excel
from models import ProcurementRequest
from punch_ai_risk import build_insights, Payload as RiskPayload


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
            txt = extract_text_from_file(temp_path)
            combined_text += f"\n\n--- FILE: {uploaded_file.filename} ---\n\n{txt}"
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        finally:
            os.remove(temp_path)

    if not combined_text.strip():
        return JSONResponse({"error": "No text extracted."}, status_code=400)

    logger.info("🧠 Sending text to GPT summarizer")
    md_summary = summarize_text_with_gpt(combined_text)


    # Convert markdown tables to JSON
    summary_blocks = [b for b in md_summary.split("\n\n") if "|" in b and "-" in b]
    tables_json = []
    for block in summary_blocks:
        df = markdown_table_to_df(block)
        tables_json.append(df.fillna("").to_dict(orient="records"))
    
    logger.info(f"✅ Returning {len(tables_json)} summary table(s)")

    return {
        "summary_tables": tables_json,
        "summary_markdown": md_summary,
    }  
    
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
        response = get_llm_client().chat.completions.create(
            model=get_chat_model("gpt-5.4"),
            messages=[{"role": "user", "content": prompt}]
        )
        return {"answer": unwrap_llm_text(response.choices[0].message.content)}
    except Exception as e:
        return {"error": str(e)}   

@app.post("/punch-ai-risk/")
async def summarize_ai_risk(payload: RiskPayload):
    return build_insights(payload)

