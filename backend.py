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
    extract_text_from_file, format_document_for_llm, summarize_text_with_gpt, markdown_table_to_df, 
)
from supplier_summary import generate_supplier_summary_excel
from models import ProcurementRequest
from punch_ai_risk import build_insights, Payload as RiskPayload
from structured_quotations import analyze_files, to_markdown


TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".msg", ".eml", ".xlsx"}

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
    logger.info("FUNCTION: health_check")
    return {"status": "ok"}
    
  
class ChatQuery(BaseModel):
    query: str
    context: str
    
    
# ─────────────────────────────────────────────────────────────
# Main Endpoint
# ─────────────────────────────────────────────────────────────

@app.post("/summarize-quotations/")
async def summarize_quotations(files: list[UploadFile] = File(...)):
    logger.info("FUNCTION: summarize_quotations")
    combined_text = ""
    saved_files = []
    direct_file_types = {".pdf", ".docx"}
    direct_files = []
    extracted_documents = []
    logger.info(f"📥 Received {len(files)} file(s)")

    for uploaded_file in files:
        extension = os.path.splitext(uploaded_file.filename or "")[1].lower()
        if extension not in SUPPORTED_EXTENSIONS:
            logger.warning("Ignoring unsupported file: %s", uploaded_file.filename)
            continue
        temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_{uploaded_file.filename}")
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(uploaded_file.file, f)

        try:
            saved_files.append((temp_path, uploaded_file.filename or "quotation"))
            txt = extract_text_from_file(temp_path)
            combined_text += format_document_for_llm(uploaded_file.filename, txt)
            if extension in direct_file_types:
                direct_files.append((temp_path, uploaded_file.filename or "quotation"))
                logger.info("Document %s: original file queued for GPT (%d locally extracted chars available as fallback)",
                            uploaded_file.filename, len(txt))
            else:
                extracted_documents.append((uploaded_file.filename or "quotation", txt))
                logger.info("Document %s: extracted locally and queued as text for GPT (%d chars)",
                            uploaded_file.filename, len(txt))
        except ValueError as exc:
            for path, _ in saved_files:
                if os.path.exists(path):
                    os.remove(path)
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception:
            for path, _ in saved_files:
                if os.path.exists(path):
                    os.remove(path)
            raise

    if not saved_files:
        return JSONResponse(
            {"error": "No supported quotation files were uploaded."},
            status_code=400
        )

    logger.info("🧠 Sending text to GPT summarizer")
    if direct_files:
        try:
            logger.info("Sending original PDF/DOCX files plus extracted documents to GPT")
            logger.info("Original files sent: %s", [name for _, name in direct_files])
            logger.info("Locally extracted files sent as text: %s", [name for name, _ in extracted_documents])
            structured = analyze_files(direct_files, extracted_documents)
            md_summary = to_markdown(structured)
        except Exception:
            logger.exception("Direct file analysis failed; using extracted-text fallback")
            md_summary = summarize_text_with_gpt(combined_text)
    else:
        logger.info("Using local extraction for mixed/native file types")
        logger.info("Locally extracted files sent to GPT as combined text: %s",
                    [name for name, _ in extracted_documents])
        md_summary = summarize_text_with_gpt(combined_text)
    for path, _ in saved_files:
        if os.path.exists(path):
            os.remove(path)


    # Convert markdown tables to JSON
    summary_blocks = [b for b in md_summary.split("\n\n") if "|" in b and "-" in b]
    tables_json = []
    for block in summary_blocks:
        try:
            df = markdown_table_to_df(block)
            tables_json.append(df.fillna("").to_dict(orient="records"))
        except ValueError:
            logger.warning("Model returned an invalid markdown table; skipping it")
    
    logger.info(f"✅ Returning {len(tables_json)} summary table(s)")

    return {
        "summary_tables": tables_json,
        "summary_markdown": md_summary,
    }  
    
@app.post("/generate-supplier-summary/")
async def generate_supplier_summary(data: ProcurementRequest):
    logger.info("FUNCTION: generate_supplier_summary")
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
    logger.info("FUNCTION: chat_about_quotation")
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
    logger.info("FUNCTION: summarize_ai_risk")
    return build_insights(payload)

