# summarize_doc.py
"""
Module for extracting text from PDF, DOCX, and TXT files
and summarizing them using an OpenAI-compatible model.
"""
import os
import re
import time
import email
from email import policy
from email.parser import BytesParser
import tempfile
import pdfplumber
import pytesseract
from PIL import Image
from langdetect import detect          # currently unused → keep if using Arabic text
from docx import Document

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill


from dotenv import load_dotenv
load_dotenv()
from llm_client import get_chat_model, get_llm_client, unwrap_llm_text


VERBOSE_MODE = __name__ == "__main__"  # True only if run locall

def log(*args, **kwargs):
    if VERBOSE_MODE:
        print(*args, **kwargs)


def clean_model_markdown(text: str) -> str:
    """Remove server-log lines if terminal output was mixed into model text."""
    kept = []
    for line in (text or "").splitlines():
        value = line.strip()
        if value.startswith("INFO:") or value.startswith("[LLM]"):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def extract_document_metadata(text: str) -> dict:
    """Extract high-value quotation metadata before the LLM compares items."""
    source = text or ""
    metadata = {
        "customer": "",
        "quotation_number": "",
        "possible_supplier": "",
        "payment_terms": "",
        "validity": "",
    }

    customer = re.search(r"(?im)^\s*Customer Name\s*[:/]?\s*(.+)$", source)
    if customer:
        metadata["customer"] = customer.group(1).strip()

    quotation = re.search(r"(?im)\b(?:Number|Qtn\s*No)\s*:?\s*([A-Z0-9][A-Z0-9-]+)", source)
    if quotation:
        metadata["quotation_number"] = quotation.group(1).strip()

    account = re.search(r"(?im)ACCOUNT NAME\s*:\s*([^|\n]+)", source)
    regards = re.search(r"(?ims)WITH BEST REGARDS,?\s*\n\s*([^\n]+)", source)
    if account:
        metadata["possible_supplier"] = account.group(1).strip()
    elif regards:
        candidate = regards.group(1).strip()
        # Prefer the complete company/location phrase on the right side of
        # pipe-separated extraction output, not a truncated normalized token.
        if "|" in candidate:
            candidate = candidate.split("|")[-1].strip()
        metadata["possible_supplier"] = re.split(r"\s+-\s+", candidate, maxsplit=1)[0].strip()

    payment = re.search(r"(?im)(?:\*\s*)?Payment\s*:\s*([^\n]+)", source)
    if payment:
        metadata["payment_terms"] = payment.group(1).split("|")[-1].strip()

    validity = re.search(r"(?im)(?:\*\s*)?Validity\s*:\s*([^\n]+)", source)
    if validity:
        metadata["validity"] = validity.group(1).split("|")[-1].strip()

    return metadata


def format_document_for_llm(filename: str, text: str) -> str:
    metadata = extract_document_metadata(text)
    metadata_lines = "\n".join(f"{key}: {value or 'Not found'}"
                                for key, value in metadata.items())
    return (f"\n\n--- BEGIN DOCUMENT: {filename} ---\n"
            f"DOCUMENT METADATA:\n{metadata_lines}\n"
            f"DOCUMENT TEXT:\n{text}\n--- END DOCUMENT: {filename} ---")


# ────────────────────────────────────────────────────────────────────────────────
# 1.  Initialise the OpenAI-compatible client
# ────────────────────────────────────────────────────────────────────────────────

MODEL = "gpt-5.6"


# ───────────────────────────────────────────────────────────────────
# 2.  Helpers to extract raw-text *and* run the description normaliser
# ───────────────────────────────────────────────────────────────────

CODE_RE   = re.compile(r"^[A-Z0-9\-\/]+$")   # first token looks like a code
BRAND_RE  = re.compile(r"^[A-Z]{3,}$")       # second token looks like BRAND


ADJECTIVES = {"SHUTTERING", "MARINE", "ORD", "INDONESIAN", "SMARTPLEX"}

# Generic numeric pair   4*8  4-8  4×8  4 X 8  →  4X8
DIM_RE = re.compile(
    r"\b(\d+)\s*[×xX*–—-]\s*(\d+)\b",         # accepts hyphen & long dash
    flags=re.UNICODE,
)


# Millimetre pair        150MM-12MM / 150MM*12 / 150 MM – 12 mm → 150MMX12MM
MM_DIM_RE = re.compile(
    r"\b(\d+)MM\s*[×xX*–—-]\s*(\d+)(?:MM)?\b",   # 2nd “MM” optional
    flags=re.UNICODE | re.IGNORECASE,
)

MM_RE    = re.compile(r"\b(\d+)\s*MM\b", re.I)                # 18 mm → 18MM


LONG_DIM_RE = re.compile(
    r"\b\d+(?:\s*[×xX*–—-]\s*\d+){2,}\b",          # ≥2 repetitions
    flags=re.UNICODE,
)

def _fold_long_dims(m: re.Match) -> str:
    """'18 * 122 * 244' → '18X122X244'."""
    nums = re.findall(r"\d+", m.group(0))
    return "X".join(nums)

def normalise_description(line: str) -> str:
    """(unchanged – keeps human wording)"""
    tokens = line.strip().split()
    if tokens and CODE_RE.match(tokens[0]):
        tokens = tokens[1:]
        if tokens and BRAND_RE.match(tokens[0]):
            tokens = tokens[1:]
    desc = " ".join(tokens)
    return re.sub(r"\s{2,}", " ", desc).strip().upper()


def canonical_key(desc: str) -> str:
    desc = desc.upper()
    desc = re.sub(r"[,/]+", " ", desc)

    # 1️⃣ 18 mm  → 18MM
    desc = MM_RE.sub(lambda m: f"{m.group(1)}MM", desc)

    # 2️⃣-a generic 4×8 pair
    desc = DIM_RE.sub(lambda m: f"{m.group(1)}X{m.group(2)}", desc)

    # 2️⃣-b long sequences 18*122*244
    desc = LONG_DIM_RE.sub(_fold_long_dims, desc)

    # 2️⃣-c 150MM-12MM pair
    desc = MM_DIM_RE.sub(lambda m: f"{m.group(1)}MMX{m.group(2)}MM", desc)

    # 3️⃣ drop filler adjectives / brands
    tokens = [t for t in desc.split()
              if t not in ADJECTIVES and not BRAND_RE.match(t)]
    return " ".join(tokens).strip()



def extract_text_from_pdf(path: str) -> str:
    """
    Per-page extraction strategy:
    1. Extract free text with pdfplumber (normalised through canonical_key).
    2. Extract tables with pdfplumber (rows joined as pipe-separated text).
    3. If a page yields nothing from either route, fall back to Tesseract OCR.
    """
    out = []

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_lines = []

            # 1. Text layer
            raw = page.extract_text() or ""
            for ln in raw.splitlines():
                pretty = normalise_description(ln)
                if not pretty:
                    continue
                key = canonical_key(pretty)
                page_lines.append(f"{key} | {pretty}")

            # 2. Structured tables (critical for price quotations)
            tables = page.extract_tables() or []
            for table in tables:
                for row in table:
                    if not row:
                        continue
                    row_text = " | ".join(str(cell or "").strip() for cell in row)
                    if row_text.strip(" |"):
                        page_lines.append(row_text)

            # 3. OCR fallback (only when page has no selectable content)
            if not page_lines:
                try:
                    img = page.to_image(resolution=300).original
                    ocr_raw = pytesseract.image_to_string(img, config="--psm 6 --oem 3")
                    for ln in ocr_raw.splitlines():
                        pretty = normalise_description(ln)
                        if pretty:
                            key = canonical_key(pretty)
                            page_lines.append(f"{key} | {pretty}")
                except Exception:
                    pass

            # Preserve page boundaries so the model and downstream reviewers can
            # identify where each extracted value came from.
            if page_lines:
                out.append(f"--- PAGE {page.page_number or len(out) + 1} ---")
                out.extend(page_lines)

    return "\n".join(out)


def extract_text_from_txt(path: str) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as file_obj:
                return file_obj.read()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("text", b"", 0, 1, "Unable to decode text file")


def extract_text_from_docx(path: str) -> str:
    document = Document(path)
    parts = []
    # Paragraphs
    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            parts.append(paragraph.text)
    # Tables (quotation documents often use Word tables for prices)
    for table in document.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                parts.append(row_text)
    return "\n".join(parts)


def extract_text_from_xlsx(path: str) -> str:
    """Extract worksheet names and rows while preserving column boundaries."""
    workbook = load_workbook(path, read_only=True, data_only=True)
    parts = []
    try:
        for worksheet in workbook.worksheets:
            parts.append(f"--- SHEET: {worksheet.title} ---")
            for row in worksheet.iter_rows(values_only=True):
                values = ["" if value is None else str(value).strip() for value in row]
                if any(values):
                    parts.append(" | ".join(values))
    finally:
        workbook.close()
    return "\n".join(parts)


def _extract_email_part(part) -> str:
    if part.get_content_disposition() == "attachment":
        filename = part.get_filename() or "attachment"
        extension = os.path.splitext(filename)[1].lower()
        supported = {".pdf", ".docx", ".txt", ".xlsx", ".eml"}
        if extension not in supported:
            return ""

        data = part.get_payload(decode=True) or b""
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temporary:
            temporary.write(data)
            temporary_path = temporary.name
        try:
            return f"\n--- ATTACHMENT: {filename} ---\n{extract_text_from_file(temporary_path)}"
        except Exception:
            return ""
        finally:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass

    if part.is_multipart():
        return "\n".join(_extract_email_part(child) for child in part.iter_parts())
    if part.get_content_type() == "text/plain":
        return part.get_content()
    return ""


def extract_text_from_eml(path: str) -> str:
    """Extract EML headers, body text, and supported quotation attachments."""
    with open(path, "rb") as file_obj:
        message = BytesParser(policy=policy.default).parse(file_obj)

    parts = []
    for header in ("subject", "from", "to", "date"):
        value = message.get(header)
        if value:
            parts.append(f"{header.title()}: {value}")
    parts.append(_extract_email_part(message))
    return "\n".join(part for part in parts if part).strip()


def extract_text_from_msg(path: str) -> str:
    """
    Extract text from an Outlook .msg email file.
    Also recursively extracts any PDF / DOCX / TXT attachments embedded in the email.
    """
    try:
        import extract_msg
    except ImportError:
        raise ValueError("extract-msg is not installed. Run: pip install extract-msg")

    with extract_msg.Message(path) as msg:
        parts = []

        if msg.subject:
            parts.append(f"Subject: {msg.subject}")
        if msg.sender:
            parts.append(f"From: {msg.sender}")
        if msg.date:
            parts.append(f"Date: {msg.date}")
        body = (msg.body or "").strip()
        if body:
            parts.append(body)

        for att in (msg.attachments or []):
            name = att.longFilename or att.shortFilename or ""
            ext = os.path.splitext(name)[1].lower()
            if ext not in (".pdf", ".docx", ".txt", ".xlsx", ".eml"):
                continue
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    tmp.write(att.data)
                    tmp_path = tmp.name
                att_text = extract_text_from_file(tmp_path)
                if att_text.strip():
                    parts.append(f"\n--- ATTACHMENT: {name} ---\n{att_text}")
            except Exception:
                pass
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

    return "\n".join(parts)


def extract_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext == ".txt":
        return extract_text_from_txt(path)
    if ext == ".docx":
        return extract_text_from_docx(path)
    if ext == ".xlsx":
        return extract_text_from_xlsx(path)
    if ext == ".msg":
        return extract_text_from_msg(path)
    if ext == ".eml":
        return extract_text_from_eml(path)
    raise ValueError(f"Unsupported file type: {ext}")


# ────────────────────────────────────────────────────────────────────────────────
# 3.  Prompt & call GPT
# ────────────────────────────────────────────────────────────────────────────────

def summarize_text_with_gpt(combined_text: str) -> str:
    """
    Feed the concatenated raw text from all quotation files to GPT and
    receive two markdown tables:
       1) Material & Supplier Comparison  (Unit-price only, no totals)
       2) Payment Terms & Quotation Validity
    """
    prompt = f"""
You are an evidence-first quotation analyst. Use only facts explicitly present in the supplied documents.

------------------------------------------------------------------------------
### 1  📦 Material & Supplier Comparison (Unit Price only)
Canonical Key (CK) ✨  
• Before tabulating, treat the **left-hand token** (everything before the first
  “|”) as the CK.  It is already normalised in Python so, for example,  
  `150MM-12MM`, `150 MM – 12 MM`, `150MM*12MM` ⇒ **CK `150MMX12MM`**  
  and `4 * 8`, `4×8`, `4-8` ⇒ **CK `4X8`** and `244 * 122 * 18` ⇒ **CK `244X122X18`**.  
• Every distinct CK is **one row**.  Do **not** merge rows with different CKs.

Table rules  
• Columns **exactly**:  
  `SN | Altered Material Name | Material description | <Supplier-1> Unit Price | <Supplier-2> Unit Price | … | Source file(s)`  
• If a supplier did **not** quote that item → Unit Price `N/A`.  
• Never invent, estimate, calculate, or copy a price from another item. Preserve unclear
  values as written or use `N/A`.
• Match items only when description, dimensions, grade, packaging, and unit are compatible.
  Similar names are not enough; keep potentially different items as separate rows.
• Do not assume a filename is the supplier name unless the document identifies the supplier.
• No totals, no VAT, no grand totals, no commentary.  
• Output valid markdown.

### 2  💳 Payment Terms & Quotation Validity
Columns: `Supplier Name | Payment Terms | Quotation Validity`  
Copy wording exactly; leave blank if absent.


--- BEGIN DOCUMENTS ---
{combined_text}
--- END DOCUMENTS ---
"""
    print(f"[LLM] Sending request to model '{get_chat_model(MODEL)}' …")
    _t0 = time.perf_counter()
    stream = get_llm_client().chat.completions.create(
        model=get_chat_model(MODEL),
        stream=True,
        messages=[
            {
                "role": "system",
                "content": 
                (
                "You are an evidence-first document summariser. Extract quotation facts exactly "
                "as written and never invent prices, units, suppliers, terms, or dates. "
                "quotations into markdown tables. For every material in a row, list the unit "
                "price/Rate quoted by each supplier across the documents. "
                "Documents may be bilingual (Arabic + English) or contain pipe-separated rows — treat them all as valid input. "
                "Preserve currency symbols and units (SAR, USD, pcs, sheets, m², kg, etc.) exactly as written. "
                "Make sure ALL materials are listed even if not comparable across suppliers. "
                "If documents conflict, preserve both values and their source files. "
                "Identify the supplier from each document header, footer, logo, address, email, "
                "or quotation metadata. Repeat the supplier name on every payment row. Never "
                "use a quotation number as the supplier name when a company name is available. "
                "No extra prose — output only the required tables."
                )
            },
            {"role": "user", "content": prompt},
        ],
    )
    chunks = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            print(delta, end="", flush=True)
            chunks.append(delta)
    print()  # newline after stream ends
    print(f"[LLM] Response received in {time.perf_counter() - _t0:.2f}s")
    return clean_model_markdown(unwrap_llm_text("".join(chunks)))


# ────────────────────────────────────────────────────────────────────────────────
# 4.  Markdown  →  DataFrame (robust version, keeps empty cells)
# ────────────────────────────────────────────────────────────────────────────────

def markdown_table_to_df(md_block: str) -> pd.DataFrame:
    """
    Convert a pipe-delimited markdown table to DataFrame **without**
    discarding empty cells.  Leading / trailing pipes are trimmed once so
    column counts remain stable.
    """
    lines = [ln.rstrip() for ln in md_block.splitlines() if "|" in ln]

    if len(lines) < 2:
        raise ValueError("No markdown table found.")

    def split_row(row: str) -> list[str]:
        row = row.strip()
        if row.startswith("|"):
            row = row[1:]
        if row.endswith("|"):
            row = row[:-1]
        return [c.strip() for c in row.split("|")]

    header = split_row(lines[0])
    if len(header) < 2 or not any("supplier" in c.lower() for c in header):
        raise ValueError("Invalid quotation table header.")
    # Skip the separator row (---|---) which is always the 2nd line
    body   = [split_row(r) for r in lines[2:] if r.strip()]

    # Pad rows that are shorter than header with empty strings
    fixed_body = [r[:len(header)] + [""] * (len(header) - len(r))
                  for r in body]

    return pd.DataFrame(fixed_body, columns=header)


# ───────────────────────────────────────────────────────────────────────────────────────────
# 5.  Run_local_test: read all files in folder and print the markdown summary
# ───────────────────────────────────────────────────────────────────────────────────────────

def run_local_test():
    folder = input("📁 Folder with quotation files: ").strip()
    if not os.path.isdir(folder):
        log("❌ Folder not found.")
        return

    EXT = {".pdf", ".docx", ".txt", ".xlsx", ".msg", ".eml"}
    files = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in EXT]
    if not files:
        log("⚠️ No supported files in folder.")
        return

    log(f"✅ Found {len(files)} file(s):", *files, sep="\n  • ")

    combined = ""
    for f in files:
        path = os.path.join(folder, f)
        log(f"   ─ reading {f}")
        try:
            txt = extract_text_from_file(path)
            combined += f"\n\n--- FILE: {f} ---\n\n{txt}"
        except Exception as exc:
            log(f"     ⚠️ {f}: {exc}")

    if not combined.strip():
        log("❌ No text extracted.")
        return

    log("🧠 Sending to GPT …")
    md_summary = summarize_text_with_gpt(combined)
    log("✅ GPT returned markdown tables\n")

    log(md_summary)               # show in console


if __name__ == "__main__":

    run_local_test()


