"""Direct document analysis with a structured response and legacy rendering."""
import json
import logging
from llm_client import get_chat_model, get_llm_client

MODEL = "grok-4.6"
logger = logging.getLogger(__name__)

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "documents": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "properties": {"file": {"type": "string"}, "supplier": {"type": "string"},
                "document_type": {"type": "string"}, "related_files": {"type": "array", "items": {"type": "string"}}},
            "required": ["file", "supplier", "document_type", "related_files"]}},
        "suppliers": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "properties": {"name": {"type": "string"}, "payment_terms": {"type": "string"}, "quotation_validity": {"type": "string"},
                "source_file": {"type": "string"}, "page": {"type": "integer"}, "evidence": {"type": "string"}},
            "required": ["name", "payment_terms", "quotation_validity", "source_file", "page", "evidence"]}},
        "items": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "properties": {"name": {"type": "string"}, "description": {"type": "string"}, "unit": {"type": "string"},
                "source_files": {"type": "array", "items": {"type": "string"}},
                "page": {"type": "integer"}, "evidence": {"type": "string"},
                "prices": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                    "properties": {"supplier": {"type": "string"}, "value": {"type": "string"}, "source_file": {"type": "string"},
                        "page": {"type": "integer"}, "evidence": {"type": "string"}},
                    "required": ["supplier", "value", "source_file", "page", "evidence"]}}},
            "required": ["name", "description", "unit", "source_files", "page", "evidence", "prices"]}}
    }, "required": ["documents", "suppliers", "items"]
}

INSTRUCTION = ("Analyze the complete document set and return only JSON matching the schema. "
               "Identify each file's supplier and purpose. Do not assume every file is a separate supplier. "
               "Recognize files from the same supplier, email attachments, and supporting documents; use related_files. "
               "Compare quotations only when comparison is meaningful. Use exact facts; never guess. "
               "Identify the supplier from the document, not the customer. Keep different dimensions, grades, packaging, or units separate. "
               "Use N/A or empty strings when absent. Use only unit prices, never totals or VAT.")

def analyze_files(paths, extracted_documents=None):
    print("FUNCTION: analyze_files")
    client = get_llm_client()
    ids = []
    try:
        content = [{"type": "input_text", "text": INSTRUCTION}]
        for path, filename in paths:
            content.append({"type": "input_text", "text": f"--- ORIGINAL DOCUMENT: {filename} ---"})
            with open(path, "rb") as stream:
                uploaded = client.files.create(file=(filename, stream), purpose="assistants")
            ids.append(uploaded.id)
            content.append({"type": "input_file", "file_id": uploaded.id})
        for filename, text in extracted_documents or []:
            content.append({"type": "input_text", "text": f"--- EXTRACTED DOCUMENT: {filename} ---\n{text}"})
        selected_model = get_chat_model(MODEL)
        logger.info("LLM FUNCTION: client.responses.create model=%s direct_files=%s extracted_files=%s",
                    selected_model, [name for _, name in paths], [name for name, _ in extracted_documents or []])
        response = client.responses.create(
            model=selected_model, input=[{"role": "user", "content": content}],
            text={"format": {"type": "json_schema", "name": "quotation_summary", "strict": True, "schema": SCHEMA}},
        )
        usage = getattr(response, "usage", None)
        if usage:
            logger.info("LLM TOKENS: model=%s input=%s output=%s total=%s",
                        selected_model, getattr(usage, "input_tokens", "?"),
                        getattr(usage, "output_tokens", "?"), getattr(usage, "total_tokens", "?"))
        data = json.loads(response.output_text)
        allowed = {filename for _, filename in paths}
        allowed.update(filename for filename, _ in extracted_documents or [])
        for item in data.get("items", []):
            if not item.get("source_files") or any(name not in allowed for name in item["source_files"]):
                item["source_files"] = [name for name in item.get("source_files", []) if name in allowed]
            for price in item.get("prices", []):
                if price.get("source_file") not in allowed:
                    price["value"] = "N/A"
                    price["source_file"] = ""
                    price["evidence"] = ""
        for supplier in data.get("suppliers", []):
            if supplier.get("source_file") not in allowed:
                supplier["source_file"] = ""
                supplier["page"] = 0
                supplier["evidence"] = ""
        return data
    finally:
        for file_id in ids:
            try:
                client.files.delete(file_id)
            except Exception:
                pass

def to_markdown(data):
    print("FUNCTION: to_markdown")
    suppliers = [s for s in data.get("suppliers", []) if s.get("name")]
    names = [s["name"] for s in suppliers]
    headers = ["SN", "Altered Material Name", "Material description"] + [f"{n} Unit Price" for n in names] + ["Source file(s)"]
    rows = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for number, item in enumerate(data.get("items", []), 1):
        prices = {p.get("supplier"): p.get("value", "N/A") for p in item.get("prices", [])}
        values = [str(number), item.get("name", ""), item.get("description", "")]
        values += [prices.get(name, "N/A") or "N/A" for name in names]
        values.append("; ".join(item.get("source_files", [])))
        rows.append("| " + " | ".join(values) + " |")
    rows += ["", "| Supplier Name | Payment Terms | Quotation Validity |", "|---|---|---|"]
    rows += [f"| {s['name']} | {s.get('payment_terms', '')} | {s.get('quotation_validity', '')} |" for s in suppliers]
    return "\n".join(rows)
