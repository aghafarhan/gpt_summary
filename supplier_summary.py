import openpyxl
from openpyxl.styles import Alignment
import os
import re
from urllib.parse import urlparse
import httpx
from typing import List

from llm_client import get_chat_model, get_llm_client, unwrap_llm_text


def verify_url(url: str) -> bool:
    """Verify that a candidate website is reachable before exporting it."""
    try:
        with httpx.Client(timeout=8.0, follow_redirects=True) as http:
            response = http.get(url, headers={"User-Agent": "ProcurementResearch/1.0"})
            return response.status_code < 400 and response.url.scheme in {"http", "https"}
    except httpx.HTTPError:
        return False


def generate_supplier_summary_excel(items: List[dict], output_file: str):
    client = get_llm_client()

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Suppliers"

    headers = ["S.NO", "SUPPLIER NAME", "PRICE ESTIMATE", "PACKAGING / MOQ", "WEBSITE LINK"]
    current_row = 2

    for idx, entry in enumerate(items, 1):
        item = entry["item_name"]

        prompt = f"""
You are a procurement research assistant with access to live web search. Search the web before
answering. Never invent a supplier, URL, price, MOQ, location, or product. Use `Quote Required`
or `Not Found` when the web evidence does not contain the requested information.

Your job is to research and identify reliable suppliers for the requested item. The business may operate in construction, hospitality, restaurants, retail, or manufacturing.

### REQUESTED ITEM:
- Item: {item}
- Primary market: Saudi Arabia

### TASK:
1. For this specific material, determine the most relevant supplier types:
   - Use manufacturers, distributors, wholesalers, authorized dealers, food-service suppliers, or equipment specialists as appropriate.
   - Avoid generic trading companies unless no more relevant supplier is available.
2. Select suppliers only from the verified search results above.
3. Return **exactly 10 Saudi-based suppliers** who offer this item or a very close variant.
4. Additionally, include **1 supplier from China** that ships internationally to Saudi Arabia.
5. For each supplier, collect:
   - Supplier name
   - Price estimate (or “Quote Required”)
   - Packaging / MOQ
   - Website link — must be a **markdown clickable link ONLY** like [Website](https://example.com)

⚠️ INSTRUCTIONS:
- Return **exactly 11 rows** per item (10 Saudi + 1 China).
- Use your judgment to select the best supplier category for the given material.
- Website link must be the supplier’s own website or direct product page (not directory or aggregator listings, unless absolutely no other link exists).
- No Amazon.
- Prefer **specialized suppliers**, **not generic shopping sites**.

### FORMAT:
Return as a markdown table with:

| # | Supplier | Price Estimate | Packaging / MOQ | Website Link |
"""

        response = client.responses.create(
            model=get_chat_model("gpt-5.6"),
            tools=[{"type": "web_search"}],
            input=prompt,
        )
        markdown = unwrap_llm_text(response.output_text)

        print(markdown)

        # Section heading
        ws.merge_cells(start_row=current_row, start_column=2, end_row=current_row, end_column=6)
        ws.cell(row=current_row, column=2).value = f"MATERIAL NAME {idx}: {item}"
        ws.cell(row=current_row, column=2).alignment = Alignment(horizontal="center")
        current_row += 1

        # Headers
        for col, header in enumerate(headers, start=2):
            ws.cell(row=current_row, column=col).value = header
        current_row += 1

        # Parse table rows
        rows = [r.strip("|").split("|") for r in markdown.split("\n") if "|" in r]
        rows = [r for r in rows if not all(set(v.strip()) <= {"-", ":"} for v in r)]
        rows = rows[1:]

        for i, row in enumerate(rows):
            if len(row) < len(headers) - 1:
                print(f"⚠️ Skipping row {i+1}: incomplete -> {row}")
                continue

            ws.cell(row=current_row, column=2).value = i + 1  # S.NO

            for j, val in enumerate(row):
                cell_val = val.strip()
                col = 2 + j
                if col - 2 >= len(headers):
                    continue

                cell = ws.cell(row=current_row, column=col)

                if headers[col - 2] == "WEBSITE LINK":
                    match = re.search(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", cell_val)
                    if match:
                        display_text = match.group(1).strip()
                        url = match.group(2).strip()
                        if verify_url(url):
                            cell.value = display_text
                            cell.hyperlink = url
                            cell.style = "Hyperlink"
                        else:
                            cell.value = "Not Found"
                    else:
                        match = re.search(r"(https?://[^\\s)]+)", cell_val)
                        if match:
                            url = match.group(1).strip()
                            if verify_url(url):
                                cell.value = url
                                cell.hyperlink = url
                                cell.style = "Hyperlink"
                            else:
                                cell.value = "Not Found"
                        else:
                            cell.value = cell_val
                else:
                    cell.value = cell_val

            current_row += 1

        current_row += 2

    wb.save(output_file)
    print(f"✅ Excel saved as: {output_file}")
