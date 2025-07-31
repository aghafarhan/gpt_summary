from pydantic import BaseModel
from typing import List

class Item(BaseModel):
    item_name: str

class ProcurementRequest(BaseModel):
    items: List[Item]
