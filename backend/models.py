from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np

class TextQueryRequest(BaseModel):
    query: str

class ProductResponse(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    price: Optional[float] = None
    image_url: Optional[str] = None
    similarity: float
    attributes: Dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True

class SearchResponse(BaseModel):
    products: List[ProductResponse]
    enhanced_response: str