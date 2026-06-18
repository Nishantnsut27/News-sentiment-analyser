from pydantic import BaseModel
from typing import Optional

class AnalyzeRequest(BaseModel):
    url: Optional[str] = None
    headline: Optional[str] = None

class AnalyzeResponse(BaseModel):
    sentiment: str
    headline: str

class FeedbackRequest(BaseModel):
    userPerception: str
    headline: str

class FeedbackResponse(BaseModel):
    status: str
