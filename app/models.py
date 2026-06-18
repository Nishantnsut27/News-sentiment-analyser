from sqlmodel import SQLModel, Field
from datetime import datetime
from typing import Optional

class SentimentFeedback(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    headline: str
    predicted_label: int
    perceived_label: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
