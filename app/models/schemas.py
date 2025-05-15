from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class ClassificationRequest(BaseModel):
    userId: str
    referencePhotoId: str
    poolPhotoIds: List[str]

class MatchedPhoto(BaseModel):
    photoId: str
    imageUrl: str
    confidence: float

class UnmatchedPhoto(BaseModel):
    photoId: str
    imageUrl: str

class ClassificationResponse(BaseModel):
    taskId: str
    matched: List[MatchedPhoto]
    unmatched: List[UnmatchedPhoto]

class ClassificationResult(BaseModel):
    taskId: str
    userId: str
    referenceImage: str
    matches: List[dict]  # Each dict contains photoId, confidence, matchedAt
    unmatchedImages: List[dict]  # Each dict contains photoId, processedAt
    createdAt: datetime = Field(default_factory=datetime.utcnow)