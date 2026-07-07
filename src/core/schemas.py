from pydantic import BaseModel, Field
from typing import List, Dict

class CandidateAnalysis(BaseModel):
    years_exp: int = Field(..., ge=0)
    education: str
    skills: List[str]
    summary: str

class Scorecard(BaseModel):
    overall_score: float
    semantic_match: float
    experience_alignment: float
    reasoning: str