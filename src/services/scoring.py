from src.core.schemas import Scorecard
from src.core.config import settings

class TalentScorer:
    @staticmethod
    def generate_scorecard(analysis: dict, semantic_score: float) -> Scorecard:
        # Experience score: Benchmarked against 5 years
        exp_score = min((analysis.get('years_exp', 0) / 5) * 100, 100)
        
        overall = (semantic_score * settings.WEIGHT_SEMANTIC) + (exp_score * settings.WEIGHT_EXPERIENCE)
        
        reasoning = (f"Candidate shows {analysis.get('years_exp')} years of experience. "
                     f"Semantic match to JD is {semantic_score:.1f}%.")
        
        return Scorecard(
            overall_score=round(overall, 2),
            semantic_match=round(semantic_score, 2),
            experience_alignment=round(exp_score, 2),
            reasoning=reasoning
        )