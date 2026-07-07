import numpy as np
from src.infra.ai_clients import AIClient
from src.infra.parser import SecureParser
from src.services.scoring import TalentScorer

class TalentOrchestrator:
    def __init__(self):
        self.ai = AIClient()
        self.parser = SecureParser()
        self.scorer = TalentScorer()

    async def process_candidate(self, file_bytes: bytes, filename: str, jd_vector: np.ndarray):
        text = self.parser.to_text(file_bytes)
        if not text: return None
        
        analysis = self.ai.analyze_resume(text)
        res_vector = self.ai.get_embeddings(text)
        
        # Calculate Cosine Similarity
        sim = np.dot(jd_vector, res_vector) / (np.linalg.norm(jd_vector) * np.linalg.norm(res_vector))
        
        scorecard = self.scorer.generate_scorecard(analysis, float(sim * 100))
        
        return {
            "name": filename,
            "score": scorecard.overall_score,
            "details": scorecard.dict(),
            "summary": analysis.get('summary', '')
        }