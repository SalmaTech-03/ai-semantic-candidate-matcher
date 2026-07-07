from fastapi import FastAPI, UploadFile, File, Form
from src.services.orchestrator import TalentOrchestrator
from typing import List

app = FastAPI()
orchestrator = TalentOrchestrator()

@app.post("/analyze")
async def analyze_resumes(jd: str = Form(...), files: List[UploadFile] = File(...)):
    jd_vector = orchestrator.ai.get_embeddings(jd)
    results = []
    
    for file in files:
        content = await file.read()
        res = await orchestrator.process_candidate(content, file.filename, jd_vector)
        if res: results.append(res)
        
    return sorted(results, key=lambda x: x['score'], reverse=True)