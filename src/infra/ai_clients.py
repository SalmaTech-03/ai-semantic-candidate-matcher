def analyze_resume(self, text: str) -> dict:
        scrubbed = scrub_pii(text)
        # We add "Return JSON" to the prompt and handle it manually
        prompt = f"Analyze resume and return ONLY a flat JSON object with keys: years_exp(int), education(str), skills(list), summary(str). Resume text: {scrubbed[:7000]}"
        
        try:
            # First attempt: standard request
            response = self.model.generate_content(prompt)
            
            # Extract the text and clean it
            res_text = response.text
            # Remove markdown code blocks if the AI added them
            clean_json = res_text.replace("```json", "").replace("```", "").strip()
            
            return json.loads(clean_json)
        except Exception as e:
            from src.utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"AI Extraction failed: {e}")
            # Fallback data so the app doesn't crash
            return {
                "years_exp": 0, 
                "education": "Not Found", 
                "skills": [], 
                "summary": "AI extraction failed for this document."
            }