# backend/app.py

import os
import re
import spacy
import fitz  # PyMuPDF
import docx
import phonenumbers
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

# --- Setup & Configuration ---
load_dotenv()

app = Flask(__name__)

# --- Model Loading ---
try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model loaded successfully.")
except OSError:
    nlp = None
    print("spaCy model not found. Some parsing features may be limited.")

try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Sentence Transformer model loaded successfully.")
except Exception as e:
    model = None
    print(f"Error loading Sentence Transformer model: {e}")

try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set in the .env file.")
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    print("Gemini model configured successfully.")
except Exception as e:
    gemini_model = None
    print(f"Error configuring Gemini: {e}")

# --- Constants ---
SKILLS_DB = [
    'python', 'java', 'c++', 'javascript', 'sql', 'git', 'docker', 'aws', 'azure', 'gcp',
    'react', 'angular', 'vue.js', 'node.js', 'flask', 'django', 'fastapi', 'html', 'css',
    'machine learning', 'deep learning', 'nlp', 'natural language processing', 'pandas',
    'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'api', 'rest', 'graphql',
    'communication', 'teamwork', 'problem-solving', 'agile', 'scrum', 'data analysis',
    'project management', 'product management'
]

# --- Core Logic (Functions remain the same) ---
def extract_text(file_stream, filename):
    text = ""
    if filename.endswith(".pdf"):
        with fitz.open(stream=file_stream.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    elif filename.endswith(".docx"):
        doc = docx.Document(file_stream)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

def extract_contact_info(text):
    name, email, phone = None, None, None
    if nlp:
        doc = nlp(text[:300])
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                name = ent.text
                break
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    if email_match:
        email = email_match.group(0)
    try:
        for match in phonenumbers.PhoneNumberMatcher(text, "US"):
            phone = phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.E164)
            break
    except Exception:
        pass
    return name, email, phone

def extract_skills(text):
    found_skills = set()
    text_lower = text.lower()
    for skill in SKILLS_DB:
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found_skills.add(skill)
    return list(found_skills)

def extract_education(text):
    education = []
    education_pattern = r'(B\.S\.?|M\.S\.?|Ph\.D\.?|Bachelor|Master|PhD)\s(of|in)\s([\w\s]+)'
    matches = re.findall(education_pattern, text, re.IGNORECASE)
    for match in matches:
        education.append(f"{match[0]} in {match[2]}")
    return education if education else ["Not Found"]

def parse_resume(text):
    name, email, phone = extract_contact_info(text)
    skills = extract_skills(text)
    education = extract_education(text)
    return {
        "name": name if name else "Not Found",
        "email": email if email else "Not Found",
        "phone": phone if phone else "Not Found",
        "skills": skills,
        "education": education
    }

def generate_ai_analysis(resume_text, jd_text):
    if not gemini_model:
        return {"error": "AI model not configured."}
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    prompt = f"""
    Analyze the following resume in the context of the provided job description.
    Provide a concise, 3-sentence professional summary for the candidate.
    ---
    JOB DESCRIPTION:
    {jd_text}
    ---
    RESUME:
    {resume_text[:4000]}
    ---
    """
    try:
        response = gemini_model.generate_content(prompt, safety_settings=safety_settings)
        return {"summary": response.text.strip()}
    except Exception as e:
        print(f"Gemini API Error or Parsing Error: {e}")
        return {"summary": f"Could not generate summary: {e}"}

# --- Reusable Analysis Function ---
def analyze_single_resume(resume_text, jd_text, jd_skills, jd_embedding):
    """Analyzes a single resume against pre-processed JD data."""
    # 1. Parse Resume
    parsed_resume = parse_resume(resume_text)
    
    # 2. Semantic Score
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(resume_embedding, jd_embedding).item()
    match_percentage = round(cosine_score * 100, 2)
    
    # 3. Skill Gap
    matching_skills = list(set(parsed_resume['skills']) & set(jd_skills))
    missing_skills = list(set(jd_skills) - set(parsed_resume['skills']))
    
    # 4. AI Summary (shorter for the table view)
    ai_summary = generate_ai_analysis(resume_text, jd_text)

    return {
        "match_score": match_percentage,
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "ai_summary": ai_summary.get("summary"),
        "contact": {
            "name": parsed_resume.get("name"),
            "email": parsed_resume.get("email"),
            "phone": parsed_resume.get("phone")
        }
    }


# --- NEW BATCH PROCESSING ENDPOINT ---
@app.route("/api/batch-analyze", methods=['POST'])
def batch_analyze_resumes():
    if 'job_description' not in request.form:
        return jsonify({"error": "Missing job description"}), 400
    
    resumes = request.files.getlist('resumes')
    if not resumes:
        return jsonify({"error": "No resume files provided"}), 400

    jd_text = request.form['job_description']
    
    # Pre-process the job description once to save time
    jd_skills = set(extract_skills(jd_text))
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    
    results = []
    
    for resume_file in resumes:
        filename = resume_file.filename
        print(f"Processing: {filename}")
        resume_text = extract_text(resume_file.stream, filename)
        
        if not resume_text:
            results.append({"filename": filename, "error": "Could not extract text"})
            continue
            
        analysis = analyze_single_resume(resume_text, jd_text, jd_skills, jd_embedding)
        analysis['filename'] = filename
        results.append(analysis)
        
    
    return jsonify({
    "candidates": results,
    "jd_skills": list(jd_skills)
})

if __name__ == "__main__":
    # Get port from environment variable, default to 5001 if not set
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=False, host="0.0.0.0", port=port)