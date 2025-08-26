# streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import spacy
import fitz  # PyMuPDF
import docx
import phonenumbers
import re
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Semantic Candidate Matcher",
    page_icon="✨",
    layout="wide"
)

# --- Load Models ---
@st.cache_resource
def load_models():
    nlp_model = spacy.load("en_core_web_sm")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Configure Gemini AI
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception:
        gemini_model = None

    return nlp_model, sentence_model, gemini_model

nlp, sentence_model, gemini_model = load_models()

# --- Skills Database ---
SKILLS_DB = [
    'python', 'java', 'c++', 'javascript', 'sql', 'git', 'docker', 'aws', 'azure', 'gcp',
    'react', 'angular', 'vue.js', 'node.js', 'flask', 'django', 'fastapi', 'html', 'css',
    'machine learning', 'deep learning', 'nlp', 'natural language processing', 'pandas',
    'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'api', 'rest', 'graphql',
    'communication', 'teamwork', 'problem-solving', 'agile', 'scrum', 'data analysis',
    'project management', 'product management'
]

# --- Resume Parsing Functions ---
def extract_text(file):
    text = ""
    ext = file.name.split(".")[-1].lower()
    if ext == "pdf":
        with fitz.open(stream=file.getvalue(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    elif ext == "docx":
        doc = docx.Document(file)
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
    pattern = r'(B\.S\.?|M\.S\.?|Ph\.D\.?|Bachelor|Master|PhD)\s(of|in)\s([\w\s]+)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    for match in matches:
        education.append(f"{match[0]} in {match[2]}")
    return education if education else ["Not Found"]

def parse_resume(text):
    name, email, phone = extract_contact_info(text)
    skills = extract_skills(text)
    education = extract_education(text)
    return {
        "name": name or "Not Found",
        "email": email or "Not Found",
        "phone": phone or "Not Found",
        "skills": skills,
        "education": education
    }

# --- AI Summary ---
def generate_ai_summary(resume_text, jd_text):
    if not gemini_model:
        return "Gemini AI not available"
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    prompt = f"""
    Analyze this resume in context of the job description. Provide a concise 3-sentence professional summary.

    JOB DESCRIPTION:
    {jd_text}

    RESUME:
    {resume_text[:4000]}
    """
    try:
        response = gemini_model.generate_content(prompt, safety_settings=safety_settings)
        return response.text.strip()
    except Exception as e:
        return f"Could not generate summary: {e}"

# --- Candidate Analysis ---
def analyze_resume(resume_text, jd_text, jd_skills, jd_embedding):
    # Semantic similarity
    resume_embedding = sentence_model.encode(resume_text, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(resume_embedding, jd_embedding).item()
    match_percentage = round(cosine_score * 100, 2)
    
    # Skills matching
    resume_skills = extract_skills(resume_text)
    matching_skills = list(set(resume_skills) & jd_skills)
    missing_skills = list(jd_skills - set(resume_skills))
    
    # AI summary
    ai_summary = generate_ai_summary(resume_text, jd_text)
    
    return {
        "match_score": match_percentage,
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "ai_summary": ai_summary
    }

# --- Streamlit UI ---
st.title("AI Semantic Candidate Matcher ✨")
st.write("Upload resumes and provide a job description to rank candidates.")

# Upload resumes and job description
col1, col2 = st.columns([1, 2])
with col1:
    uploaded_files = st.file_uploader("Upload Resumes (PDF/DOCX)", type=["pdf","docx"], accept_multiple_files=True)
with col2:
    job_description = st.text_area("Paste Job Description", height=250)

if st.button("Rank Candidates"):
    if uploaded_files and job_description:
        with st.spinner("Analyzing resumes..."):
            jd_skills = set(extract_skills(job_description))
            jd_embedding = sentence_model.encode(job_description, convert_to_tensor=True)
            
            results = []
            for file in uploaded_files:
                text = extract_text(file)
                parsed_resume = parse_resume(text)
                analysis = analyze_resume(text, job_description, jd_skills, jd_embedding)
                analysis.update(parsed_resume)
                analysis["filename"] = file.name
                results.append(analysis)
            
            # Display results
            df = pd.DataFrame(results).sort_values(by='match_score', ascending=False)
            df['matching_skill_count'] = df['matching_skills'].apply(len)
            df['missing_skill_count'] = df['missing_skills'].apply(len)
            
            st.header("Ranked Candidate Results")
            st.dataframe(df[['filename','name','email','phone','match_score','matching_skill_count','missing_skill_count','ai_summary']], use_container_width=True)
            
            # Download CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results CSV", csv, "candidate_results.csv", "text/csv")
    else:
        st.warning("Please upload resumes and provide a job description.")
