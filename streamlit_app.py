# streamlit_app.py

import streamlit as st
import pandas as pd
import spacy
import fitz  # PyMuPDF
import docx
import re
import os
import google.generativeai as genai

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Semantic Candidate Matcher",
    page_icon="✨",
    layout="wide"
)

# --- LOAD MODELS (Lighter Version) ---
@st.cache_resource
def load_models():
    """Loads all necessary AI models into memory."""
    print("Loading AI models...")
    nlp = spacy.load("en_core_web_sm")
    # This is the new, lighter sentence encoder
    nlp.add_pipe("universal_sentence_encoder")
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}. Ensure GOOGLE_API_KEY is set in secrets.")
        gemini_model = None
    print("Models loaded successfully.")
    return nlp, gemini_model

# --- BACKEND LOGIC ---
SKILLS_DB = [
    'python', 'java', 'c++', 'javascript', 'sql', 'git', 'docker', 'aws', 'azure', 'gcp', 'react', 'angular',
    'vue.js', 'node.js', 'flask', 'django', 'fastapi', 'html', 'css', 'machine learning', 'deep learning',
    'nlp', 'natural language processing', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch',
    'api', 'rest', 'graphql', 'communication', 'teamwork', 'problem-solving', 'agile', 'scrum', 'data analysis',
    'project management', 'product management'
]

def extract_text_from_file(file_uploader_object):
    try:
        file_extension = os.path.splitext(file_uploader_object.name)[1].lower()
        if file_extension == ".pdf":
            with fitz.open(stream=file_uploader_object.getvalue(), filetype="pdf") as doc:
                return "".join(page.get_text() for page in doc)
        elif file_extension == ".docx":
            doc = docx.Document(file_uploader_object)
            return "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        st.error(f"Error reading file {file_uploader_object.name}: {e}")
    return ""

def extract_skills(text):
    found_skills = set()
    text_lower = text.lower()
    for skill in SKILLS_DB:
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found_skills.add(skill)
    return list(found_skills)

def generate_ai_summary(resume_text, jd_text, gemini_model):
    if not gemini_model:
        return "Gemini model not available."
    safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
    prompt = f"Analyze the resume in context of the job description. Provide a concise, 3-sentence professional summary. JOB DESCRIPTION: {jd_text} --- RESUME: {resume_text[:4000]} ---"
    try:
        response = gemini_model.generate_content(prompt, safety_settings=safety_settings)
        return response.text.strip()
    except Exception as e:
        return f"Could not generate summary: {e}"

def analyze_resume(resume_text, jd_doc, jd_skills_set, nlp, gemini_model):
    resume_doc = nlp(resume_text)
    # Use the spaCy similarity method
    cosine_score = resume_doc.similarity(jd_doc)
    match_percentage = round(cosine_score * 100, 2)
    
    resume_skills = set(extract_skills(resume_text))
    matching_skills = list(resume_skills & jd_skills_set)
    missing_skills = list(jd_skills_set - resume_skills)
    
    ai_summary = generate_ai_summary(resume_text, jd_doc.text, gemini_model)
    
    return {
        "match_score": match_percentage,
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "ai_summary": ai_summary,
    }

# --- MAIN UI ---
st.title("AI Semantic Candidate Matcher ✨")
st.write("A free, efficient tool to rank candidates. Upload resumes and a job description to begin.")

try:
    nlp, gemini_model = load_models()
except Exception as e:
    st.error(f"Fatal error during model loading: {e}")
    st.stop()

col1, col2 = st.columns([1, 2])
with col1:
    st.header("1. Upload Resumes")
    uploaded_files = st.file_uploader("Select one or more resumes", type=["pdf", "docx"], accept_multiple_files=True)
with col2:
    st.header("2. Paste Job Description")
    job_description = st.text_area("Paste the full job description here", height=250)

if st.button("Rank Candidates", type="primary", use_container_width=True):
    if uploaded_files and job_description:
        with st.spinner(f"Analyzing {len(uploaded_files)} resumes..."):
            # Pre-process JD once
            jd_skills_set = set(extract_skills(job_description))
            jd_doc = nlp(job_description) # Process with NLP model once
            
            results = []
            for file in uploaded_files:
                resume_text = extract_text_from_file(file)
                if resume_text:
                    analysis = analyze_resume(resume_text, jd_doc, jd_skills_set, nlp, gemini_model)
                    analysis['filename'] = file.name
                    results.append(analysis)
            
            if results:
                st.header("Ranked Candidate Results")
                df = pd.DataFrame(results).sort_values(by='match_score', ascending=False)
                df['matching_skill_count'] = df['matching_skills'].apply(len)
                df['missing_skill_count'] = df['missing_skills'].apply(len)
                
                st.dataframe(
                    df[['filename', 'match_score', 'matching_skill_count', 'missing_skill_count', 'ai_summary']],
                    use_container_width=True,
                    column_config={
                        'filename': st.column_config.TextColumn("Candidate Resume", width="medium"),
                        'match_score': st.column_config.ProgressColumn("Match Score (%)", format="%.2f%%", min_value=0, max_value=100),
                        'matching_skill_count': "Matching Skills",
                        'missing_skill_count': "Missing Skills",
                        'ai_summary': st.column_config.TextColumn("AI Summary", width="large")
                    },
                    hide_index=True
                )
    else:
        st.warning("Please upload at least one resume and a job description.")