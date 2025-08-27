# app.py

import streamlit as st
import pandas as pd
import spacy
import fitz  # PyMuPDF
import docx
import re
from sentence_transformers import SentenceTransformer, util
import os
import google.generativeai as genai

# --- PAGE CONFIGURATION (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="AI Semantic Candidate Matcher",
    page_icon="✨",
    layout="wide"
)

# --- LOAD MODELS (Cached for Performance) ---
# This decorator ensures models are loaded only once
@st.cache_resource
def load_models():
    """Loads all necessary AI models into memory."""
    print("Loading AI models...")
    # Load NLP model for skill extraction
    nlp = spacy.load("en_core_web_sm")
    # Load a high-performance sentence transformer model for semantic search
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Configure the Gemini API using Streamlit's built-in secrets management
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        gemini_model = genai.GenerativeModel('gemini-1.0-pro')
    except Exception as e:
        # Display a user-friendly error if the API key is not set
        st.error(f"Error configuring Gemini API: {e}. Please ensure you have a GOOGLE_API_KEY secret set in your Hugging Face Space settings.")
        gemini_model = None
    print("Models loaded successfully.")
    return nlp, model, gemini_model

# --- BACKEND LOGIC (Helper Functions) ---

# A database of skills to search for
SKILLS_DB = [
    'python', 'java', 'c++', 'javascript', 'sql', 'git', 'docker', 'aws', 'azure', 'gcp',
    'react', 'angular', 'vue.js', 'node.js', 'flask', 'django', 'fastapi', 'html', 'css',
    'machine learning', 'deep learning', 'nlp', 'natural language processing', 'pandas',
    'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'api', 'rest', 'graphql',
    'communication', 'teamwork', 'problem-solving', 'agile', 'scrum', 'data analysis',
    'project management', 'product management'
]

def extract_text_from_file(file_uploader_object):
    """Extracts text from a file uploaded via Streamlit's file_uploader."""
    try:
        # Get the file extension from the uploaded file's name
        file_extension = os.path.splitext(file_uploader_object.name)[1].lower()
        if file_extension == ".pdf":
            # Use PyMuPDF to open the PDF from the file's bytes
            with fitz.open(stream=file_uploader_object.getvalue(), filetype="pdf") as doc:
                return "".join(page.get_text() for page in doc)
        elif file_extension == ".docx":
            # Use python-docx to open the DOCX from the file-like object
            doc = docx.Document(file_uploader_object)
            return "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        # Display an error to the user if a file can't be read
        st.error(f"Error reading file {file_uploader_object.name}: {e}")
        return ""
    return ""

def extract_skills(text):
    """Extracts a list of skills from a given text using spaCy and regex."""
    found_skills = set()
    text_lower = text.lower()
    for skill in SKILLS_DB:
        # Use word boundaries to ensure we match whole words only
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found_skills.add(skill)
    return list(found_skills)

def generate_ai_summary(resume_text, jd_text, gemini_model):
    """Generates a brief summary of the candidate using the Gemini API."""
    if not gemini_model:
        return "Gemini model is not available or configured correctly."
        
    # Safety settings to ensure the model generates appropriate content
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    prompt = f"""Analyze the following resume in the context of the provided job description. Provide a concise, 3-sentence professional summary for the candidate, highlighting their key strengths and alignment with the role.
    JOB DESCRIPTION:
    ---
    {jd_text}
    ---
    RESUME:
    {resume_text[:4000]} 
    ---
    """
    try:
        response = gemini_model.generate_content(prompt, safety_settings=safety_settings)
        return response.text.strip()
    except Exception as e:
        # Return a user-friendly error message if the API call fails
        return f"Could not generate summary due to an API error: {e}"

def analyze_resume(resume_text, jd_text, jd_skills_set, jd_embedding, sentence_model, gemini_model):
    """Performs a full analysis of a single resume against the job description."""
    # 1. Calculate Semantic Similarity Score
    resume_embedding = sentence_model.encode(resume_text)
    cosine_score = util.pytorch_cos_sim(resume_embedding, jd_embedding).item()
    match_percentage = round(cosine_score * 100, 2)
    
    # 2. Perform Skill Gap Analysis
    resume_skills = set(extract_skills(resume_text))
    matching_skills = list(resume_skills & jd_skills_set)
    missing_skills = list(jd_skills_set - resume_skills)
    
    # 3. Generate AI-Powered Summary
    ai_summary = generate_ai_summary(resume_text, jd_text, gemini_model)
    
    return {
        "match_score": match_percentage,
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "ai_summary": ai_summary,
    }

# --- MAIN APPLICATION UI ---
st.title("AI Semantic Candidate Matcher ✨")
st.write("A high-performance tool to rank candidates. Upload resumes and a job description to begin.")

# Load models and handle potential errors during startup
try:
    nlp, sentence_model, gemini_model = load_models()
except Exception as e:
    st.error(f"A fatal error occurred during model loading: {e}")
    st.stop() # Stop the app if essential models can't be loaded

# --- Input Area ---
col1, col2 = st.columns([1, 2])
with col1:
    st.header("1. Upload Resumes")
    uploaded_files = st.file_uploader("Select one or more resumes", type=["pdf", "docx"], accept_multiple_files=True)
with col2:
    st.header("2. Paste Job Description")
    job_description = st.text_area("Paste the full job description here", height=250)

# --- Analysis Trigger ---
if st.button("Rank Candidates", type="primary", use_container_width=True):
    if uploaded_files and job_description:
        with st.spinner(f"Analyzing {len(uploaded_files)} resumes... This can take a moment."):
            
            # Pre-process the job description once for efficiency
            jd_skills_set = set(extract_skills(job_description))
            jd_embedding = sentence_model.encode(job_description)
            
            # Process each resume
            results = []
            for file in uploaded_files:
                resume_text = extract_text_from_file(file)
                if resume_text:
                    analysis = analyze_resume(resume_text, job_description, jd_skills_set, jd_embedding, sentence_model, gemini_model)
                    analysis['filename'] = file.name
                    results.append(analysis)
            
            # --- Display Results ---
            if results:
                st.header("Ranked Candidate Results")
                # Convert results to a DataFrame for easy sorting and display
                df = pd.DataFrame(results).sort_values(by='match_score', ascending=False)
                
                # Create columns for skill counts for a cleaner table view
                df['matching_skill_count'] = df['matching_skills'].apply(len)
                df['missing_skill_count'] = df['missing_skills'].apply(len)
                
                # Display the results in a styled, interactive table
                st.dataframe(
                    df[['filename', 'match_score', 'matching_skill_count', 'missing_skill_count', 'ai_summary']],
                    use_container_width=True,
                    column_config={
                        'filename': st.column_config.TextColumn("Candidate Resume", width="medium"),
                        'match_score': st.column_config.ProgressColumn(
                            "Match Score (%)", format="%.2f%%", min_value=0, max_value=100
                        ),
                        'matching_skill_count': "Matching Skills",
                        'missing_skill_count': "Missing Skills",
                        'ai_summary': st.column_config.TextColumn("AI Summary", width="large")
                    },
                    hide_index=True
                )
    else:
        st.warning("Please upload at least one resume and provide a job description.")