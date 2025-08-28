# streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import spacy
import fitz  # PyMuPDF
import docx
import re
from sentence_transformers import SentenceTransformer, util
import os
import google.generativeai as genai

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Semantic Candidate Matcher",
    page_icon="✨",
    layout="wide"
)

# --- LOAD MODELS (Cached for Performance) ---
@st.cache_resource
def load_models():
    """Loads all necessary AI models into memory."""
    print("Loading AI models...")
    nlp = spacy.load("en_core_web_sm")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}. Ensure GOOGLE_API_KEY is set in secrets.")
        gemini_model = None
    print("Models loaded successfully.")
    return nlp, model, gemini_model

# --- BACKEND LOGIC (Helper Functions) ---
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
    if not gemini_model: return "Gemini model not available."
    safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
    prompt = f"Analyze the resume in context of the job description. Provide a concise, 3-sentence professional summary. JOB DESCRIPTION: {jd_text} --- RESUME: {resume_text[:4000]} ---"
    try:
        response = gemini_model.generate_content(prompt, safety_settings=safety_settings)
        return response.text.strip()
    except Exception as e:
        return f"Could not generate summary: {e}"

def analyze_resume(resume_text, jd_text, jd_skills_set, jd_embedding, sentence_model, gemini_model):
    resume_embedding = sentence_model.encode(resume_text)
    cosine_score = util.pytorch_cos_sim(resume_embedding, jd_embedding).item()
    match_percentage = round(cosine_score * 100, 2)
    resume_skills = set(extract_skills(resume_text))
    matching_skills = list(resume_skills & jd_skills_set)
    missing_skills = list(jd_skills_set - resume_skills)
    ai_summary = generate_ai_summary(resume_text, jd_text, gemini_model)
    return {"match_score": match_percentage, "matching_skills": matching_skills, "missing_skills": missing_skills, "ai_summary": ai_summary}

def create_skill_gap_chart(candidate_data):
    matching_count = len(candidate_data['matching_skills'])
    missing_count = len(candidate_data['missing_skills'])
    fig = go.Figure(data=[
        go.Bar(name='Matching Skills', x=['Skills'], y=[matching_count], marker_color='#4CAF50', text=matching_count, textposition='auto'),
        go.Bar(name='Missing Skills', x=['Skills'], y=[missing_count], marker_color='#F44336', text=missing_count, textposition='auto')
    ])
    fig.update_layout(barmode='stack', title="Skill Gap Overview", yaxis_title="Number of Skills", legend_title="Skill Type", margin=dict(t=30, b=0, l=0, r=0))
    return fig

# --- UI State Initialization ---
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'selected_candidate' not in st.session_state:
    st.session_state.selected_candidate = None

# --- MAIN APP LAYOUT ---
st.title("AI Semantic Candidate Matcher ✨")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Inputs")
    uploaded_files = st.file_uploader("1. Upload Resumes", type=["pdf", "docx"], accept_multiple_files=True)
    job_description = st.text_area("2. Paste Job Description", height=200)

    if st.button("Rank Candidates", type="primary", use_container_width=True):
        if uploaded_files and job_description:
            with st.spinner(f"Analyzing {len(uploaded_files)} resumes..."):
                nlp, sentence_model, gemini_model = load_models()
                jd_skills_set = set(extract_skills(job_description))
                jd_embedding = sentence_model.encode(job_description)
                results = []
                for file in uploaded_files:
                    resume_text = extract_text_from_file(file)
                    if resume_text:
                        analysis = analyze_resume(resume_text, job_description, jd_skills_set, jd_embedding, sentence_model, gemini_model)
                        analysis['filename'] = file.name
                        results.append(analysis)
                
                if results:
                    st.session_state.results_df = pd.DataFrame(results).sort_values(by='match_score', ascending=False)
                    # Automatically select the top candidate to view first
                    st.session_state.selected_candidate = st.session_state.results_df.iloc[0]['filename']
                else:
                    st.session_state.results_df = None
                    st.warning("Could not process any of the uploaded resumes.")
        else:
            st.warning("Please upload resumes and a job description.")

# --- Main Page Display ---
if st.session_state.results_df is None:
    st.info("Upload resumes and a job description in the sidebar to get started.")
else:
    # Create a two-column layout for the "Master-Detail" view
    master_col, detail_col = st.columns([1, 2], gap="large")

    with master_col:
        st.header("Ranked Candidates")
        st.write(f"Click on a resume to view its detailed analysis.")
        
        # --- NEW: Display each candidate as a selectable button ---
        for index, row in st.session_state.results_df.iterrows():
            # When a button is clicked, it updates the selected_candidate in the session state
            if st.button(f"📄 {row['filename']} ({row['match_score']:.2f}%)", key=row['filename'], use_container_width=True):
                st.session_state.selected_candidate = row['filename']

    with detail_col:
        if st.session_state.selected_candidate:
            # Get the full data for the currently selected candidate
            candidate_details = st.session_state.results_df[st.session_state.results_df['filename'] == st.session_state.selected_candidate].iloc[0]
            
            st.header(f"Deep Dive Analysis: {candidate_details['filename']}")
            
            st.subheader("🤖 AI Summary")
            st.info(candidate_details['ai_summary'])
            st.divider()

            chart_col, skills_col = st.columns(2)
            with chart_col:
                st.subheader("📊 Visual Skill Gap")
                skill_chart = create_skill_gap_chart(candidate_details)
                st.plotly_chart(skill_chart, use_container_width=True)
            with skills_col:
                st.subheader("✅ Matching Skills")
                st.write(f"`{', '.join(candidate_details['matching_skills'])}`")
                st.subheader("❌ Missing Skills")
                st.write(f"`{', '.join(candidate_details['missing_skills'])}`")
        else:
            # This message shows if for some reason no candidate is selected
            st.info("Select a candidate from the list on the left to see their detailed analysis.")