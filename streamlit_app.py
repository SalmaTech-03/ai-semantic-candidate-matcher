# app.py

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

# --- Initialize Session State ---
# This is crucial for keeping data persistent across interactions (like filtering)
if 'results_df' not in st.session_state:
    st.session_state.results_df = None

# --- LOAD MODELS (Cached for Performance) ---
@st.cache_resource
def load_models():
    """Loads all necessary AI models into memory."""
    print("Loading AI models...")
    nlp = sp.load("en_core_web_sm")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        gemini_model = genai.GenerativeModel('gemini-1.0-pro')
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}. Ensure GOOGLE_API_KEY is set in secrets.")
        gemini_model = None
    print("Models loaded successfully.")
    return nlp, model, gemini_model

# --- BACKEND LOGIC (Helper Functions) ---

SKILLS_DB = [
    'python', 'java', 'c++', 'javascript', 'sql', 'git', 'docker', 'aws', 'azure', 'gcp',
    'react', 'angular', 'vue.js', 'node.js', 'flask', 'django', 'fastapi', 'html', 'css',
    'machine learning', 'deep learning', 'nlp', 'natural language processing', 'pandas',
    'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'api', 'rest', 'graphql',
    'communication', 'teamwork', 'problem-solving', 'agile', 'scrum', 'data analysis',
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

def analyze_resume(resume_text, jd_text, jd_skills_set, jd_embedding, sentence_model, gemini_model):
    resume_embedding = sentence_model.encode(resume_text)
    cosine_score = util.pytorch_cos_sim(resume_embedding, jd_embedding).item()
    match_percentage = round(cosine_score * 100, 2)
    
    resume_skills = set(extract_skills(resume_text))
    matching_skills = list(resume_skills & jd_skills_set)
    missing_skills = list(jd_skills_set - resume_skills)
    
    ai_summary = generate_ai_summary(resume_text, jd_text, gemini_model)
    
    return {
        "match_score": match_percentage,
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "ai_summary": ai_summary,
    }

def create_skill_gap_chart(candidate_data):
    """Generates a Plotly bar chart for a single candidate's skill gap."""
    matching_count = len(candidate_data['matching_skills'])
    missing_count = len(candidate_data['missing_skills'])
    
    fig = go.Figure(data=[
        go.Bar(name='Matching Skills', x=['Skills'], y=[matching_count], marker_color='#4CAF50', text=matching_count, textposition='auto'),
        go.Bar(name='Missing Skills', x=['Skills'], y=[missing_count], marker_color='#F44336', text=missing_count, textposition='auto')
    ])
    fig.update_layout(
        barmode='stack',
        title=f"Skill Gap Analysis for {candidate_data['filename']}",
        yaxis_title="Number of Skills",
        legend_title="Skill Type"
    )
    return fig

# --- MAIN APPLICATION UI ---
st.title("AI Semantic Candidate Matcher ✨")
st.write("An advanced tool to screen, rank, and analyze candidates. Upload resumes and a job description to begin.")

# Load models safely
try:
    nlp, sentence_model, gemini_model = load_models()
except Exception as e:
    st.error(f"A fatal error occurred during model loading: {e}")
    st.stop()

# --- SIDEBAR FOR INPUTS AND FILTERS ---
with st.sidebar:
    st.header("Inputs")
    uploaded_files = st.file_uploader("1. Upload Resumes", type=["pdf", "docx"], accept_multiple_files=True)
    job_description = st.text_area("2. Paste Job Description", height=200)

    if st.button("Rank Candidates", type="primary", use_container_width=True):
        if uploaded_files and job_description:
            with st.spinner(f"Analyzing {len(uploaded_files)} resumes..."):
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
                    # Store the full results in the session state
                    st.session_state.results_df = pd.DataFrame(results).sort_values(by='match_score', ascending=False)
                else:
                    st.session_state.results_df = None
                    st.warning("Could not process any of the uploaded resumes.")
        else:
            st.warning("Please upload resumes and a job description.")

    # --- Interactive Filtering (only appears after results are generated) ---
    if st.session_state.results_df is not None:
        st.divider()
        st.header("Interactive Filters")
        
        # Filter by Match Score
        score_threshold = st.slider("Minimum Match Score (%)", 0, 100, 50)
        
        # Apply the filter
        df_filtered = st.session_state.results_df[st.session_state.results_df['match_score'] >= score_threshold]
    else:
        df_filtered = pd.DataFrame() # Create an empty DataFrame if no results

# --- MAIN PAGE DISPLAY ---

if st.session_state.results_df is None:
    st.info("Upload resumes and a job description in the sidebar to get started.")
else:
    st.header("Ranked & Filtered Candidate Results")
    st.write(f"Showing {len(df_filtered)} of {len(st.session_state.results_df)} total candidates.")

    if not df_filtered.empty:
        # Create a selectable table for the "deep dive" feature
        df_display = df_filtered.copy()
        df_display['Select'] = False
        df_display['matching_skill_count'] = df_display['matching_skills'].apply(len)
        df_display['missing_skill_count'] = df_display['missing_skills'].apply(len)
        
        # Use the data_editor to make a clickable table
        edited_df = st.data_editor(
            df_display[['Select', 'filename', 'match_score', 'matching_skill_count', 'missing_skill_count']],
            use_container_width=True,
            column_config={
                'Select': st.column_config.CheckboxColumn("Select", width="small"),
                'filename': "Candidate Resume",
                'match_score': st.column_config.ProgressColumn("Match Score (%)", format="%.2f%%", min_value=0, max_value=100),
                'matching_skill_count': "Matching Skills",
                'missing_skill_count': "Missing Skills",
            },
            hide_index=True,
            key="candidate_selector"
        )
        
        # --- Candidate Deep Dive Display ---
        selected_row = edited_df[edited_df.Select]
        if not selected_row.empty:
            # Get the filename of the first selected candidate
            selected_filename = selected_row.iloc[0]['Candidate Resume']
            # Find the full data for that candidate from the original unfiltered DataFrame
            candidate_details = st.session_state.results_df[st.session_state.results_df['filename'] == selected_filename].iloc[0]
            
            st.divider()
            st.header(f"Deep Dive Analysis: {candidate_details['filename']}")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("📊 Visual Skill Gap")
                skill_chart = create_skill_gap_chart(candidate_details)
                st.plotly_chart(skill_chart, use_container_width=True)
            with col2:
                st.subheader("🤖 AI Summary")
                st.info(candidate_details['ai_summary'])
                st.markdown("**✅ Matching Skills:**")
                st.write(f"`{', '.join(candidate_details['matching_skills'])}`")
                st.markdown("**❌ Missing Skills:**")
                st.write(f"`{', '.join(candidate_details['missing_skills'])}`")
    else:
        st.warning("No candidates match the current filter criteria.")