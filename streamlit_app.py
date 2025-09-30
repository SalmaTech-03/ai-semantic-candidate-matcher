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
import json
import numpy as np
import faiss  # For FAANG-level RAG
import google.generativeai as genai

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Enterprise AI Talent Platform",
    page_icon="✨",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
.skill-badge-container { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 5px; margin-bottom: 15px; }
.skill-badge { display: inline-block; padding: 6px 12px; border-radius: 16px; font-size: 14px; font-weight: 500; line-height: 1; }
.matching-skill { background-color: rgba(4, 170, 109, 0.2); color: #04AA6D; border: 1px solid #04AA6D; }
.missing-skill { background-color: rgba(255, 75, 75, 0.2); color: #FF4B4B; border: 1px solid #FF4B4B; }
.st-info { border: 1px solid #4f8bf9; }
.quote { font-style: italic; border-left: 4px solid #4f8bf9; padding-left: 15px; margin-top: 10px; color: #DCDCDC; }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODELS (Cached for Performance) ---
@st.cache_resource
def load_models():
    """Loads all necessary AI models into memory."""
    print("Loading AI models...")
    nlp = spacy.load("en_core_web_sm")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}. Ensure GOOGLE_API_KEY is set in secrets.")
        gemini_model = None
    print("Models loaded successfully.")
    return nlp, model, gemini_model

# --- CORE LOGIC (Helper Functions) ---
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
            with fitz.open(stream=file_uploader_object.getvalue(), filetype="pdf") as doc: return "".join(page.get_text() for page in doc)
        elif file_extension == ".docx":
            doc = docx.Document(file_uploader_object)
            return "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        st.error(f"Error reading file {file_uploader_object.name}: {e}")
    return ""

def extract_skills(text, skills_db):
    found_skills = set()
    text_lower = text.lower()
    for skill in skills_db:
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
    except Exception as e: return f"Could not generate summary: {e}"

def extract_structured_data(resume_text, gemini_model):
    if not gemini_model: return {"total_years_experience": 0, "education_level": "Not Found", "key_achievements": []}
    prompt = f"""Analyze the following resume text and extract the information into a valid JSON object with these exact keys: "total_years_experience", "education_level", "key_achievements".
- "total_years_experience": An integer for total years of professional experience.
- "education_level": A string, one of ["Bachelor's", "Master's", "PhD", "Other", "Not Found"].
- "key_achievements": A list of 2-3 strings of impactful achievements.
Resume Text: --- {resume_text} ---"""
    try:
        response = gemini_model.generate_content(prompt)
        json_str = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_str)
    except Exception as e:
        print(f"Error parsing structured data from Gemini: {e}")
        return {"total_years_experience": 0, "education_level": "Not Found", "key_achievements": []}

def analyze_resume(resume_text, jd_text, jd_skills_set, jd_embedding, requirements, sentence_model, gemini_model):
    # 1. Semantic Score
    resume_embedding = sentence_model.encode(resume_text)
    semantic_score = round(util.pytorch_cos_sim(resume_embedding, jd_embedding).item() * 100, 2)
    # 2. Skill Analysis & Score
    resume_skills = set(extract_skills(resume_text, SKILLS_DB))
    matching_skills = list(resume_skills & jd_skills_set)
    missing_skills = list(jd_skills_set - resume_skills)
    skill_score = round((len(matching_skills) / len(jd_skills_set)) * 100) if jd_skills_set else 0
    # 3. Structured Data Extraction & Scoring
    structured_data = extract_structured_data(resume_text, gemini_model)
    years_experience = structured_data.get("total_years_experience", 0)
    experience_score = min((years_experience / requirements['experience']) * 100, 120) if requirements['experience'] > 0 else 100
    edu_levels = {"Any": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
    education_score = 100 if edu_levels.get(structured_data.get("education_level"), 0) >= edu_levels[requirements['education']] else 0
    # 4. Weighted Final Score
    weights = {'semantic': 0.4, 'skills': 0.3, 'experience': 0.2, 'education': 0.1}
    final_score = round((semantic_score * weights['semantic']) + (skill_score * weights['skills']) + (experience_score * weights['experience']) + (education_score * weights['education']), 2)
    # 5. AI Summary
    ai_summary = generate_ai_summary(resume_text, jd_text, gemini_model)
    return {"final_score": final_score, "semantic_score": semantic_score, "skill_score": skill_score, "experience_score": experience_score, "education_score": education_score, "matching_skills": matching_skills, "missing_skills": missing_skills, "ai_summary": ai_summary, "structured_data": structured_data, "resume_text": resume_text}

@st.cache_data
def chunk_text(text, chunk_size=300, chunk_overlap=50):
    words = text.split(); chunks = [];
    for i in range(0, len(words), chunk_size - chunk_overlap): chunks.append(" ".join(words[i:i + chunk_size]));
    return chunks
@st.cache_resource
def create_faiss_index(_chunks, _sentence_model):
    embeddings = _sentence_model.encode(_chunks); index = faiss.IndexFlatL2(embeddings.shape[1]); index.add(embeddings.astype('float32')); return index
def search_faiss_index(index, question, chunks, sentence_model, k=3):
    question_embedding = sentence_model.encode([question]).astype('float32'); distances, indices = index.search(question_embedding, k); return [chunks[i] for i in indices[0]]
def answer_question_with_rag(question, index, chunks, sentence_model, gemini_model):
    if not gemini_model: return "Gemini model not available.", []
    relevant_context = search_faiss_index(index, question, chunks, sentence_model); context_str = "\n---\n".join(relevant_context)
    prompt = f"Based ONLY on the following context from a resume, answer the user's question. If the answer is not in the context, state that clearly.\n\nContext:\n---\n{context_str}\n---\n\nQuestion: {question}\n\nAnswer:";
    safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
    try:
        response = gemini_model.generate_content(prompt, safety_settings=safety_settings); return response.text.strip(), relevant_context
    except Exception as e: return f"Could not generate answer: {e}", []

def generate_skill_badges_html(skills_list, skill_type):
    if not skills_list: return "<i>None found.</i>"
    class_name = "matching-skill" if skill_type == "matching" else "missing-skill"
    badges_html = "".join([f'<span class="skill-badge {class_name}">{skill.title()}</span>' for skill in skills_list])
    return f'<div class="skill-badge-container">{badges_html}</div>'

# --- UI State Initialization ---
if 'results_df' not in st.session_state: st.session_state.results_df = None
if 'selected_candidate' not in st.session_state: st.session_state.selected_candidate = None

# --- MAIN APP LAYOUT ---
st.title("Enterprise AI Talent Platform ✨")

with st.sidebar:
    st.header("Job Requirements")
    job_description = st.text_area("1. Paste Job Description", height=200, key="jd_input")
    required_experience = st.number_input("2. Min. Years of Experience", min_value=0, max_value=30, value=3, key="exp_input")
    required_education = st.selectbox("3. Min. Education", ["Any", "Bachelor's", "Master's", "PhD"], key="edu_input")
    st.divider()
    st.header("Upload Resumes")
    uploaded_files = st.file_uploader("4. Upload Candidate Resumes", type=["pdf", "docx"], accept_multiple_files=True)

    if st.button("Rank Candidates", type="primary", use_container_width=True):
        if uploaded_files and job_description:
            with st.spinner(f"Performing multi-faceted analysis on {len(uploaded_files)} resumes..."):
                nlp, sentence_model, gemini_model = load_models()
                jd_skills_set = set(extract_skills(job_description, SKILLS_DB))
                jd_embedding = sentence_model.encode(job_description)
                requirements = {'experience': required_experience, 'education': required_education}
                results = []
                for file in uploaded_files:
                    resume_text = extract_text_from_file(file)
                    if resume_text:
                        analysis = analyze_resume(resume_text, job_description, jd_skills_set, jd_embedding, requirements, sentence_model, gemini_model)
                        analysis['filename'] = file.name
                        results.append(analysis)
                if results:
                    st.session_state.results_df = pd.DataFrame(results).sort_values(by='final_score', ascending=False)
                    st.session_state.selected_candidate = st.session_state.results_df.iloc[0]['filename']
                else: st.session_state.results_df = None; st.warning("Could not process any resumes.")
        else: st.warning("Please provide requirements and upload resumes.")

if st.session_state.results_df is None:
    st.info("Define job requirements and upload resumes in the sidebar to begin.")
else:
    master_col, detail_col = st.columns([1, 2], gap="large")
    with master_col:
        st.header("Ranked Candidates")
        for index, row in st.session_state.results_df.iterrows():
            if st.button(f"📄 {row['filename']} ({row['final_score']:.1f}%)", key=row['filename'], use_container_width=True):
                st.session_state.selected_candidate = row['filename']

    with detail_col:
        if st.session_state.selected_candidate:
            candidate_details = st.session_state.results_df[st.session_state.results_df['filename'] == st.session_state.selected_candidate].iloc[0]
            st.header(f"Suitability Report: {candidate_details['filename']}")
            
            st.subheader("Combined Suitability Score")
            st.progress(int(candidate_details['final_score']), text=f"{candidate_details['final_score']:.1f}%")
            
            st.subheader("Score Breakdown")
            score_col1, score_col2, score_col3, score_col4 = st.columns(4)
            score_col1.metric("Semantic Match", f"{candidate_details['semantic_score']:.1f}%")
            score_col2.metric("Skill Match", f"{candidate_details['skill_score']:.0f}%")
            score_col3.metric("Experience Match", f"{candidate_details['experience_score']:.0f}%")
            score_col4.metric("Education Match", f"{candidate_details['education_score']:.0f}%")
            st.divider()
            
            st.subheader("🤖 AI Summary & Key Achievements")
            st.info(candidate_details['ai_summary'])
            if candidate_details['structured_data'].get('key_achievements'):
                for ach in candidate_details['structured_data']['key_achievements']: st.markdown(f"- _{ach}_")
            st.divider()
            
            st.subheader("❓ Ask an Evidence-Based Question (RAG)")
            nlp, sentence_model, gemini_model = load_models()
            resume_chunks = chunk_text(candidate_details['resume_text'])
            faiss_index = create_faiss_index(resume_chunks, sentence_model)
            question = st.text_input("Ask a specific question:", key=f"rag_{candidate_details['filename']}")
            if question:
                with st.spinner("Searching resume with FAISS and generating answer..."):
                    answer, context = answer_question_with_rag(question, faiss_index, resume_chunks, sentence_model, gemini_model)
                    st.markdown(f"**Answer:** {answer}")
                    if context:
                        with st.expander("Show supporting evidence from resume"):
                            for sentence in context: st.markdown(f"<p class='quote'>{sentence}</p>", unsafe_allow_html=True)