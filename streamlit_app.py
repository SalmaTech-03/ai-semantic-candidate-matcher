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

# --- PAGE CONFIGURATION (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="AI Semantic Candidate Matcher",
    page_icon="✨",
    layout="wide"
)

# --- CUSTOM CSS FOR SKILL BADGES ---
st.markdown("""
<style>
.skill-badge-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 5px;
    margin-bottom: 15px;
}
.skill-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 16px;
    font-size: 14px;
    font-weight: 500;
    line-height: 1;
}
.matching-skill {
    background-color: rgba(4, 170, 109, 0.2);
    color: #04AA6D;
    border: 1px solid #04AA6D;
}
.missing-skill {
    background-color: rgba(255, 75, 75, 0.2);
    color: #FF4B4B;
    border: 1px solid #FF4B4B;
}
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

# --- BACKEND LOGIC (Helper Functions) ---
SKILLS_DB = [
    'python', 'java', 'c++', 'javascript', 'sql', 'git', 'docker', 'aws', 'azure', 'gcp', 'react', 'angular',
    'vue.js', 'node.js', 'flask', 'django', 'fastapi', 'html', 'css', 'machine learning', 'deep learning',
    'nlp', 'natural language processing', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch',
    'api', 'rest', 'graphql', 'communication', 'teamwork', 'problem-solving', 'agile', 'scrum', 'data analysis',
    'project management', 'product management'
]

def extract_text_from_file(file_uploader_object):
    """Extracts text from a file uploaded via Streamlit."""
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
    """Extracts a list of skills from text based on the SKILLS_DB."""
    found_skills = set()
    text_lower = text.lower()
    for skill in SKILLS_DB:
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found_skills.add(skill)
    return list(found_skills)

def generate_ai_summary(resume_text, jd_text, gemini_model):
    """Generates a brief summary for a candidate using the Gemini API."""
    if not gemini_model: return "Gemini model not available."
    safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
    prompt = f"Analyze the resume in context of the job description. Provide a concise, 3-sentence professional summary. JOB DESCRIPTION: {jd_text} --- RESUME: {resume_text[:4000]} ---"
    try:
        response = gemini_model.generate_content(prompt, safety_settings=safety_settings)
        return response.text.strip()
    except Exception as e:
        return f"Could not generate summary: {e}"

def analyze_resume(resume_text, jd_text, jd_skills_set, jd_embedding, sentence_model, gemini_model):
    """Performs a full analysis of a single resume."""
    resume_embedding = sentence_model.encode(resume_text)
    cosine_score = util.pytorch_cos_sim(resume_embedding, jd_embedding).item()
    match_percentage = round(cosine_score * 100, 2)
    resume_skills = set(extract_skills(resume_text))
    matching_skills = list(resume_skills & jd_skills_set)
    missing_skills = list(jd_skills_set - resume_skills)
    ai_summary = generate_ai_summary(resume_text, jd_text, gemini_model)
    return {"match_score": match_percentage, "matching_skills": matching_skills, "missing_skills": missing_skills, "ai_summary": ai_summary}

def create_skill_gap_chart(candidate_data, jd_skills_set):
    """Generates a Plotly Donut Chart for a single candidate's skill gap."""
    matching_count = len(candidate_data['matching_skills'])
    total_required = len(jd_skills_set)
    missing_count = total_required - matching_count if total_required >= matching_count else 0
    
    if total_required == 0:
        st.write("No skills were identified in the Job Description to create a chart.")
        return None

    match_percentage = (matching_count / total_required) * 100 if total_required > 0 else 0
    labels = ['Matching Skills', 'Missing Skills']
    values = [matching_count, missing_count]
    colors = ['#4CAF50', '#F44336']

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6, marker_colors=colors, textinfo='value', hoverinfo='label+percent')])
    fig.update_layout(
        title={'text': f"<b>Skill Coverage: {match_percentage:.1f}%</b>", 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        annotations=[dict(text=f'<b>{matching_count}</b><br>of<br><b>{total_required}</b>', x=0.5, y=0.5, font_size=20, showarrow=False)],
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=60, b=20, l=0, r=0)
    )
    return fig

def generate_skill_badges_html(skills_list, skill_type):
    """Generates a string of HTML for skill badges."""
    if not skills_list:
        return "<i>None found.</i>"
    class_name = "matching-skill" if skill_type == "matching" else "missing-skill"
    badges_html = "".join([f'<span class="skill-badge {class_name}">{skill.title()}</span>' for skill in skills_list])
    return f'<div class="skill-badge-container">{badges_html}</div>'

# --- UI State Initialization ---
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'selected_candidate' not in st.session_state:
    st.session_state.selected_candidate = None
if 'jd_skills' not in st.session_state:
    st.session_state.jd_skills = set()

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
                st.session_state.jd_skills = set(extract_skills(job_description))
                jd_embedding = sentence_model.encode(job_description)
                results = []
                for file in uploaded_files:
                    resume_text = extract_text_from_file(file)
                    if resume_text:
                        analysis = analyze_resume(resume_text, job_description, st.session_state.jd_skills, jd_embedding, sentence_model, gemini_model)
                        analysis['filename'] = file.name
                        results.append(analysis)
                
                if results:
                    st.session_state.results_df = pd.DataFrame(results).sort_values(by='match_score', ascending=False)
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
    master_col, detail_col = st.columns([1, 2], gap="large")

    with master_col:
        st.header("Ranked Candidates")
        st.write(f"Click on a resume to view details.")
        
        for index, row in st.session_state.results_df.iterrows():
            if st.button(f"📄 {row['filename']} ({row['match_score']:.2f}%)", key=row['filename'], use_container_width=True):
                st.session_state.selected_candidate = row['filename']

    with detail_col:
        if st.session_state.selected_candidate:
            candidate_details = st.session_state.results_df[st.session_state.results_df['filename'] == st.session_state.selected_candidate].iloc[0]
            
            st.header(f"Deep Dive Analysis: {candidate_details['filename']}")
            
            st.subheader("🤖 AI Summary")
            st.info(candidate_details['ai_summary'])
            st.divider()

            chart_col, skills_col = st.columns(2)
            with chart_col:
                st.subheader("📊 Visual Skill Gap")
                skill_chart = create_skill_gap_chart(candidate_details, st.session_state.jd_skills)
                if skill_chart:
                    st.plotly_chart(skill_chart, use_container_width=True)
            with skills_col:
                st.subheader("✅ Matching Skills")
                matching_html = generate_skill_badges_html(candidate_details['matching_skills'], "matching")
                st.markdown(matching_html, unsafe_allow_html=True)
                
                st.subheader("❌ Missing Skills")
                missing_html = generate_skill_badges_html(candidate_details['missing_skills'], "missing")
                st.markdown(missing_html, unsafe_allow_html=True)
        else:
            st.info("Select a candidate from the list on the left to see their detailed analysis.")