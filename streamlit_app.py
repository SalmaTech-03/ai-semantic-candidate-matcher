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

# --- LOAD MODELS (Cached for Performance) ---
@st.cache_resource
def load_models():
    """Loads all necessary AI models into memory."""
    print("Loading AI models...")
    nlp = spacy.load("en_core_web_sm")
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
    fig.update_layout(barmode='stack', title=f"Skill Gap Analysis for {candidate_data['filename']}", yaxis_title="Number of Skills", legend_title="Skill Type")
    return fig

# --- FLASHCARD PAGE FUNCTION ---
def render_flashcards_page():
    st.title("💡 Project Flashcards")
    st.write("Hover over any card to 'flip' it and see the details of the technologies and concepts used in this project.")
    st.divider()

    st.markdown("""
    <style>
    .flashcard-container { perspective: 1000px; height: 350px; width: 100%; margin-bottom: 20px; }
    .flashcard { width: 100%; height: 100%; position: relative; transform-style: preserve-3d; transition: transform 0.6s; border-radius: 15px; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); }
    .flashcard-container:hover .flashcard { transform: rotateY(180deg); }
    .flashcard-front, .flashcard-back { position: absolute; width: 100%; height: 100%; backface-visibility: hidden; display: flex; flex-direction: column; justify-content: center; align-items: center; padding: 20px; border-radius: 15px; text-align: center; }
    .flashcard-front { background-color: #262730; color: white; border: 1px solid #4f8bf9; }
    .flashcard-back { background-color: #4f8bf9; color: white; transform: rotateY(180deg); }
    .flashcard h3 { font-size: 24px; margin-bottom: 15px; } .flashcard p { font-size: 16px; line-height: 1.6; }
    </style>
    """, unsafe_allow_html=True)
    
    flashcards = {
        "The Core Idea": {"front": "What did I build?", "back": "An **AI Semantic Candidate Matcher**. A sophisticated web application that goes beyond keyword search to understand the contextual meaning of resumes and job descriptions, ranking candidates by their true suitability for a role."},
        "The User Interface": {"front": "How did I build the UI?", "back": "Using **Streamlit**. I created an interactive, multi-file upload interface and a dynamic results dashboard with progress bars, sortable tables, and a detailed \"deep-dive\" view with visual charts."},
        "The 'Brain' - Semantic Search": {"front": "What is the core AI technology?", "back": "**Semantic Search** using **Sentence Transformers**. I converted unstructured text into high-dimensional numerical vectors (embeddings) and calculated their **cosine similarity** to find the best contextual match."},
        "Generative AI Integration": {"front": "How are summaries generated?", "back": "By integrating the **Google Gemini API**. I used **prompt engineering** to instruct a powerful Large Language Model (LLM) to act as an HR assistant, generating unique, professional summaries for each candidate."},
        "Containerization & Deployment": {"front": "How is the app deployed?", "back": "The application is deployed on **Streamlit Community Cloud**. It's packaged with all its dependencies in a reproducible environment, a core concept of modern tools like **Docker** and **Kubernetes**."},
        "The Final Success": {"front": "What is the final result?", "back": "A **live, publicly accessible, high-performance AI application**. This is a professional and fully functional portfolio project that demonstrates a complete, end-to-end software engineering and AI skill set."}
    }

    cols = st.columns(2)
    for i, (key, card) in enumerate(flashcards.items()):
        with cols[i % 2]:
            st.subheader(key)
            st.html(f"""<div class="flashcard-container"><div class="flashcard"><div class="flashcard-front"><h3>{card['front']}</h3></div><div class="flashcard-back"><p>{card['back']}</p></div></div></div>""")

# --- MAIN MATCHER APP FUNCTION ---
def render_matcher_page():
    st.title("AI Semantic Candidate Matcher ✨")
    st.write("An advanced tool to screen, rank, and analyze candidates. Upload resumes and a job description to begin.")

    if 'results_df' not in st.session_state:
        st.session_state.results_df = None

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
                    else:
                        st.session_state.results_df = None
                        st.warning("Could not process any of the uploaded resumes.")
            else:
                st.warning("Please upload resumes and a job description.")

        if st.session_state.results_df is not None:
            st.divider()
            st.header("Interactive Filters")
            score_threshold = st.slider("Minimum Match Score (%)", 0, 100, 50)
            df_filtered = st.session_state.results_df[st.session_state.results_df['match_score'] >= score_threshold]
        else:
            df_filtered = pd.DataFrame()

    if st.session_state.results_df is None:
        st.info("Upload resumes and a job description in the sidebar to get started.")
    else:
        st.header("Ranked & Filtered Candidate Results")
        st.write(f"Showing {len(df_filtered)} of {len(st.session_state.results_df)} total candidates.")

        if not df_filtered.empty:
            df_display = df_filtered.copy()
            df_display['Select'] = False
            df_display['matching_skill_count'] = df_display['matching_skills'].apply(len)
            df_display['missing_skill_count'] = df_display['missing_skills'].apply(len)
            
            edited_df = st.data_editor(
                df_display[['Select', 'filename', 'match_score', 'matching_skill_count', 'missing_skill_count']],
                use_container_width=True,
                column_config={'Select': st.column_config.CheckboxColumn("Select", width="small"), 'filename': "Candidate Resume", 'match_score': st.column_config.ProgressColumn("Match Score (%)", format="%.2f%%", min_value=0, max_value=100), 'matching_skill_count': "Matching Skills", 'missing_skill_count': "Missing Skills"},
                hide_index=True, key="candidate_selector"
            )
            
            selected_row = edited_df[edited_df.Select]
            if not selected_row.empty:
                selected_filename = selected_row.iloc[0]['filename']
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

# --- NAVIGATION ---
PAGES = {
    "Candidate Matcher": render_matcher_page,
    "Project Flashcards": render_flashcards_page
}

st.sidebar.divider()
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page()