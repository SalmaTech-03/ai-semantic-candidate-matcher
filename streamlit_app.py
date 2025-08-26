# streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import spacy
import fitz  # PyMuPDF
import docx
import re
import phonenumbers
from sentence_transformers import SentenceTransformer, util
import os
import google.generativeai as genai

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Semantic Candidate Matcher",
    page_icon="✨",
    layout="wide"
)

# --- Load Models (Cached for Performance) ---
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception:
        gemini_model = None
    return nlp, sentence_model, gemini_model

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

# --- Helper Functions ---
def extract_text(file_obj):
    text = ""
    ext = os.path.splitext(file_obj.name)[1].lower()
    try:
        if ext == ".pdf":
            with fitz.open(stream=file_obj.getvalue(), filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()
        elif ext == ".docx":
            doc = docx.Document(file_obj)
            for para in doc.paragraphs:
                text += para.text + "\n"
    except Exception as e:
        st.error(f"Error reading {file_obj.name}: {e}")
    return text

def extract_contact_info(text):
    name, email, phone = None, None, None
    if nlp:
        doc = nlp(text[:300])
        for ent in doc.ents:
            if ent.label_ == "PERSON":
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
    return name or "Not Found", email or "Not Found", phone or "Not Found"

def extract_skills(text):
    found_skills = set()
    text_lower = text.lower()
    for skill in SKILLS_DB:
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found_skills.add(skill)
    return list(found_skills)

def generate_ai_summary(resume_text, jd_text):
    if not gemini_model:
        return "Gemini model not available"
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    prompt = f"""
    Analyze the resume in context of the job description. Provide a 3-sentence professional summary.
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

def analyze_resume(resume_text, jd_text, jd_skills_set, jd_embedding):
    # Semantic similarity
    resume_embedding = sentence_model.encode(resume_text, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(resume_embedding, jd_embedding).item()
    match_score = round(cosine_score * 100, 2)

    # Skills
    resume_skills = extract_skills(resume_text)
    matching_skills = list(set(resume_skills) & jd_skills_set)
    missing_skills = list(jd_skills_set - set(resume_skills))

    # Contact
    name, email, phone = extract_contact_info(resume_text)

    # AI Summary
    ai_summary = generate_ai_summary(resume_text, jd_text)

    return {
        "match_score": match_score,
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "ai_summary": ai_summary,
        "name": name,
        "email": email,
        "phone": phone
    }

def create_skill_gap_chart(candidate):
    matching = len(candidate['matching_skills'])
    missing = len(candidate['missing_skills'])
    fig = go.Figure(data=[
        go.Bar(name='Matching', x=['Skills'], y=[matching], marker_color='#4CAF50', text=matching, textposition='auto'),
        go.Bar(name='Missing', x=['Skills'], y=[missing], marker_color='#F44336', text=missing, textposition='auto')
    ])
    fig.update_layout(
        barmode='stack',
        title=f"Skill Gap: {candidate['name']}",
        yaxis_title="Number of Skills"
    )
    return fig

# --- Streamlit UI ---
st.title("AI Semantic Candidate Matcher ✨")
st.write("Upload resumes and a job description to rank candidates.")

# Inputs
col1, col2 = st.columns([1, 2])
with col1:
    uploaded_files = st.file_uploader("Upload Resumes (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
with col2:
    job_description = st.text_area("Paste Job Description", height=250)

if st.button("Rank Candidates"):
    if uploaded_files and job_description:
        with st.spinner(f"Analyzing {len(uploaded_files)} resumes..."):
            jd_skills_set = set(extract_skills(job_description))
            jd_embedding = sentence_model.encode(job_description, convert_to_tensor=True)
            results = []
            for file in uploaded_files:
                text = extract_text(file)
                if text:
                    analysis = analyze_resume(text, job_description, jd_skills_set, jd_embedding)
                    analysis['filename'] = file.name
                    results.append(analysis)

            if results:
                df = pd.DataFrame(results).sort_values(by='match_score', ascending=False)
                df['matching_count'] = df['matching_skills'].apply(len)
                df['missing_count'] = df['missing_skills'].apply(len)

                # Display table
                st.header("Ranked Candidates")
                st.dataframe(
                    df[['filename','name','email','phone','match_score','matching_count','missing_count','ai_summary']],
                    use_container_width=True
                )

                # Skill gap chart per candidate
                st.header("Skill Gap Visualization")
                for candidate in results:
                    st.subheader(candidate['name'])
                    st.plotly_chart(create_skill_gap_chart(candidate), use_container_width=True)

                # Download CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results CSV", csv, "candidate_results.csv", "text/csv")
    else:
        st.warning("Please upload at least one resume and provide a job description.")
