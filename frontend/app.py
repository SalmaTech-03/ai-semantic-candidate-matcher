import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Semantic Candidate Matcher",
    page_icon="✨",
    layout="wide"
)

# --- Backend API URL ---
# This is the new, correct line 14
BACKEND_URL = "http://localhost:5001/api/batch-analyze"

# --- Initialize Session State ---
if 'app_state' not in st.session_state:
    st.session_state.app_state = {
        "results": None,
        "jd_skills": [],
        "candidate_statuses": {}
    }

# --- Helper Functions ---
@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV string for download."""
    return df.to_csv(index=False).encode('utf-8')

def create_skill_gap_chart(candidate):
    """Generates a Plotly bar chart for a single candidate's skill gap."""
    matching_count = len(candidate['matching_skills'])
    missing_count = len(candidate['missing_skills'])
    
    fig = go.Figure(data=[
        go.Bar(name='Matching Skills', x=['Skills'], y=[matching_count], marker_color='#4CAF50', text=matching_count, textposition='auto'),
        go.Bar(name='Missing Skills', x=['Skills'], y=[missing_count], marker_color='#F44336', text=missing_count, textposition='auto')
    ])
    fig.update_layout(
        barmode='stack',
        title=f"Skill Gap for {candidate['filename']}",
        yaxis_title="Number of Skills",
        legend_title="Skill Type"
    )
    return fig

# --- Candidate Report (fixed: use expander instead of dialog) ---
def show_candidate_report(candidate):
    """Displays a clean, shareable report for a single candidate."""
    with st.expander(f"Candidate Report: {candidate['filename']}", expanded=True):
        st.header(f"Analysis for: {candidate['filename']}")
        st.subheader(f"Overall Match Score: {candidate['match_score']:.2f}%")
        st.progress(int(candidate['match_score']))
        st.divider()

        st.subheader("🤖 AI Summary")
        st.info(candidate['ai_summary'])
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("✅ Matching Skills")
            st.write(", ".join(candidate['matching_skills']))
        with col2:
            st.subheader("❌ Missing Skills")
            st.write(", ".join(candidate['missing_skills']))

# --- Sidebar for Inputs & Filters ---
with st.sidebar:
    st.header("Inputs")
    uploaded_files = st.file_uploader(
        "Upload Resumes (PDF, DOCX)", type=["pdf", "docx"], accept_multiple_files=True
    )
    job_description = st.text_area("Paste Job Description", height=250)

    if st.button("Rank Candidates", type="primary", use_container_width=True):
        if uploaded_files and job_description:
            with st.spinner(f"Analyzing {len(uploaded_files)} resumes..."):
                files = [("resumes", (file.name, file.getvalue(), file.type)) for file in uploaded_files]
                data = {"job_description": job_description}
                try:
                    response = requests.post(BACKEND_URL, files=files, data=data)
                    if response.status_code == 200:
                        api_response = response.json()
                        st.session_state.app_state["results"] = api_response.get("candidates", [])
                        st.session_state.app_state["jd_skills"] = api_response.get("jd_skills", [])
                        st.session_state.app_state["candidate_statuses"] = {
                            res['filename']: "Pending" for res in st.session_state.app_state["results"] if 'filename' in res
                        }
                        st.success("Analysis Complete!")
                    else:
                        st.error(f"Backend Error: {response.json().get('error', 'Unknown error')}")
                except requests.exceptions.RequestException:
                    st.error("Connection Error: Could not connect to the backend.")
        else:
            st.warning("Please upload resumes and a job description.")

    if st.session_state.app_state["results"]:
        st.divider()
        st.header("Interactive Filters")
        score_threshold = st.slider("Minimum Match Score (%)", 0, 100, 50)
        must_have_skills = st.multiselect("Filter by Must-Have Skills", options=st.session_state.app_state["jd_skills"])
        status_filter = st.multiselect("Filter by Status", options=["Pending", "Shortlisted", "Rejected"])

# --- Main Page Layout ---
st.title("AI Semantic Candidate Matcher ✨")

if not st.session_state.app_state["results"]:
    st.info("Upload resumes and a job description in the sidebar to get started.")
else:
    df = pd.DataFrame(st.session_state.app_state["results"])
    df['Status'] = df['filename'].map(st.session_state.app_state["candidate_statuses"])

    df_filtered = df[df['match_score'] >= score_threshold]
    if must_have_skills:
        df_filtered = df_filtered[df_filtered['matching_skills'].apply(lambda s: all(r in s for r in must_have_skills))]
    if status_filter:
        df_filtered = df_filtered[df_filtered['Status'].isin(status_filter)]

    st.header("Ranked & Filtered Candidate Results")
    st.write(f"Showing {len(df_filtered)} of {len(df)} candidates.")

    if not df_filtered.empty:
        df_display = df_filtered.copy().sort_values(by='match_score', ascending=False)
        df_display['Rank'] = range(1, len(df_display) + 1)
        df_display['View Report'] = [False] * len(df_display)

        edited_df = st.data_editor(
            df_display[['Rank', 'filename', 'Status', 'match_score', 'ai_summary', 'View Report']],
            use_container_width=True,
            column_config={
                'filename': st.column_config.TextColumn("Candidate Resume", width="medium"),
                'Status': st.column_config.SelectboxColumn("Status", options=["Pending", "Shortlisted", "Rejected"], width="small"),
                'match_score': st.column_config.ProgressColumn("Match Score", format="%.2f%%", min_value=0, max_value=100),
                'ai_summary': st.column_config.TextColumn("AI Summary", width="large"),
                'View Report': st.column_config.CheckboxColumn("View Report", width="small")
            },
            hide_index=True,
            key="candidate_table"
        )
        
        report_candidate_row = edited_df[edited_df['View Report']].iloc[0] if not edited_df[edited_df['View Report']].empty else None
        
        if report_candidate_row is not None:
            candidate_full_data = df[df['filename'] == report_candidate_row['filename']].iloc[0]
            show_candidate_report(candidate_full_data)

        # Update status dictionary
        for index, row in edited_df.iterrows():
            st.session_state.app_state["candidate_statuses"][row["filename"]] = row["Status"]

        csv_data = convert_df_to_csv(edited_df.drop(columns=['View Report']))
        st.download_button("📥 Download Filtered Results as CSV", csv_data, "filtered_candidates.csv", "text/csv")
    else:
        st.warning("No candidates match the current filter criteria.")
