import streamlit as st
import httpx
import asyncio

st.set_page_config(page_title="Enterprise AI Talent Platform", layout="wide")
st.title("Enterprise AI Talent Platform ✨")

with st.sidebar:
    st.header("Job Requirements")
    jd = st.text_area("1. Paste Job Description", height=200)
    files = st.file_uploader("2. Upload Resumes", type=["pdf", "docx"], accept_multiple_files=True)

async def call_backend(jd_text, uploaded_files):
    async with httpx.AsyncClient(timeout=120.0) as client:
        data = {"jd": jd_text}
        files_data = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
        response = await client.post("http://localhost:8000/analyze", data=data, files=files_data)
        return response.json()

if st.button("Rank Candidates", type="primary"):
    if jd and files:
        with st.spinner(f"Analyzing {len(files)} resumes..."):
            try:
                results = asyncio.run(call_backend(jd, files))
                for res in results:
                    with st.expander(f"📄 {res['name']} - Score: {res['score']}%"):
                        st.write(f"**Reasoning:** {res['details']['reasoning']}")
                        st.write(f"**Summary:** {res['summary']}")
            except Exception as e:
                st.error(f"Backend connection failed. Is the API running on Port 8000? ({e})")