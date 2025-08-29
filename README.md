# AI Semantic Candidate Matcher ✨

This is an advanced web application that uses semantic search and generative AI to intelligently match candidate resumes with job descriptions. It goes beyond simple keyword searching to understand the contextual meaning of skills and experience, providing recruiters with a ranked and analyzed list of the most suitable candidates.



## Core Features
- **Batch Resume Upload:** Analyze multiple resumes (`.pdf`, `.docx`) simultaneously.
- **Semantic Matching:** Uses a `Sentence-Transformers` model to calculate a contextual match score between each resume and the job description.
- **AI-Powered Summaries:** Integrates with the Google Gemini API to generate a concise, professional summary for each candidate.
- **Interactive Dashboard:** A "master-detail" view allows users to click on a ranked candidate to see a detailed analysis.
- **Visual Skill Gap Analysis:** A dynamic donut chart visually represents the candidate's matching vs. missing skills.
- **Modern UI:** Skills are displayed as clean, colored badges for instant readability.

## Technology Stack
*   **Language:** Python 3.10
*   **Frontend:** Streamlit
*   **Semantic Search:** Sentence-Transformers, PyTorch, Transformers
*   **Generative AI:** Google Gemini API (`gemini-2.5-flash`)
*   **NLP:** spaCy
*   **Data Handling:** Pandas
*   **Visualization:** Plotly
*   **Deployment:** Streamlit Community Cloud
*   **Version Control:** Git & GitHub

## How to Run Locally
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/your-repo-name.git
    cd your-repo-name
    ```
2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    pip install -r requirements.txt
    ```
3.  **Set up your secrets:**
    *   Create a file named `.streamlit/secrets.toml`.
    *   Add your Google API key to it:
        ```toml
        GOOGLE_API_KEY = "your_api_key_here"
        ```
4.  **Run the application:**
    ```bash
    streamlit run streamlit_app.py
    ```
