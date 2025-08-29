# AI Semantic Candidate Matcher ✨

This is an advanced web application that uses semantic search and generative AI to intelligently match candidate resumes with job descriptions. It goes beyond simple keyword searching to understand the contextual meaning of skills and experience, providing recruiters with a ranked and analyzed list of the most suitable candidates.



## Core Features
-   **Batch Resume Upload:** Analyze multiple resumes (`.pdf`, `.docx`) simultaneously against a single job description.
-   **High-Performance Semantic Matching:** Uses a `Sentence-Transformers` model to calculate a contextual match score, understanding synonyms and related concepts.
-   **Instant AI-Powered Summaries:** Integrates with the Google Gemini API (`gemini-1.5-flash`) to generate a concise, professional summary for each candidate.
-   **Interactive "Master-Detail" Dashboard:** A professional UI where clicking a candidate in the ranked list instantly displays their detailed report.
-   **Visual Skill Gap Analysis:** A dynamic and visually appealing Plotly Donut Chart represents the candidate's skill coverage at a glance.
-   **Polished UI:** Matching and missing skills are displayed as clean, colored badges for instant readability.

---

## Technology Stack

This project leverages a modern, Python-based stack designed for high-performance AI tasks and rapid, interactive web deployment.

| Category                | Technologies                                                                                                                                                                                                                                                                                                                                                                                                          |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Language & Frontend** | ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white) |
| **AI / ML Engine**      | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) ![Transformers](https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black) ![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=for-the-badge&logo=spacy&logoColor=white) ![Google Gemini](https://img.shields.io/badge/Google_Gemini-8E77D5?style=for-the-badge&logo=google&logoColor=white) |
| **Deployment & Tools**  | ![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white) ![Git](https://img.shields.io/badge/GIT-E44C30?style=for-the-badge&logo=git&logoColor=white)                                                                                                                                                                                                                         |

---

## How It Works: The Semantic Pipeline

The application follows a sophisticated pipeline to analyze and rank candidates:

1.  **Text Extraction:** Raw text is extracted from uploaded resume files (`.pdf`, `.docx`) using PyMuPDF and python-docx.
2.  **Embedding Generation:** The high-performance `all-MiniLM-L6-v2` Sentence-Transformer model converts the full text of the job description and each resume into a high-dimensional vector (embedding). This vector mathematically represents the contextual meaning of the text.
3.  **Similarity Calculation:** The **Cosine Similarity** is calculated between the job description's embedding and each resume's embedding. This produces a score from 0 to 1, indicating true semantic relevance.
4.  **Summarization & Analysis:** The top candidates are further analyzed.
    *   **Skill Extraction:** spaCy scans the text to find and list specific technical skills.
    *   **Generative Summary:** A carefully crafted prompt is sent with the resume and job text to the Google Gemini API, which returns a unique, professional summary.
5.  **Interactive Visualization:** The final data—scores, summaries, and skill lists—is presented in the interactive Streamlit dashboard.

---

## Deployment Journey

This project was a practical exercise in deploying resource-intensive AI applications.

-   **The Challenge:** The AI models, particularly PyTorch and Transformers, require more than 1GB of RAM. Initial deployment attempts on other free PaaS solutions failed due to strict memory limits (e.g., 512MB on Render), causing the application to crash on startup.
-   **The Solution:** The application was successfully deployed on **Streamlit Community Cloud**, which provides a generous free tier with the necessary memory (1GB+) and CPU resources to run heavy AI models.
-   **The Result:** A stable, performant, and completely free live application with a seamless CI/CD pipeline. Every `git push` to the `main` branch automatically triggers a new deployment.

---

## How to Run Locally

To run this application on your local machine, follow these steps.

### Prerequisites
-   Python 3.10+
-   Git

### Setup Instructions
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SalmaTech-03/ai-semantic-candidate-matcher.git
    cd ai-semantic-candidate-matcher
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    # Create and activate a virtual environment
    python -m venv .venv
    # On Windows:
    .venv\Scripts\activate
    # On macOS/Linux:
    # source .venv/bin/activate

    # Install the required libraries
    pip install -r requirements.txt
    ```

3.  **Set up your secrets:**
    *   Create a folder in your project root named `.streamlit`.
    *   Inside the `.streamlit` folder, create a new file named `secrets.toml`.
    *   Add your Google API key to this file in the following format:
        ```toml
        GOOGLE_API_KEY = "your_api_key_here"
        ```

4.  **Run the application:**
    ```bash
    streamlit run streamlit_app.py
    ```
