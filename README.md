# Enterprise AI Talent Platform ✨


An advanced, enterprise-grade application that demonstrates a full-stack, AI-driven solution to modern recruitment challenges. This platform moves beyond simple matching to provide a holistic, multi-faceted analysis of candidate suitability, incorporating explainable AI and scalable architecture principles.


## 🚀 Core Features (FAANG-Level)
-   **Multi-Faceted Scoring:** Ranks candidates not just on a single score, but on a weighted combination of **Semantic Match**, **Skill Coverage**, **Years of Experience**, and **Educational Background**.
-   **AI-Powered Structured Data Extraction:** Uses the Google Gemini LLM as a zero-shot parser to intelligently extract complex entities like "total years of experience" and "key achievements" from unstructured resume text.
-   **Evidence-Based Q&A (RAG):** Implements an advanced Retrieval-Augmented Generation pipeline. Recruiters can ask specific questions to a resume and receive answers backed by cited evidence from the text, powered by an in-memory **FAISS vector store**.
-   **Explainable AI (XAI):** The UI provides a full score breakdown, showing *why* a candidate was ranked highly and highlighting their key achievements.
-   **Interactive Dashboard:** A professional "master-detail" interface allows for rapid, fluid exploration of the ranked candidate pool.

---

## 🛠️ Technology Stack

| Category                      | Technologies                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Language & Frontend**       | ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)                                               |
| **AI / ML Engine**            | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) ![Transformers](https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black) ![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=for-the-badge&logo=spacy&logoColor=white) ![Google Gemini](https://img.shields.io/badge/Google_Gemini-8E77D5?style=for-the-badge&logo=google&logoColor=white) |
| **Vector Store & Retrieval** | ![Facebook AI](https://img.shields.io/badge/Facebook_AI-1877F2?style=for-the-badge&logo=facebook&logoColor=white) (FAISS)                                                                                                                                                                                                                                                                                                                                              |
| **Deployment & Version Control** | ![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white) ![Git](https://img.shields.io/badge/GIT-E44C30?style=for-the-badge&logo=git&logoColor=white)                                                                                                                                                                                                                                                                 |

---

## 🔬 Architectural Highlights

1.  **Text Extraction & Chunking:** Text is extracted and broken into smaller, overlapping chunks to create a dense context for retrieval.
2.  **Vector Store Creation:** On selecting a candidate, their resume chunks are instantly embedded and indexed into an in-memory **FAISS** vector store for ultra-fast similarity search.
3.  **Hybrid Analysis:**
    *   **Initial Ranking:** A fast, multi-faceted score (semantic, skill, etc.) is calculated for all candidates.
    *   **Deep Dive & RAG:** For a selected candidate, the system performs a deeper analysis, enabling the RAG Q&A pipeline to query the specific candidate's vector store.
4.  **Explainable Generation:** The Gemini LLM is prompted not just to answer questions, but to do so *only* using the context retrieved from the FAISS search, ensuring answers are grounded in the resume's actual text.

---

## 🚀 How to Run Locally

1.  **Clone & Setup:**
    ```bash
    git clone https://github.com/SalmaTech-03/ai-semantic-candidate-matcher.git
    cd ai-semantic-candidate-matcher
    python -m venv .venv
    .venv\Scripts\activate  # On Windows
    pip install -r requirements.txt
    ```
2.  **Set Secrets:** Create a `.streamlit/secrets.toml` file and add your `GOOGLE_API_KEY`.
3.  **Run:**
    ```bash
    streamlit run streamlit_app.py
    
