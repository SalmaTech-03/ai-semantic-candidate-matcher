# AI Semantic Candidate Matcher

AI Semantic Candidate Matcher is a Python application that ranks resumes against a job description by combining semantic similarity from sentence embeddings with structured resume information extracted using Google Gemini.

The application is designed to reduce manual resume screening by comparing candidate resumes with a job description using contextual similarity instead of relying only on keyword matching. It extracts structured information such as years of experience, education, skills, and a short summary, then combines that information with semantic similarity to produce a weighted ranking.

The project separates document parsing, AI integration, orchestration, scoring, and API handling into independent modules. This modular organization makes the application easier to understand, maintain, and extend.

---

# Problem Statement

Recruiters often receive a large number of resumes for a single job opening.

Traditional Applicant Tracking Systems (ATS) primarily rely on keyword matching. This approach can miss qualified candidates who describe their experience using different terminology.

For example:

* A job description may mention **REST APIs**
* A resume may mention **backend service development**

Although these concepts are related, keyword-based systems may not recognize the similarity.

This project addresses that limitation by:

* extracting text from uploaded resumes
* generating semantic embeddings
* comparing resumes with the job description using cosine similarity
* extracting structured candidate information using Google Gemini
* producing a weighted ranking based on semantic similarity and experience

The goal is to assist recruiters by providing an additional ranking mechanism rather than replacing manual evaluation.

---

# Project Goals

The project focuses on the following objectives:

* Parse PDF and DOCX resumes.
* Extract readable text from supported documents.
* Mask email addresses and phone numbers before sending resume text to the language model.
* Generate semantic embeddings for contextual similarity.
* Extract structured candidate information using Google Gemini.
* Calculate a weighted candidate score.
* Present ranked candidates through a Streamlit interface.
* Expose resume analysis through a FastAPI endpoint.

---

# Features

## Document Parsing

The application supports:

* PDF documents
* Microsoft Word (.docx) documents

Document parsing is handled by the `SecureParser` component.

Supported libraries:

* PyMuPDF
* python-docx
* python-magic

The parser detects the MIME type of the uploaded file and selects the appropriate parser.

If parsing fails or the document format is unsupported, an empty string is returned and the document is skipped.

---

## Resume Text Extraction

After parsing, the extracted document text becomes the input for the remaining pipeline.

The current implementation supports text-based documents.

Scanned PDFs and image-based resumes are not processed because OCR is not implemented.

---

## PII Scrubbing

Before resume text is sent to Google Gemini, the application masks common personally identifiable information.

Currently masked:

* Email addresses
* Phone numbers

This helps reduce unnecessary exposure of personal information during AI processing.

Masking is performed before constructing the LLM prompt.

---

## AI-Based Resume Analysis

The application uses Google Gemini to extract structured resume information.

The current implementation requests the following fields:

* years of experience
* education
* skills
* summary

If the AI response cannot be parsed successfully, fallback values are returned instead of terminating the request.

This allows the remaining scoring pipeline to continue operating.

---

## Semantic Similarity

The application generates sentence embeddings using the Sentence Transformers model:

`all-MiniLM-L6-v2`

Both the resume text and the job description are converted into dense vectors.

Cosine similarity is then calculated between these vectors to estimate contextual similarity.

Unlike simple keyword matching, semantic embeddings can identify conceptually related language.

---

## Weighted Candidate Scoring

Candidate ranking is based on two factors.

### Semantic Similarity

Measures how closely the resume content matches the supplied job description.

### Experience Alignment

Normalizes the extracted years of experience against a five-year benchmark.

Both values are combined using configurable weights defined in:

`src/core/config.py`

This produces the final candidate score returned by the API.

---

## Logging

The project includes a reusable logging utility.

Application events and parsing failures are logged using Python's logging module.

The logger currently outputs formatted log messages to standard output.

---

## Docker Support

A Docker Compose configuration is included for running both services locally.

Current services:

* FastAPI backend
* Streamlit frontend

---

# High-Level Architecture

The application follows a layered architecture.

Each layer has a specific responsibility.

```text
                +-----------------------+
                |    Streamlit UI       |
                +-----------+-----------+
                            |
                            v
                +-----------------------+
                |      FastAPI API      |
                +-----------+-----------+
                            |
                            v
               +-------------------------+
               |   TalentOrchestrator    |
               +-----------+-------------+
                           |
        +------------------+------------------+
        |                  |                  |
        v                  v                  v
+---------------+   +---------------+   +---------------+
| SecureParser  |   |   AIClient    |   | TalentScorer |
+---------------+   +---------------+   +---------------+
        |                  |
        |                  |
        +--------+---------+
                 |
                 v
     Sentence Transformer Embeddings
```

The `TalentOrchestrator` coordinates communication between each component.

It is responsible for:

* parsing uploaded documents
* generating embeddings
* invoking Google Gemini
* calculating semantic similarity
* generating the final scorecard
* returning ranked results

---

# Request Flow

The following sequence describes how a request is processed.

1. The user enters a job description.
2. One or more resumes are uploaded.
3. Streamlit sends a multipart request to the FastAPI backend.
4. FastAPI passes the request to `TalentOrchestrator`.
5. The parser extracts document text.
6. Personally identifiable information is masked.
7. Google Gemini extracts structured candidate information.
8. Sentence embeddings are generated.
9. Cosine similarity is calculated.
10. Experience is normalized.
11. The weighted score is calculated.
12. Results are sorted by score.
13. Ranked candidates are returned to the frontend.

---

# Design Decisions

## Layered Organization

The project separates:

* API endpoints
* business logic
* infrastructure
* configuration
* utility functions

This reduces coupling between components and keeps responsibilities clearly defined.

---

## Orchestrator Pattern

Rather than allowing API endpoints to directly coordinate parsing, AI calls, and scoring, the application delegates this workflow to a dedicated orchestrator.

This keeps request handlers relatively small while centralizing the processing pipeline.

---

## Semantic Similarity Instead of Keyword Matching

Sentence embeddings were chosen because they compare the contextual meaning of text rather than requiring exact keyword matches.

This improves matching when resumes and job descriptions describe similar concepts using different wording.

---

## Environment-Based Configuration

Application configuration is managed using Pydantic Settings.

Sensitive values such as API keys are loaded from environment variables instead of being hardcoded.

Scoring weights are also configurable without modifying application logic.

---

## Privacy Before AI Processing

Resume text is scrubbed before it is sent to Google Gemini.

Although the current implementation masks only email addresses and phone numbers, separating this logic into a dedicated utility makes it easier to extend with additional masking rules in the future.

# Technology Stack

The application is implemented entirely in Python and combines web frameworks, natural language processing libraries, and Google's Gemini API.

| Category         | Technology                                 | Purpose                                     |
| ---------------- | ------------------------------------------ | ------------------------------------------- |
| Language         | Python 3.10+                               | Primary programming language                |
| Backend API      | FastAPI                                    | REST API for resume analysis                |
| Frontend         | Streamlit                                  | User interface                              |
| LLM              | Google Gemini 1.5 Flash                    | Structured resume information extraction    |
| Embeddings       | Sentence Transformers (`all-MiniLM-L6-v2`) | Semantic similarity                         |
| Vector Math      | NumPy                                      | Cosine similarity calculation               |
| Document Parsing | PyMuPDF                                    | PDF text extraction                         |
| Document Parsing | python-docx                                | DOCX text extraction                        |
| MIME Detection   | python-magic                               | File type validation                        |
| Configuration    | Pydantic Settings                          | Environment configuration                   |
| Validation       | Pydantic                                   | Request and response schemas                |
| HTTP Client      | httpx                                      | Communication between Streamlit and FastAPI |
| Containerization | Docker Compose                             | Local multi-service deployment              |

---

# Project Structure

```text
.
├── src/
│   ├── api/
│   │   └── main.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   └── schemas.py
│   │
│   ├── infra/
│   │   ├── ai_clients.py
│   │   └── parser.py
│   │
│   ├── services/
│   │   ├── orchestrator.py
│   │   └── scoring.py
│   │
│   └── utils/
│       ├── logger.py
│       └── scrubber.py
│
├── streamlit_app.py
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .env
```

---

# Directory Overview

## `src/api`

Contains the FastAPI application.

Responsibilities include:

* exposing REST endpoints
* receiving uploaded files
* validating incoming requests
* returning ranked candidate results

Current endpoint:

```
POST /analyze
```

---

## `src/core`

Contains shared application configuration and data models.

### `config.py`

Defines application settings using Pydantic Settings.

Examples include:

* Gemini API key
* scoring weights
* application metadata

### `schemas.py`

Defines structured models returned throughout the application.

Examples:

* CandidateAnalysis
* Scorecard

Using schemas provides validation and a consistent data contract.

---

## `src/infra`

Contains infrastructure-related components.

### `parser.py`

Responsible for:

* MIME detection
* PDF parsing
* DOCX parsing

This layer isolates file-processing logic from business logic.

### `ai_clients.py`

Responsible for:

* constructing prompts
* masking PII
* calling Google Gemini
* parsing JSON responses
* returning fallback values if extraction fails

Keeping AI integration separate makes it easier to replace the provider later if needed.

---

## `src/services`

Contains application business logic.

### `orchestrator.py`

Coordinates the complete processing workflow.

Responsibilities include:

* document parsing
* AI analysis
* embedding generation
* cosine similarity
* scoring
* assembling the response

### `scoring.py`

Calculates candidate scores.

Separating scoring into its own service keeps ranking logic independent from parsing and AI integration.

---

## `src/utils`

Contains reusable helper utilities.

### `logger.py`

Creates configured logger instances.

### `scrubber.py`

Masks email addresses and phone numbers before resume text is sent to Google Gemini.

---

## `streamlit_app.py`

Provides the user interface.

Responsibilities include:

* collecting the job description
* uploading resumes
* calling the backend API
* displaying ranked candidates

The frontend contains minimal business logic.

---

# Installation

## Prerequisites

Before running the application, ensure the following are installed.

* Python 3.10 or later
* Git
* pip

The project also requires the native **libmagic** library because `python-magic` depends on it.

### Ubuntu / Debian

```bash
sudo apt-get install libmagic1
```

### macOS

```bash
brew install libmagic
```

### Windows

`python-magic` may require additional setup depending on the installation method. Refer to the project's documentation for platform-specific instructions.

---

# Clone the Repository

```bash
git clone https://github.com/SalmaTech-03/AI-semantic-candidate-matcher.git

cd AI-semantic-candidate-matcher
```

---

# Create a Virtual Environment

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

The first execution of the application may download the Sentence Transformers model used for embeddings.

---

# Environment Configuration

Create a file named:

```
.env
```

Example:

```text
GEMINI_API_KEY=your_api_key_here
```

The application loads configuration using Pydantic Settings.

---

# Running the Backend

Start the FastAPI server.

```bash
uvicorn src.api.main:app --reload
```

Default URL:

```
http://localhost:8000
```

---

# Running the Frontend

Open another terminal.

Run:

```bash
streamlit run streamlit_app.py
```

Default URL:

```
http://localhost:8501
```

---

# Running with Docker Compose

The repository includes a Docker Compose configuration.

Build and start both services:

```bash
docker compose up --build
```

The following services will start.

| Service            | Port |
| ------------------ | ---- |
| FastAPI Backend    | 8000 |
| Streamlit Frontend | 8501 |

---

# Configuration

Application configuration is centralized in:

```
src/core/config.py
```

Current configurable values include:

| Setting             | Description                             |
| ------------------- | --------------------------------------- |
| `APP_NAME`          | Application name                        |
| `API_V1_STR`        | API prefix                              |
| `GEMINI_API_KEY`    | Google Gemini API key                   |
| `EMBEDDING_MODEL`   | Sentence Transformer model name         |
| `WEIGHT_SEMANTIC`   | Weight assigned to semantic similarity  |
| `WEIGHT_EXPERIENCE` | Weight assigned to experience alignment |

Changing the scoring weights allows experimentation without modifying the scoring implementation.

---

# Development Workflow

A typical request follows these steps during development:

1. Start the FastAPI backend.
2. Start the Streamlit application.
3. Paste a job description.
4. Upload one or more PDF or DOCX resumes.
5. Submit the request.
6. The backend parses each resume.
7. Resume text is scrubbed before AI processing.
8. Google Gemini extracts structured information.
9. Sentence embeddings are generated.
10. Cosine similarity is calculated.
11. Scores are generated.
12. Ranked candidates are displayed in the Streamlit interface.

---

# Dependencies

The project currently depends on the following major libraries.

* FastAPI
* Streamlit
* Google Generative AI SDK
* Sentence Transformers
* NumPy
* PyMuPDF
* python-docx
* python-magic
* Pydantic
* Pydantic Settings
* SQLAlchemy (listed in `requirements.txt` but not used in the current implementation)
* httpx
* uvicorn

Where practical, dependencies are isolated to the modules that require them to reduce coupling between components.

# API Documentation

The backend exposes a REST API built with FastAPI. The current implementation provides a single endpoint for processing resumes against a job description.

## Base URL

```text
http://localhost:8000
```

---

## Analyze Resumes

**Endpoint**

```http
POST /analyze
```

### Request Type

```text
multipart/form-data
```

### Form Fields

| Field   | Type   | Required | Description                     |
| ------- | ------ | -------- | ------------------------------- |
| `jd`    | String | Yes      | Job description text            |
| `files` | File[] | Yes      | One or more PDF or DOCX resumes |

### Example Request

```bash
curl -X POST http://localhost:8000/analyze \
  -F "jd=Python developer with FastAPI experience" \
  -F "files=@resume1.pdf" \
  -F "files=@resume2.docx"
```

### Example Response

```json
[
  {
    "name": "resume.pdf",
    "score": 82.3,
    "details": {
      "overall_score": 82.3,
      "semantic_match": 85.0,
      "experience_alignment": 79.5,
      "reasoning": "Candidate shows 4 years of experience. Semantic match to JD is 85.0%."
    },
    "summary": "Experienced Python developer with FastAPI and REST API development."
  }
]
```

---

# Resume Processing Workflow

Each uploaded resume follows the same processing pipeline.

```text
Upload Resume
      │
      ▼
Validate File Type
      │
      ▼
Extract Text
      │
      ▼
Mask Email & Phone Number
      │
      ▼
Send to Google Gemini
      │
      ▼
Extract Structured Information
      │
      ▼
Generate Resume Embedding
      │
      ▼
Generate Job Description Embedding
      │
      ▼
Calculate Cosine Similarity
      │
      ▼
Generate Final Score
      │
      ▼
Return Ranked Results
```

This workflow is coordinated by the `TalentOrchestrator` service.

---

# Document Parsing

Resume parsing is implemented in:

```text
src/infra/parser.py
```

Currently supported formats include:

* PDF
* DOCX

The parser performs MIME type detection using `python-magic` before attempting extraction.

PDF files are processed using PyMuPDF.

DOCX files are processed using `python-docx`.

If parsing fails, an empty string is returned and the resume is skipped.

---

# PII Scrubbing

Before resume text is sent to Google Gemini, personally identifiable information is masked.

Current implementation masks:

* Email addresses
* Phone numbers

Example:

```text
Original

john.doe@email.com
+91 9876543210

↓

Masked

[EMAIL]
[PHONE]
```

This logic is implemented in:

```text
src/utils/scrubber.py
```

---

# AI Processing

Google Gemini is responsible for extracting structured information from resumes.

The prompt instructs the model to return JSON containing:

* years of experience
* education
* skills
* summary

The application then parses the returned JSON.

If parsing fails, fallback values are returned so processing can continue.

Fallback values include:

```json
{
  "years_exp": 0,
  "education": "Not Found",
  "skills": [],
  "summary": "AI extraction failed for this document."
}
```

---

# Semantic Similarity

The project uses the Sentence Transformers model:

```text
all-MiniLM-L6-v2
```

Both the job description and resume are converted into dense vector embeddings.

Similarity is computed using cosine similarity.

Mathematically,

```text
cosine_similarity =
(A · B)
/ (||A|| × ||B||)
```

where:

* A = Job Description embedding
* B = Resume embedding

A higher cosine similarity indicates greater semantic similarity between the resume and the job description.

---

# Scoring Algorithm

The final score combines two independent components.

## 1. Semantic Similarity

Obtained from cosine similarity.

Configured using:

```python
WEIGHT_SEMANTIC
```

---

## 2. Experience Alignment

Calculated from the extracted years of experience.

Current implementation normalizes experience using a fixed five-year benchmark.

Example:

```text
5 years → 100%

2.5 years → 50%

10 years → capped at 100%
```

Configured using:

```python
WEIGHT_EXPERIENCE
```

---

## Overall Score

The implementation calculates:

```text
overall_score =
(semantic_similarity × WEIGHT_SEMANTIC)
+
(experience_alignment × WEIGHT_EXPERIENCE)
```

Both weights are configurable through `src/core/config.py`.

---

# Logging

Logging utilities are located in:

```text
src/utils/logger.py
```

The application uses Python's built-in logging module.

Log messages include:

* parsing failures
* AI extraction failures
* general application events

Current configuration writes logs to standard output.

---

# Error Handling

The application handles several failure scenarios gracefully.

## Invalid File Type

Unsupported MIME types return an empty result.

---

## Parsing Failure

If PDF or DOCX parsing fails:

* the exception is logged
* processing continues

---

## Gemini Failure

If Gemini returns invalid JSON or raises an exception:

* the error is logged
* fallback analysis data is returned

This prevents the application from terminating because of a single failed resume.

---

## Backend Connection Failure

The Streamlit frontend catches connection failures and displays an error message if the FastAPI backend is unavailable.

---

# Security Considerations

The current implementation includes several basic security measures.

## Implemented

* MIME type validation before parsing
* PII masking for email addresses
* PII masking for phone numbers
* Environment variable configuration using `.env`

## Not Implemented

The repository currently does **not** include:

* user authentication
* authorization
* rate limiting
* request throttling
* HTTPS configuration
* audit logging
* API key authentication for backend endpoints

These would be required before deploying the application to a public environment.

---

# Performance Considerations

Current implementation characteristics:

* resumes are processed sequentially
* embeddings are generated individually
* no embedding cache
* no response cache
* no asynchronous batch processing
* no background task queue

For small batches this is acceptable, but processing time will increase linearly as more resumes are uploaded.

---

# Troubleshooting

## FastAPI Not Running

Symptom:

```text
Connection refused
```

Solution:

```bash
uvicorn src.api.main:app --reload
```

---

## Invalid Gemini API Key

Verify:

```text
GEMINI_API_KEY
```

exists in:

```text
.env
```

---

## PDF Cannot Be Parsed

Possible causes:

* scanned PDF
* encrypted PDF
* corrupted document

Only PDFs containing selectable text are supported.

---

## Unsupported File Type

Only the following formats are accepted:

* `.pdf`
* `.docx`

Other document formats are ignored.

---

## Missing libmagic

If MIME detection fails, install the native library for your operating system before running the application.

---

# Known Limitations

The current implementation has the following limitations:

* Resume processing is sequential.
* Experience scoring uses a fixed five-year benchmark.
* Structured extraction depends on Google Gemini responses.
* No OCR support for scanned PDFs.
* No persistent storage.
* No authentication or authorization.
* No automated test suite.
* No CI/CD pipeline.
* No API versioning beyond the current implementation.
* SQLAlchemy is included as a dependency but is not currently used by the application.

These are potential areas for future development rather than implemented functionality.
## API

### Endpoint

`POST /analyze`

Analyzes one or more resumes against a supplied job description.

### Request

**Content-Type**

multipart/form-data

| Field | Type | Required | Description |
|------|------|----------|-------------|
| jd | string | Yes | Job description text |
| files | File[] | Yes | One or more PDF or DOCX resumes |

### Response

```json
[
  {
    "name": "candidate_resume.pdf",
    "score": 82.3,
    "details": {
      "overall_score": 82.3,
      "semantic_match": 85.0,
      "experience_alignment": 80.0,
      "reasoning": "Candidate shows 4 years of experience. Semantic match is 85.0%."
    },
    "summary": "Full-stack developer with experience in Python and FastAPI."
  }
]
```

### Processing Flow

For each uploaded resume the backend performs the following operations:

1. Validate uploaded document type.
2. Extract document text.
3. Mask email addresses and phone numbers.
4. Generate semantic embedding.
5. Request structured candidate analysis from Gemini.
6. Compute semantic similarity.
7. Calculate final weighted score.
8. Return ranked results.

---

# Logging

The project uses Python's built-in logging module.

Current logging includes:

- Parser failures
- AI extraction failures
- General application events

Logs are written to standard output.

---

# Error Handling

Current implementation includes handling for:

- Unsupported document types
- Parsing failures
- Invalid AI responses
- Backend connection failures
- Missing extracted text

When AI extraction fails, fallback values are returned instead of terminating the request.

---

# Security

Current security measures include:

- Environment variable configuration
- PII masking before LLM requests
- MIME type validation
- Separation of backend and frontend services

Not currently implemented:

- Authentication
- Authorization
- HTTPS configuration
- Rate limiting
- Request size limits
- Audit logging

---

# Performance Characteristics

Current implementation:

- Sequential resume processing
- In-memory embedding generation
- Stateless API
- No persistent storage

Potential bottlenecks:

- LLM response latency
- Embedding generation
- Large document parsing

---

# Known Limitations

Current limitations include:

- Text-based PDF files only (OCR is not implemented)
- DOCX and PDF are the only supported formats
- Resume processing is sequential
- Experience scoring uses a fixed five-year benchmark
- No database persistence
- No authentication
- No caching
- AI output depends on Gemini response quality
- JSON parsing relies on model output formatting

---

# Future Improvements

Potential enhancements include:

- Parallel resume processing
- Batch embedding generation
- Structured Gemini responses using Pydantic validation
- OCR support for scanned resumes
- Persistent storage
- User authentication
- REST API versioning
- Automated testing
- CI/CD pipeline
- Request validation middleware
- Monitoring and metrics
- Retry strategy for AI requests
- Background task processing
- Container separation for frontend and backend
- API documentation improvements

---

# Testing

The repository currently does not contain an automated test suite.

Recommended additions:

Unit tests

- Parser
- Scoring engine
- PII scrubber
- Configuration

Integration tests

- API endpoint
- Resume upload
- Gemini integration
- Streamlit to FastAPI communication

End-to-end tests

- Upload multiple resumes
- Rank candidates
- Error scenarios

---

# Repository Structure

```
.
├── src/
│   ├── api/
│   ├── core/
│   ├── infra/
│   ├── services/
│   └── utils/
├── .devcontainer/
├── .streamlit/
├── docker-compose.yml
├── requirements.txt
├── streamlit_app.py
└── README.md
```

---

# Contributing

Contributions are welcome.

When contributing:

- Follow the existing project structure.
- Keep business logic inside the services package.
- Keep infrastructure-specific code inside infra.
- Update documentation when functionality changes.
- Test changes manually before submitting a pull request.

---

# License

This project is licensed under the MIT License.

If a LICENSE file is added to the repository, this section should reference it.

---

# Acknowledgements

This project uses several open-source libraries:

- FastAPI
- Streamlit
- Google Generative AI SDK
- Sentence Transformers
- PyMuPDF
- python-docx
- NumPy
- Pydantic
- python-magic

Please refer to each project's license for additional information.
