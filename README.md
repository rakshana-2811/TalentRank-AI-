# AI Resume Screener

An easy-to-use, recruiter-friendly tool that ranks PDF resumes against a
job description using OpenAI embeddings and provides a short AI-generated
explanation for why each candidate matches (or doesn't).

This project is built to be approachable for hiring teams and engineers:
- Minimal setup — run a CLI or a Streamlit app
- Clear outputs — ranked candidates, similarity scores, and concise explanations
- Modular code — easy to extend or integrate into hiring workflows

## Features

- Extracts text from PDF resumes using `pypdf`.
- Generates semantic embeddings with OpenAI model `text-embedding-3-small`.
- Computes cosine similarity using `numpy` and ranks candidates.
- Produces short professional explanations with `gpt-4o-mini` for each candidate.
- Comes with both a CLI (`main.py`) and a Streamlit UI (`app.py`).

## Tech Stack

- Python 3.10+
- OpenAI Python SDK (`openai`) for embeddings and explanations
- `pypdf` for PDF text extraction
- `numpy` for cosine similarity computations
- `streamlit` + `pandas` for the interactive app
- `python-dotenv` for managing the `OPENAI_API_KEY`

## Setup Instructions

1. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
# or on macOS / Linux: source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your OpenAI key:

```
OPENAI_API_KEY=sk-...
```

4. Run the CLI (quick):

```bash
python main.py --resumes-dir resumes --job "Senior Python developer with ML experience"
```

5. Or run the Streamlit app (interactive):

```bash
streamlit run app.py
```

## Example Output

CLI output (example):

```
Ranked candidates:
 1. jane_doe.pdf: 0.8721
 2. john_smith.pdf: 0.7453
 3. intern_amy.pdf: 0.4120
```

Explanation for `jane_doe.pdf` (3 lines):

- Strong match on required skills: 6+ years Python, ML pipelines, and SQL.
- Demonstrated product impact with deployed ML models in production.
- Minor gap: limited experience with cloud infra automation (Terraform).

The Streamlit UI displays the same ranking in a table and shows the
AI-generated explanation directly beneath each candidate, keeping decision
makers focused on key evidence.

## Folder Structure

```
ResumeAI/
├─ app.py                      # Streamlit single-file app
├─ main.py                     # Simple CLI runner
├─ requirements.txt
├─ .env.example
├─ README.md
├─ resume_screener/
│  ├─ __init__.py
│  ├─ extractor.py             # PDF text extraction helpers
│  ├─ embeddings.py            # OpenAI embeddings + similarity helper
│  ├─ similarity.py            # cosine similarity & ranking
│  ├─ explainer.py             # GPT-based match explanation
└─ resumes/                    # Drop PDF resumes here for CLI
```



