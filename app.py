import streamlit as st
from io import BytesIO
from typing import List
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="AI Resume Screener", layout="centered")

from resume_screener import embeddings, similarity, explainer
from pypdf import PdfReader


def extract_text_from_bytes(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using pypdf and return as a string."""
    text_parts: List[str] = []
    try:
        reader = PdfReader(BytesIO(file_bytes))
        for page in reader.pages:
            ptext = page.extract_text()
            if ptext:
                text_parts.append(ptext)
    except Exception:
        return ""
    return "\n".join(text_parts)


def rank_and_explain(resumes: List[dict], job_description: str):
    """Compute embeddings, rank resumes, and return results with explanations.

    `resumes` is a list of {'filename': str, 'bytes': bytes, 'text': str}.
    Returns a list of dicts: filename, score, explanation.
    """
    # Ensure either OpenAI key or local embeddings is configured
    use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "").lower() == "true"
    if not os.getenv("OPENAI_API_KEY") and not use_local:
        st.error("Neither OPENAI_API_KEY nor USE_LOCAL_EMBEDDINGS is configured. Check your .env file.")
        return []

    texts = [r["text"] for r in resumes]

    try:
        resume_embs = embeddings.embed_texts(texts)
        job_emb = embeddings.get_embedding(job_description)
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return []

    scored = []
    for i, r in enumerate(resumes):
        score = float(similarity.cosine_similarity(job_emb, resume_embs[i]))
        scored.append({"filename": r["filename"], "text": r["text"], "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)

    # Generate explanations for each candidate (concise)
    for item in scored:
        try:
            item["explanation"] = explainer.generate_match_explanation(item["text"], job_description)
        except Exception:
            item["explanation"] = "(Explanation unavailable)"

    return scored


def main():
    st.title("AI Resume Screener")

    uploaded = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

    job_description = st.text_area("Job description", height=160)

    if st.button("Run Screener"):
        if not uploaded:
            st.warning("Please upload at least one PDF resume.")
            return
        if not job_description or not job_description.strip():
            st.warning("Please provide a job description.")
            return

        resumes = []
        for f in uploaded:
            try:
                raw = f.read()
                text = extract_text_from_bytes(raw)
            except Exception:
                raw = b""
                text = ""
            resumes.append({"filename": f.name, "bytes": raw, "text": text})

        with st.spinner("Scoring resumes..."):
            results = rank_and_explain(resumes, job_description)

        if not results:
            st.info("No results to display.")
            return

        # Display results table
        st.subheader("Ranked results")
        rows = [{"Rank": idx + 1, "Filename": r["filename"], "Score": round(r["score"], 4)}
                for idx, r in enumerate(results)]
        st.table(rows)

        # Show explanation for each candidate
        st.subheader("Explanations")
        for idx, r in enumerate(results, start=1):
            st.markdown(f"**{idx}. {r['filename']} — {r['score']:.4f}**")
            st.write(r.get("explanation", "(No explanation)"))
            st.markdown("---")


if __name__ == "__main__":
    main()
