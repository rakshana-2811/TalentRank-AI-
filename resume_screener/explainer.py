from dotenv import load_dotenv
import os
from openai import OpenAI
from typing import Optional


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def _rule_based_explanation(resume_text: str, job_description: str) -> str:
    """Generate a simple rule-based explanation without API calls.
    
    Matches keywords and extracts basic insights for when GPT is unavailable.
    """
    job_lower = job_description.lower()
    resume_lower = resume_text.lower()
    
    # Common job keywords
    keywords = {
        "python": ["python", "py"],
        "machine learning": ["machine learning", "ml", "deep learning", "neural"],
        "data": ["data", "analytics", "analysis"],
        "backend": ["backend", "api", "server"],
        "frontend": ["frontend", "react", "vue", "typescript", "javascript"],
        "sql": ["sql", "database", "postgres", "mysql"],
        "aws": ["aws", "cloud", "azure", "gcp"],
        "leadership": ["led", "managed", "director", "manager", "team lead"],
    }
    
    # Extract job requirements
    job_skills = set()
    for skill, terms in keywords.items():
        if any(term in job_lower for term in terms):
            job_skills.add(skill)
    
    # Check resume coverage
    matched = 0
    if job_skills:
        for skill, terms in keywords.items():
            if skill in job_skills and any(term in resume_lower for term in terms):
                matched += 1
    
    coverage = matched / len(job_skills) if job_skills else 0
    
    # Simple explanation based on coverage
    if coverage > 0.75:
        return f"Strong match. Resume aligns well with {matched}/{len(job_skills)} key job requirements. Good technical fit."
    elif coverage > 0.5:
        return f"Moderate match. Resume covers {matched}/{len(job_skills)} key requirements. Could be a viable candidate with minor gaps."
    elif coverage > 0.25:
        return f"Partial match. Only {matched}/{len(job_skills)} key requirements found. Significant skill gaps present."
    else:
        return f"Limited match. Minimal alignment with job requirements ({matched} skills). May need additional training."


def generate_match_explanation(resume_text: str, job_description: str, *,
                               model: str = "gpt-4o-mini", max_tokens: int = 150,
                               temperature: float = 0.2) -> str:
    """Return a short (3-4 lines) professional explanation of match quality.

    Uses GPT-4o-mini if OPENAI_API_KEY is set; falls back to rule-based
    explanation (free, no API calls) otherwise.
    """
    # Try GPT-based explanation first if API key is available
    if OPENAI_API_KEY and _client:
        try:
            system_msg = (
                "You are an objective, professional hiring analyst. "
                "Provide a short, professional explanation (3-4 lines) summarizing why "
                "the candidate is a good match or not for the role. Focus on 2-3 key points: "
                "relevant skills, experience gaps, and standout strengths. Be concise and neutral."
            )

            user_msg = (
                "Job description:\n" + job_description + "\n\n"
                "Candidate resume:\n" + (resume_text or "(no text)") + "\n\n"
                "Respond with a brief 3-4 line explanation."
            )

            resp = _client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_msg},
                          {"role": "user", "content": user_msg}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = resp["choices"][0]["message"]["content"].strip()
            lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
            return "\n".join(lines[:4])
        except Exception:
            # Fall through to rule-based if API fails
            pass
    
    # Use rule-based explanation (always works, no API)
    return _rule_based_explanation(resume_text, job_description)
