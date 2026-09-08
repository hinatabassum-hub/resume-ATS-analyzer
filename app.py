import io
import json
import os
import re
from typing import Any, Dict, List, Optional

import streamlit as st
from docx import Document
from google import genai
from google.genai import types
from pypdf import PdfReader

MODEL_NAME = "gemini-2.5-flash"
MAX_FILE_MB = 10
MAX_TEXT_CHARS = 50000


def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract selectable text from a PDF."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def extract_docx_text(file_bytes: bytes) -> str:
    """Extract paragraphs and table text from a DOCX."""
    doc = Document(io.BytesIO(file_bytes))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]

    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))

    return "\n".join(parts).strip()


def extract_text(uploaded_file) -> str:
    """Extract text from supported resume formats."""
    data = uploaded_file.getvalue()
    suffix = uploaded_file.name.lower().rsplit(".", 1)[-1]

    if suffix == "pdf":
        return extract_pdf_text(data)
    if suffix == "docx":
        return extract_docx_text(data)
    if suffix == "txt":
        return data.decode("utf-8", errors="ignore").strip()

    raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT.")


def get_api_key() -> Optional[str]:
    """Read Gemini API key from Streamlit secrets or environment."""
    try:
        key = st.secrets.get("AQ.Ab8RN6L7r6asZVMHYLu-iI8KnrPfpW7SM3F2LjmNGpVI7k7DdQ")
        if key:
            return str(key).strip()
    except Exception:
        pass

    return os.getenv("GEMINI_API_KEY")


def local_ats_checks(resume_text: str, job_description: str = "") -> Dict[str, Any]:
    """
    Fast deterministic checks that complement the AI assessment.
    This is not a replacement for a real ATS parser.
    """
    text = resume_text.lower()
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#./-]*\b", resume_text)
    word_count = len(words)

    section_patterns = {
        "Contact information": r"(email|phone|linkedin|github)",
        "Summary / Objective": r"(professional summary|summary|objective|profile)",
        "Experience": r"(work experience|professional experience|employment|experience)",
        "Education": r"(education|academic)",
        "Skills": r"(skills|technical skills|core competencies)",
    }

    sections_found = [
        section for section, pattern in section_patterns.items()
        if re.search(pattern, text)
    ]

    bullet_lines = sum(
        1 for line in resume_text.splitlines()
        if re.match(r"^\s*[-•*▪◦]\s+", line)
    )

    action_verbs = {
        "achieved", "built", "created", "designed", "developed", "delivered",
        "improved", "increased", "implemented", "led", "managed", "optimized",
        "reduced", "automated", "launched", "analyzed", "coordinated",
    }
    action_verb_hits = sum(
        1 for word in re.findall(r"\b[a-zA-Z]+\b", text)
        if word in action_verbs
    )

    metrics = re.findall(
        r"(?:\b\d+(?:\.\d+)?\s*%|\$\s?\d+(?:[.,]\d+)*|\b\d+(?:\.\d+)?\s*(?:k|m|million|billion|years?|months?))",
        text,
        flags=re.I,
    )

    job_keywords = []
    keyword_matches = []
    if job_description.strip():
        jd_tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#./-]{2,}\b", job_description.lower())
        stopwords = {
            "the", "and", "for", "with", "from", "that", "this", "are", "you",
            "your", "our", "will", "have", "has", "job", "role", "work", "years",
            "into", "about", "they", "their", "who", "but", "not", "can", "all",
        }
        seen = set()
        for token in jd_tokens:
            if token not in stopwords and token not in seen:
                seen.add(token)
                job_keywords.append(token)

        keyword_matches = [k for k in job_keywords if k in text]

    return {
        "word_count": word_count,
        "sections_found": sections_found,
        "missing_sections": [
            s for s in section_patterns if s not in sections_found
        ],
        "bullet_lines": bullet_lines,
        "action_verb_hits": action_verb_hits,
        "metrics_found": len(metrics),
        "keyword_match_count": len(keyword_matches),
        "keyword_total": len(job_keywords),
        "matched_keywords": keyword_matches[:30],
        "filename": "",
    }


def analyze_with_gemini(
    resume_text: str,
    job_description: str,
    local_checks: Dict[str, Any],
) -> Dict[str, Any]:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "Gemini API key not found. Add GEMINI_API_KEY to Streamlit Secrets "
            "or set it as an environment variable."
        )

    client = genai.Client(api_key=api_key)

    schema = {
        "type": "object",
        "properties": {
            "ats_score": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "Overall ATS-readiness score from 0 to 100."
            },
            "score_breakdown": {
                "type": "object",
                "properties": {
                    "format_and_structure": {"type": "integer", "minimum": 0, "maximum": 100},
                    "keyword_alignment": {"type": "integer", "minimum": 0, "maximum": 100},
                    "experience_and_impact": {"type": "integer", "minimum": 0, "maximum": 100},
                    "skills_and_completeness": {"type": "integer", "minimum": 0, "maximum": 100},
                },
                "required": [
                    "format_and_structure",
                    "keyword_alignment",
                    "experience_and_impact",
                    "skills_and_completeness",
                ],
            },
            "summary": {
                "type": "string",
                "description": "Short explanation of the resume's biggest strengths and weaknesses."
            },
            "strengths": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
            },
            "improvements": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 10,
            },
            "missing_keywords": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 30,
            },
            "rewrite_examples": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 6,
            },
        },
        "required": [
            "ats_score",
            "score_breakdown",
            "summary",
            "strengths",
            "improvements",
            "missing_keywords",
            "rewrite_examples",
        ],
    }

    prompt = f"""
You are an expert resume reviewer and ATS optimization specialist.

Analyze the resume below. Give a practical ATS-readiness score from 0-100.
Important: this is an estimated ATS-readiness score, not a score produced by
a specific employer's ATS. Do not invent facts that are not in the resume.

Use these dimensions:
- format_and_structure: clear headings, parseable structure, standard sections,
  sensible length, bullets, and absence of obvious parsing problems.
- keyword_alignment: alignment with the supplied job description. If no job
  description is supplied, score based on role-relevant terminology and clarity,
  and do NOT pretend there is a keyword match calculation.
- experience_and_impact: action verbs, measurable achievements, clarity,
  relevance, and results.
- skills_and_completeness: skills, education, contact details, consistency,
  and other useful resume information.

Local deterministic checks:
{json.dumps(local_checks, indent=2)}

Job description:
{job_description.strip() if job_description.strip() else "(Not provided)"}

Resume:
--- BEGIN RESUME ---
{resume_text[:MAX_TEXT_CHARS]}
--- END RESUME ---

Return only the requested JSON structure.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    try:
        result = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Gemini returned invalid JSON.") from exc

    # Clamp/normalize the score defensively.
    result["ats_score"] = max(0, min(100, int(result.get("ats_score", 0))))
    return result


def render_score(score: int) -> None:
    label = (
        "Excellent ATS readiness" if score >= 85 else
        "Good ATS readiness" if score >= 70 else
        "Needs improvement" if score >= 50 else
        "Major improvements recommended"
    )

    st.metric("Estimated ATS Score", f"{score}/100")
    st.progress(score / 100)
    st.caption(label)


st.set_page_config(
    page_title="AI Resume ATS Checker",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AI Resume ATS Checker")
st.write(
    "Upload a resume and get an estimated ATS score, strengths, missing keywords, "
    "and practical improvements powered by Gemini 2.5 Flash."
)

with st.sidebar:
    st.header("Settings")
    st.info(
        "For a more useful keyword score, paste the job description. "
        "Without a job description, keyword alignment is estimated from the resume itself."
    )
    st.caption(f"Gemini model: `{MODEL_NAME}`")
    st.caption(f"Maximum upload size: {MAX_FILE_MB} MB")

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf", "docx", "txt"],
    help="Supported formats: PDF, DOCX, TXT.",
)

job_description = st.text_area(
    "Job description (optional, recommended)",
    height=180,
    placeholder="Paste the job description here to get keyword matching and missing-keyword suggestions.",
)

if uploaded_file:
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        st.error(f"File is too large ({size_mb:.1f} MB). Please upload a file under {MAX_FILE_MB} MB.")
        st.stop()

    if st.button("🔍 Analyze Resume", type="primary", use_container_width=True):
        try:
            with st.spinner("Reading your resume..."):
                resume_text = extract_text(uploaded_file)

            if len(resume_text.strip()) < 100:
                st.error(
                    "Very little text could be extracted. If this is a scanned/image-only PDF, "
                    "please use a text-based PDF or DOCX version."
                )
                st.stop()

            if len(resume_text) > MAX_TEXT_CHARS:
                resume_text = resume_text[:MAX_TEXT_CHARS]

            local_checks = local_ats_checks(resume_text, job_description)
            local_checks["filename"] = uploaded_file.name

            with st.spinner("Gemini is analyzing your resume..."):
                result = analyze_with_gemini(
                    resume_text,
                    job_description,
                    local_checks,
                )

            st.session_state["analysis"] = result
            st.session_state["local_checks"] = local_checks
            st.session_state["resume_text"] = resume_text

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.info(
                "Check that your Gemini API key is configured and that the uploaded "
                "resume contains selectable text."
            )

if "analysis" in st.session_state:
    result = st.session_state["analysis"]
    checks = st.session_state["local_checks"]

    st.divider()

    col1, col2 = st.columns([1, 2])
    with col1:
        render_score(result["ats_score"])
    with col2:
        st.subheader("Quick technical checks")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Words", checks["word_count"])
        c2.metric("Sections", len(checks["sections_found"]))
        c3.metric("Bullets", checks["bullet_lines"])
        c4.metric("Metrics", checks["metrics_found"])

    st.subheader("🧾 Score Breakdown")
    breakdown = result["score_breakdown"]
    cols = st.columns(4)
    labels = [
        ("Format & Structure", "format_and_structure"),
        ("Keyword Alignment", "keyword_alignment"),
        ("Experience & Impact", "experience_and_impact"),
        ("Skills & Completeness", "skills_and_completeness"),
    ]
    for col, (label, key) in zip(cols, labels):
        with col:
            st.metric(label, f"{breakdown[key]}/100")

    st.subheader("💡 Overall Assessment")
    st.write(result["summary"])

    left, right = st.columns(2)

    with left:
        st.subheader("✅ Strengths")
        for item in result["strengths"]:
            st.markdown(f"- {item}")

    with right:
        st.subheader("🚀 Improvements")
        for item in result["improvements"]:
            st.markdown(f"- {item}")

    st.subheader("🔑 Missing / Useful Keywords")
    if result["missing_keywords"]:
        st.write(", ".join(result["missing_keywords"]))
    else:
        st.success("No major missing keywords were identified from the supplied job description.")

    st.subheader("✍️ Rewrite Examples")
    for item in result["rewrite_examples"]:
        st.markdown(f"- {item}")

    if checks["missing_sections"]:
        st.subheader("📌 Sections You May Need")
        st.write(", ".join(checks["missing_sections"]))

    st.caption(
        "Note: ATS scores vary by employer, ATS vendor, job description, and resume format. "
        "Use this score as an optimization guide, not a hiring prediction."
    )
