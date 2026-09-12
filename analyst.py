"""
ThesisForge AI - Analyst Mode
File: analyst.py

Purpose:
- Activate Analyst Mode using /analyst
- Analyze an uploaded thesis/research paper
- Use local TF-IDF retrieval to select representative thesis evidence
- Avoid oversized API requests
- Evaluate quality, methodology, clarity, depth, novelty, impact, and weaknesses
- Return exactly five top-level sections
- Give a final grade from 10 to 100
- Give a final pitch decision

Dependencies:
    pip install groq python-dotenv scikit-learn

Environment variable / Streamlit secret:
    GROQ_API_KEY=your_groq_key

Expected integration:
    from analyst import is_analyst_command, run_analyst_mode
"""

import os
from typing import List, Optional

from dotenv import load_dotenv
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

DEFAULT_MODEL = os.getenv(
    "GROQ_ANALYST_MODEL",
    "openai/gpt-oss-120b",
)

ANALYST_SYSTEM_PROMPT = """
You are THESISFORGE AI - ANALYST MODE.

You are a rigorous but fair academic thesis examiner.

Your task is to evaluate a thesis using the thesis evidence supplied by the
application. Do not claim that you performed live web research or verified
external literature unless such evidence is explicitly included in the prompt.

==================================================
CORE ANALYSIS
==================================================

Evaluate, where evidence is available:

- research problem
- research questions
- objectives
- abstract
- introduction
- literature review
- methodology
- research design
- data or sample
- experiments or procedures
- analysis
- results
- discussion
- conclusion
- limitations
- references/citations
- clarity
- academic depth
- logical consistency
- evidence quality
- alignment between objectives, methodology, results, and conclusion
- contribution
- originality/novelty
- practical and academic impact

If a required thesis element is absent from the supplied evidence, explicitly
say that it could not be verified.

==================================================
ANTI-HALLUCINATION RULES
==================================================

Never fabricate:

- papers
- authors
- journals
- DOIs
- URLs
- statistics
- research findings
- citations
- academic standards
- experimental results

Never claim absolute novelty simply because an identical title is not visible.

When evidence is missing, say:

"I could not verify this from the available thesis evidence."

Clearly distinguish between:

- claims stated by the thesis
- conclusions that follow from the supplied thesis evidence
- general academic evaluation

==================================================
NOVELTY EVALUATION
==================================================

Use appropriate descriptions such as:

- heavily repeated
- well established
- incrementally different
- novel application/context
- novel geographical application
- novel methodology
- novel dataset/evidence
- potentially novel findings
- genuinely distinctive

A common research topic can still contain a novel method, dataset, population,
application, geography, or finding.

==================================================
GRADING
==================================================

Give exactly one final score from 10 to 100.

Use this scale:

10 = fundamentally deficient
20 = very poor
30 = poor
40 = below acceptable
50 = borderline
60 = acceptable
70 = good
80 = very good
90 = excellent
100 = exceptionally strong

Do not give scores above 90 casually.

The score should reflect:

- research quality
- methodology
- clarity
- depth
- evidence
- literature engagement
- logical consistency
- originality/contribution
- impact potential
- overall readiness

==================================================
PITCH DECISION
==================================================

Choose exactly one:

- YES - PITCH IT
- YES, BUT IMPROVE IT FIRST
- NO - REWORK REQUIRED

==================================================
EXACT OUTPUT FORMAT
==================================================

Return EXACTLY these five top-level sections:

# 1. QUALITY OF THE THESIS

Include:
- overall assessment
- strongest aspects
- clarity
- depth
- methodology quality
- evidence quality
- logical consistency
- literature quality where visible
- missing elements
- concise conclusion

# 2. HISTORY AND MODERN STANDARDS

Include:
- research-area context visible from the thesis
- established approaches mentioned in the thesis
- whether the methods appear academically appropriate
- whether anything appears weak, dated, or insufficiently justified
- comparison against normal modern academic expectations
- a clear statement that live external literature was not independently
  verified unless the application supplied such evidence

# 3. IMPACT OF THE THESIS

Include:
- academic impact
- practical impact
- industry/policy/social relevance where applicable
- realistic contribution
- publication/research potential
- limitations on claimed impact

# 4. WEAKNESSES AND AREAS FOR IMPROVEMENT

Include:
- major weaknesses
- minor weaknesses
- unsupported claims
- methodology concerns
- missing literature where apparent
- clarity issues
- originality concerns
- evidence gaps
- prioritized concrete improvements

# 5. FINAL GRADE AND PITCH DECISION

Include:
- FINAL SCORE: X/100
- concise justification
- strongest reason for the score
- biggest factor preventing a higher score
- novelty verdict
- CAN THIS THESIS BE PITCHED?
- exactly one allowed pitch decision
- final examiner-style statement

Style:
- demanding but fair
- academic
- concise
- specific
- direct
- no fake praise
- no hidden chain-of-thought
"""


def is_analyst_command(text: str) -> bool:
    """Return True when the user activates Analyst Mode with /analyst."""
    if not text:
        return False

    stripped = text.strip().lower()
    return stripped == "/analyst" or stripped.startswith("/analyst ")


def _get_client() -> Groq:
    """Create the Groq client from the configured API key."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your .env file "
            "or Streamlit secrets."
        )

    return Groq(api_key=api_key)


def _split_thesis(
    text: str,
    chunk_size: int = 3500,
    overlap: int = 300,
) -> List[str]:
    """Split thesis text into overlapping character chunks."""
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


def _retrieve_chunks(
    chunks: List[str],
    query: str,
    top_k: int = 2,
) -> List[str]:
    """
    Retrieve the most relevant thesis chunks using TF-IDF cosine similarity.
    """
    if not chunks:
        return []

    if len(chunks) <= top_k:
        return chunks

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=12000,
    )

    matrix = vectorizer.fit_transform(chunks + [query])

    chunk_vectors = matrix[:-1]
    query_vector = matrix[-1]

    similarities = cosine_similarity(
        query_vector,
        chunk_vectors,
    ).flatten()

    indexes = similarities.argsort()[::-1][:top_k]

    return [chunks[index] for index in indexes]


def _build_analysis_context(thesis_text: str) -> str:
    """
    Build a bounded, representative thesis context.

    The function combines:
    - thesis opening
    - thesis ending
    - research-problem/objective excerpts
    - methodology excerpts
    - results/discussion excerpts
    - literature/novelty excerpts

    This keeps the Groq request small enough for free-tier limits while still
    covering the areas needed for an academic thesis-level assessment.
    """
    chunks = _split_thesis(thesis_text)

    if not chunks:
        return ""

    selected: List[str] = []

    # Preserve the start of the document.
    selected.extend(chunks[:2])

    # Preserve the end of the document.
    if len(chunks) > 2:
        selected.extend(chunks[-2:])

    retrieval_queries = [
        (
            "abstract introduction research problem research questions "
            "objectives motivation scope contribution",
            2,
        ),
        (
            "methodology methods research design dataset sample participants "
            "data collection experiment procedure statistical analysis",
            2,
        ),
        (
            "results findings discussion evaluation conclusion limitations "
            "future work",
            2,
        ),
        (
            "literature review related work prior research novelty originality "
            "references contribution",
            2,
        ),
    ]

    for query, top_k in retrieval_queries:
        selected.extend(
            _retrieve_chunks(
                chunks=chunks,
                query=query,
                top_k=top_k,
            )
        )

    # Remove duplicate chunks while preserving order.
    unique_chunks: List[str] = []
    seen = set()

    for chunk in selected:
        fingerprint = chunk[:250]

        if fingerprint not in seen:
            unique_chunks.append(chunk)
            seen.add(fingerprint)

    # Keep the final request comfortably below Groq free-tier TPM limits.
    max_context_chars = 16000
    context_parts: List[str] = []
    used_chars = 0

    for index, chunk in enumerate(unique_chunks, start=1):
        labeled_chunk = (
            f"\n\n===== THESIS EXCERPT {index} =====\n"
            f"{chunk}"
        )

        if used_chars + len(labeled_chunk) > max_context_chars:
            break

        context_parts.append(labeled_chunk)
        used_chars += len(labeled_chunk)

    return "".join(context_parts).strip()


def run_analyst_mode(
    thesis_text: str,
    model: Optional[str] = None,
) -> str:
    """
    Run ThesisForge Analyst Mode.

    The thesis is first split and locally retrieved with TF-IDF.
    Only a bounded representative context is sent to Groq.
    """
    if not thesis_text or not thesis_text.strip():
        raise ValueError(
            "No thesis text was provided for analysis."
        )

    selected_model = model or DEFAULT_MODEL
    client = _get_client()

    analysis_context = _build_analysis_context(thesis_text)

    if not analysis_context:
        raise RuntimeError(
            "The thesis could not be converted into usable analysis context."
        )

    user_prompt = f"""
Perform the final ThesisForge Analyst Mode evaluation using the thesis
evidence below.

Important:
- These excerpts were selected from across the uploaded thesis using local
  TF-IDF retrieval.
- Do not assume information that is not present.
- Do not fabricate external research or citations.
- Do not claim live web verification.
- Follow the exact five-section structure from the system prompt.
- Give one final score from 10 to 100.
- End with exactly one allowed pitch decision.

================ THESIS EVIDENCE ================

{analysis_context}

=================================================
"""

    response = client.chat.completions.create(
        model=selected_model,
        messages=[
            {
                "role": "system",
                "content": ANALYST_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
        max_tokens=2500,
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "Groq returned an empty Analyst Mode response."
        )

    return content.strip()


def render_analyst_mode(thesis_text: str) -> None:
    """
    Optional standalone Streamlit renderer.

    The main ThesisForge application can instead call run_analyst_mode()
    directly when /analyst is entered.
    """
    import streamlit as st

    st.subheader("ThesisForge AI - Analyst Mode")
    st.caption(
        "Quality, methodology, novelty, impact, weaknesses, "
        "and final grading."
    )

    if not thesis_text or not thesis_text.strip():
        st.warning(
            "Upload and process a thesis before running /analyst."
        )
        return

    if st.button(
        "Run Analyst Mode",
        type="primary",
        key="run_analyst_mode",
    ):
        with st.spinner("Analyzing thesis..."):
            try:
                result = run_analyst_mode(thesis_text)
                st.markdown(result)
            except Exception as exc:
                st.error(f"Analyst Mode failed: {exc}")
