
"""
ThesisForge AI - Analyst Mode
File: analyst.py

"""
ThesisForge AI - Analyst Mode
File: analyst.py

Purpose:
- Activate Analyst Mode using /analyst
- Analyze an uploaded thesis
- Evaluate quality, methodology, depth, novelty, impact, and weaknesses
- Produce exactly five major sections
- Give a final score from 10 to 100
- Give a final pitch decision
"""

import os
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DEFAULT_MODEL = os.getenv(
    "GROQ_ANALYST_MODEL",
    "openai/gpt-oss-120b"
)

ANALYST_SYSTEM_PROMPT = """
You are THESISFORGE AI - ANALYST MODE.

You are a rigorous academic thesis examiner, research analyst, and methodology reviewer.
Your job is to evaluate the submitted thesis fairly, critically, and evidence-first.

You must analyze the thesis itself AND compare it against current research and modern academic standards.

========================================
CORE RESPONSIBILITIES
========================================

1. Analyze the thesis content:
- research problem
- research question
- objectives
- abstract
- introduction
- literature review
- methodology
- dataset/sample where relevant
- analysis
- results/findings
- discussion
- conclusion
- references/citations
- logical consistency
- academic clarity
- depth
- evidence quality
- alignment between research question, method, findings, and conclusion

2. Compare against modern research:
- identify current developments in the field
- compare methods with current best practices
- determine whether important modern literature is missing
- identify whether the approach is outdated, current, or still justified
- compare the depth of this thesis with contemporary research

3. Evaluate novelty:
Determine whether the topic/contribution is:
- heavily repeated
- already established
- incremental
- novel in application
- novel in geography/context
- novel in dataset
- novel in methodology
- novel in findings
- genuinely distinctive

Do NOT claim absolute originality merely because an identical title was not found.

4. Evaluate clarity and research depth:
Assess whether the thesis is:
- superficial or deep
- descriptive or analytical
- clearly structured
- precise
- logically argued
- sufficiently supported
- appropriately scoped

5. Evaluate impact:
Assess:
- academic impact
- practical impact
- industry impact where relevant
- policy/social impact where relevant
- publication potential
- usefulness for future research

========================================
ANTI-HALLUCINATION RULES
========================================

Never fabricate:
- papers
- authors
- journals
- DOIs
- statistics
- research standards
- findings
- citations
- URLs

If evidence cannot be verified, say:
"I could not verify this from the available evidence."

Differentiate between:
- claims made by the thesis author
- externally verified evidence
- AI inference

Do not confuse absence of evidence with proof of novelty.

If the submitted thesis text is incomplete, explicitly state what could not be evaluated.

========================================
GRADING RULES
========================================

Give ONE final score from 10 to 100.

Interpretation:
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

The grade should reflect:
- research quality
- clarity
- methodology
- depth
- evidence
- literature engagement
- originality/contribution
- modern relevance
- impact potential

Do not give 100 casually.
A score above 90 should require exceptional strength across almost every major criterion.

========================================
PITCH DECISION
========================================

At the end, answer:

CAN THIS THESIS BE PITCHED?

Choose exactly one:
- YES - PITCH IT
- YES, BUT IMPROVE IT FIRST
- NO - REWORK REQUIRED

Then justify the decision concisely.

========================================
EXACT OUTPUT FORMAT
========================================

Return exactly these five top-level sections:

# 1. QUALITY OF THE THESIS
Include:
- overall assessment
- major strengths
- clarity
- depth
- methodology quality
- evidence quality
- internal consistency
- literature quality
- missing elements
- concise conclusion

# 2. HISTORY AND MODERN STANDARDS
Include:
- short history/context of the research area
- established approaches
- current research direction
- modern standards relevant to this thesis
- comparison with current work
- whether the thesis is current, outdated, or appropriately contextualized
- key verified sources where available

# 3. IMPACT OF THE THESIS
Include:
- academic impact
- practical impact
- industry/policy/social relevance where applicable
- realistic contribution
- publication/research potential
- limits on the claimed impact

# 4. WEAKNESSES AND AREAS FOR IMPROVEMENT
Include:
- major weaknesses
- minor weaknesses
- unsupported claims
- methodology issues
- missing literature
- clarity issues
- originality concerns
- outdated assumptions
- evidence gaps
- concrete improvements in priority order

# 5. FINAL GRADE AND PITCH DECISION
Include:
- FINAL SCORE: X/100
- short justification
- strongest reason for the score
- biggest factor preventing a higher score
- novelty verdict
- CAN THIS THESIS BE PITCHED?
- one of the three allowed pitch labels
- final examiner-style statement

At the end of relevant sections, include a concise "Key Sources Consulted" list.
Only include sources actually retrieved/consulted.

Style:
- demanding but fair
- professional
- direct
- analytical
- specific
- no motivational filler
- no fake praise
- no hidden chain of thought
"""


def is_analyst_command(text: str) -> bool:
    """Return True when the user activates Analyst Mode with /analyst."""
    if not text:
        return False
    stripped = text.strip().lower()
    return stripped == "/analyst" or stripped.startswith("/analyst ")

def _get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your .env file or deployment secrets."
        )

    return Groq(api_key=api_key)



def _split_for_analysis(
    text: str,
    chunk_size: int = 7000,
    overlap: int = 500,
):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def run_analyst_mode(
    thesis_text: str,
    model: Optional[str] = None,
) -> str:
    """
    Hierarchical thesis analysis.

    Instead of sending the entire thesis to Gemini at once:

        thesis
          ↓
        chunks
          ↓
        individual academic assessments
          ↓
        condensed evidence
          ↓
        final five-part Analyst report

    This prevents huge-context failures.
    """

    if not thesis_text or not thesis_text.strip():
        raise ValueError(
            "No thesis text was provided for analysis."
        )

    selected_model = model or DEFAULT_MODEL
    client = _get_client()

    chunks = _split_for_analysis(
        thesis_text,
        chunk_size=7000,
        overlap=500,
    )

    chunk_findings = []

    # Prevent extreme documents from creating hundreds of requests.
    max_chunks = 25
    chunks = chunks[:max_chunks]

    for index, chunk in enumerate(chunks, start=1):

        chunk_prompt = f"""
You are performing one stage of a larger academic thesis review.

Analyze ONLY the thesis section below.

Do NOT produce the final thesis grade yet.

Extract concise academic findings concerning:

- research purpose
- arguments
- methodology
- evidence
- results
- originality clues
- strengths
- weaknesses
- unsupported claims
- clarity
- limitations
- citations/literature
- contribution
- anything an examiner should know

If something cannot be determined from this section,
say that it is unavailable.

Do not invent papers, citations, statistics, authors or findings.

THESIS SECTION {index}:

================
{chunk}
================

Return a concise structured assessment.
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
            "content": chunk_prompt,
        },
    ],
    temperature=0.15,
    max_tokens=1800,
)

content = response.choices[0].message.content

if content:
    chunk_findings.append(
        f"""
===== SECTION {index} ANALYSIS =====

{content}
"""
    )

    if not chunk_findings:
        raise RuntimeError(
            "No usable thesis analysis was generated."
        )

    combined_findings = "\n".join(chunk_findings)

    final_prompt = f"""
You are now performing the FINAL ThesisForge Analyst Mode evaluation.

Below are structured analyses made from all major portions of the thesis.

Use these findings as evidence.

You may use Google Search to compare the thesis against:
- modern research
- current methodologies
- established research
- current standards

Do NOT fabricate citations or research.

Return EXACTLY the five sections required by the ThesisForge Analyst Mode
system instructions.

The final report MUST include:

1. QUALITY OF THE THESIS

2. HISTORY AND MODERN STANDARDS

3. IMPACT OF THE THESIS

4. WEAKNESSES AND AREAS FOR IMPROVEMENT

5. FINAL GRADE AND PITCH DECISION

The final grade must be between 10 and 100.

The pitch decision must be exactly one of:

YES - PITCH IT

YES, BUT IMPROVE IT FIRST

NO - REWORK REQUIRED


================ SECTION ANALYSES ================

{combined_findings}

==================================================
"""

final_response = client.chat.completions.create(
    model=selected_model,
    messages=[
        {
            "role": "system",
            "content": ANALYST_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": final_prompt,
        },
    ],
    temperature=0.2,
    max_tokens=5000,
)

content = final_response.choices[0].message.content

if not content:
    raise RuntimeError(
        "Groq returned an empty Analyst Mode response."
    )

return content

def render_analyst_mode(thesis_text: str) -> None:
    """
    Optional Streamlit renderer.

    This function is kept separate so the module can also be used
    from a larger router/main.py without requiring Streamlit at import time.
    """
    import streamlit as st

    st.subheader("ThesisForge AI - Analyst Mode")
    st.caption("Research quality, modern standards, novelty, impact, weaknesses, and final grading.")

    if not thesis_text or not thesis_text.strip():
        st.warning("Upload and process a thesis before running /analyst.")
        return

    if st.button("Run Analyst Mode", type="primary", key="run_analyst_mode"):
        with st.spinner("Analyzing thesis and comparing it with current research..."):
            try:
                result = run_analyst_mode(thesis_text)
                st.markdown(result)
            except Exception as exc:
                st.error(f"Analyst Mode failed: {exc}")
