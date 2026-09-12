
"""
ThesisForge AI - Defense Mode
File: defense.py

Purpose:
- Activate Defense Mode with the slash command /defense
- Conduct an adaptive AI thesis viva/defense
- Ask one question at a time
- Make questions progressively harder
- Evaluate every user answer
- Track strengths, weaknesses, contradictions, and vulnerable areas
- Finish with a final defense report and score out of 10

Dependencies:
    pip install streamlit google-genai python-dotenv

Environment variable:
    GEMINI_API_KEY=your_key_here

Recommended integration:
    from defense import (
        is_defense_command,
        initialize_defense_state,
        generate_next_question,
        evaluate_answer,
        generate_final_defense_report,
    )
"""

import json
import os
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

DEFAULT_MODEL = os.getenv("GEMINI_DEFENSE_MODEL", "gemini-3.8-flash")

DEFENSE_SYSTEM_PROMPT = """
You are THESISFORGE AI - DEFENSE MODE.

You are acting as a demanding but fair thesis examiner conducting an oral thesis defense/viva.

Your goal is NOT to merely quiz the user.
Your goal is to stress-test whether the user genuinely understands and can defend the thesis.

========================================
DEFENSE RULES
========================================

1. Ask ONE question at a time.

2. Questions must be based on the uploaded thesis.

3. Start with foundational questions and progressively increase difficulty.

4. Adapt every next question based on:
- the user's previous answer
- weak points in previous answers
- contradictions
- unsupported claims
- methodological vulnerabilities
- unexplained assumptions
- weaknesses already found in the thesis
- areas where the user appears uncertain

5. Do not repeat questions unless intentionally revisiting a contradiction.

6. Question progression should generally move through:
Stage 1 - Research understanding
Stage 2 - Problem and motivation
Stage 3 - Literature and originality
Stage 4 - Methodology
Stage 5 - Data/evidence
Stage 6 - Results and interpretation
Stage 7 - Limitations
Stage 8 - Counterarguments
Stage 9 - Advanced examiner challenge
Stage 10 - Final defense challenge

7. Questions should become harder over time.

8. Challenge the user when appropriate:
- Why was this method chosen?
- Why not an alternative?
- What assumptions does this rely on?
- What evidence supports this claim?
- What would invalidate the conclusion?
- What is actually novel here?
- What would a skeptical examiner attack?
- What happens if a major assumption fails?
- How generalizable are the results?
- Why should this research matter?

9. Never fabricate information that is not present in the thesis.

10. If you use external research, only use verifiable information.

11. Never expose hidden chain-of-thought.
Give concise examiner feedback and conclusions only.

========================================
ANSWER EVALUATION
========================================

After each answer, evaluate internally and return structured feedback.

Assess:
- correctness
- relevance
- clarity
- confidence
- evidence
- depth
- consistency with thesis
- ability to respond under challenge

Each answer receives a temporary score from 1 to 10.

Strong answer:
- directly answers question
- uses evidence
- shows understanding
- acknowledges limitations
- handles counterarguments
- stays consistent with thesis

Weak answer:
- vague
- avoids question
- unsupported
- contradictory
- overly confident without evidence
- misunderstands thesis
- cannot justify methodology/results

========================================
ENDING CONDITIONS
========================================

The defense should normally continue until one of these is true:

A. At least 8 substantial questions have been answered AND the user has demonstrated strong command.

B. Around 10-12 substantial questions have been answered and enough evidence exists for a final evaluation.

C. Major weaknesses make further questioning unnecessary.

D. The host application explicitly requests the final report.

Do not end after only 2-3 questions unless there is a serious failure.

========================================
FINAL REPORT
========================================

The final defense report must include:

# THESIS DEFENSE RESULT

## Overall Performance
Concise examiner-style summary.

## Strong Points
Specific strengths demonstrated during the defense.

## Weak Points
Specific weaknesses, uncertainty, contradictions, or poorly defended areas.

## Best Answer
Identify the strongest defended point and explain why.

## Most Vulnerable Area
Identify the weakest defended area and explain why.

## Examiner Assessment
Explain whether the user genuinely appeared to understand the thesis.

## Overall Defense Rating
Give exactly one rating:
X/10

## Final Decision
Choose exactly one:
- DEFENDED SUCCESSFULLY
- DEFENDED, BUT WITH MAJOR RESERVATIONS
- DEFENSE NOT YET CONVINCING

## Final Examiner Statement
A concise final statement directly addressing the candidate.
"""


def is_defense_command(text: str) -> bool:
    """Return True when the user activates Defense Mode with /defense."""
    if not text:
        return False
    stripped = text.strip().lower()
    return stripped == "/defense" or stripped.startswith("/defense ")


def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to your .env file or deployment secrets."
        )
    return genai.Client(api_key=api_key)

def _split_thesis(
    text: str,
    chunk_size: int = 4000,
    overlap: int = 400,
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


def _retrieve_thesis_context(
    thesis_text: str,
    query: str,
    top_k: int = 4,
) -> str:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    chunks = _split_thesis(thesis_text)

    if not chunks:
        return ""

    if len(chunks) <= top_k:
        return "\n\n---\n\n".join(chunks)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=12000,
    )

    matrix = vectorizer.fit_transform(
        chunks + [query]
    )

    similarities = cosine_similarity(
        matrix[-1],
        matrix[:-1],
    ).flatten()

    indexes = similarities.argsort()[::-1][:top_k]

    return "\n\n--- RELEVANT THESIS SECTION ---\n\n".join(
        chunks[index]
        for index in indexes
    )


def initialize_defense_state(thesis_text: str) -> Dict[str, Any]:
    """Create the initial state for a new thesis defense."""
    if not thesis_text or not thesis_text.strip():
        raise ValueError("No thesis text was provided for Defense Mode.")

    return {
        "active": True,
        "thesis_text": thesis_text,
        "question_number": 0,
        "history": [],
        "strengths": [],
        "weaknesses": [],
        "contradictions": [],
        "scores": [],
        "complete": False,
    }


def _history_text(state: Dict[str, Any]) -> str:
    if not state.get("history"):
        return "No previous questions have been asked."

    parts: List[str] = []
    for item in state["history"]:
        parts.append(
            f"""
Question {item.get('question_number')}:
{item.get('question')}

Candidate answer:
{item.get('answer')}

Examiner evaluation:
{item.get('feedback')}

Temporary score:
{item.get('score')}/10
"""
        )
    return "\n".join(parts)


def generate_next_question(
    state: Dict[str, Any],
    model: Optional[str] = None,
) -> str:
    """
    Generate the next adaptive viva question.

    Returns only the next question.
    """
    if state.get("complete"):
        raise RuntimeError("The defense has already been completed.")

    selected_model = model or DEFAULT_MODEL
    client = _get_client()

    next_number = int(state.get("question_number", 0)) + 1
    history = _history_text(state)retrieval_query = f"""
Generate the next thesis defense question.

Question number:
{next_number}

Previous defense:
{history}

Weaknesses:
{state.get('weaknesses', [])}

Contradictions:
{state.get('contradictions', [])}
"""

relevant_thesis = _retrieve_thesis_context(
    thesis_text=state["thesis_text"],
    query=retrieval_query,
    top_k=5,
)

    prompt = f"""
Conduct the next step of this thesis defense.

This will be question number {next_number}.

THESIS:
================
{state['relevant_thesis']}
================

PREVIOUS DEFENSE HISTORY:
================
{history}
================

Known strengths:
{state.get('strengths', [])}

Known weaknesses:
{state.get('weaknesses', [])}

Known contradictions:
{state.get('contradictions', [])}

Generate ONE examiner question only.

Requirements:
- make it appropriate for question number {next_number}
- progressively increase difficulty
- adapt to previous weaknesses where useful
- do not repeat earlier questions
- do not provide the answer
- do not provide feedback yet
- output only the question text
"""

    response = client.models.generate_content(
        model=selected_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=DEFENSE_SYSTEM_PROMPT,
            temperature=0.35,
            max_output_tokens=1000,
        ),
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty defense question.")

    state["question_number"] = next_number
    return response.text.strip()


def evaluate_answer(
    state: Dict[str, Any],
    question: str,
    answer: str,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate one candidate answer and update defense state.

    Returns:
        {
            "feedback": str,
            "score": int,
            "strengths": list[str],
            "weaknesses": list[str],
            "contradictions": list[str],
            "should_end": bool
        }
    """
    if not answer or not answer.strip():
        raise ValueError("The candidate answer is empty.")

    selected_model = model or DEFAULT_MODEL
    client = _get_client()
relevant_thesis = _retrieve_thesis_context(
    thesis_text=state["thesis_text"],
    query=f"""
Question:
{question}

Candidate answer:
{answer}
""",
    top_k=4,
)
    prompt = f"""
Evaluate the candidate's latest answer.

THESIS:
================
{state['relevant_thesis']}
================

QUESTION:
{question}

CANDIDATE ANSWER:
{answer}

PREVIOUS DEFENSE HISTORY:
{_history_text(state)}

Return VALID JSON ONLY with this exact structure:

{{
  "feedback": "concise examiner feedback",
  "score": 1,
  "strengths": ["specific strength"],
  "weaknesses": ["specific weakness"],
  "contradictions": ["specific contradiction if any"],
  "should_end": false
}}

Rules:
- score must be an integer from 1 to 10
- should_end should usually remain false until at least 8 substantial questions
- identify contradictions only when genuinely present
- feedback should be concise but meaningful
- no markdown
- no text outside the JSON
"""

    response = client.models.generate_content(
        model=selected_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=DEFENSE_SYSTEM_PROMPT,
            temperature=0.15,
            max_output_tokens=1500,
            response_mime_type="application/json",
        ),
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty answer evaluation.")

    try:
        evaluation = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Gemini returned invalid JSON during answer evaluation: {exc}"
        ) from exc

    score = int(evaluation.get("score", 1))
    score = max(1, min(score, 10))

    strengths = evaluation.get("strengths") or []
    weaknesses = evaluation.get("weaknesses") or []
    contradictions = evaluation.get("contradictions") or []

    state["history"].append(
        {
            "question_number": state["question_number"],
            "question": question,
            "answer": answer,
            "feedback": evaluation.get("feedback", ""),
            "score": score,
        }
    )

    state["scores"].append(score)
    state["strengths"].extend(strengths)
    state["weaknesses"].extend(weaknesses)
    state["contradictions"].extend(contradictions)

    should_end = bool(evaluation.get("should_end", False))

    if state["question_number"] >= 12:
        should_end = True

    state["complete"] = should_end

    return {
        "feedback": evaluation.get("feedback", ""),
        "score": score,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "contradictions": contradictions,
        "should_end": should_end,
    }


def generate_final_defense_report(
    state: Dict[str, Any],
    model: Optional[str] = None,
) -> str:
    """Generate the final examiner-style thesis defense report."""
    if not state.get("history"):
        raise ValueError("No defense answers are available to evaluate.")

    selected_model = model or DEFAULT_MODEL
    client = _get_client()

    average_score = (
        sum(state.get("scores", [])) / len(state.get("scores", []))
        if state.get("scores")
        else 0
    )
final_query = f"""
Evaluate the overall thesis defense.

Strengths:
{state.get('strengths', [])}

Weaknesses:
{state.get('weaknesses', [])}

Contradictions:
{state.get('contradictions', [])}

Defense history:
{_history_text(state)}
"""

relevant_thesis = _retrieve_thesis_context(
    thesis_text=state["thesis_text"],
    query=final_query,
    top_k=6,
)
    prompt = f"""
Produce the final thesis defense report.

THESIS:
================
{state['relevant_thesis']}
================

FULL DEFENSE HISTORY:
================
{_history_text(state)}
================

Accumulated strengths:
{state.get('strengths', [])}

Accumulated weaknesses:
{state.get('weaknesses', [])}

Accumulated contradictions:
{state.get('contradictions', [])}

Raw average temporary score:
{average_score:.2f}/10

Follow the exact final report structure from the system instructions.

The final overall rating must be a reasoned holistic score out of 10.
Do not simply copy the arithmetic average if the quality of the defense
supports a slightly different examiner judgment.

Return markdown.
"""

    response = client.models.generate_content(
        model=selected_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=DEFENSE_SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=5000,
        ),
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty final defense report.")

    state["complete"] = True
    return response.text


def render_defense_mode(thesis_text: str) -> None:
    """
    Optional standalone Streamlit renderer for Defense Mode.

    For a larger application, the main app may call the core functions
    directly instead.
    """
    import streamlit as st

    st.subheader("ThesisForge AI - Defense Mode")
    st.caption("Adaptive AI viva: one question at a time, increasing difficulty.")

    if not thesis_text or not thesis_text.strip():
        st.warning("Upload and process a thesis before running /defense.")
        return

    if "defense_state" not in st.session_state:
        st.session_state.defense_state = None

    if st.session_state.defense_state is None:
        if st.button("Start Thesis Defense", type="primary"):
            st.session_state.defense_state = initialize_defense_state(thesis_text)
            st.rerun()
        return

    state = st.session_state.defense_state

    for item in state["history"]:
        with st.chat_message("assistant"):
            st.markdown(f"**Question {item['question_number']}**")
            st.markdown(item["question"])

        with st.chat_message("user"):
            st.markdown(item["answer"])

        with st.chat_message("assistant"):
            st.markdown(
                f"**Examiner feedback:** {item['feedback']}\n\n"
                f"**Answer score:** {item['score']}/10"
            )

    if state.get("complete"):
        if "final_defense_report" not in st.session_state:
            with st.spinner("Preparing final defense report..."):
                st.session_state.final_defense_report = generate_final_defense_report(state)

        st.markdown(st.session_state.final_defense_report)

        if st.button("Start New Defense"):
            st.session_state.defense_state = initialize_defense_state(thesis_text)
            st.session_state.pop("final_defense_report", None)
            st.rerun()
        return

    if "current_defense_question" not in st.session_state:
        with st.spinner("Preparing examiner question..."):
            st.session_state.current_defense_question = generate_next_question(state)

    current_question = st.session_state.current_defense_question

    with st.chat_message("assistant"):
        st.markdown(f"**Question {state['question_number']}**")
        st.markdown(current_question)

    answer = st.chat_input("Defend your thesis...")

    if answer:
        with st.spinner("Examiner is evaluating your answer..."):
            evaluation = evaluate_answer(
                state=state,
                question=current_question,
                answer=answer,
            )

        st.session_state.pop("current_defense_question", None)

        if evaluation["should_end"]:
            state["complete"] = True

        st.rerun()
