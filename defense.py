"""
ThesisForge AI - Defense Mode
File: defense.py

Purpose:
- Activate Defense Mode using /defense
- Conduct an adaptive thesis viva/defense
- Ask one question at a time
- Make questions progressively harder
- Evaluate each candidate answer
- Track strengths, weaknesses, contradictions, and scores
- Use TF-IDF retrieval so the whole thesis is not sent on every request
- Finish with an overall defense report and rating out of 10

Dependencies:
    pip install groq python-dotenv scikit-learn

Environment variable / Streamlit secret:
    GROQ_API_KEY=your_groq_key

Expected integration:
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
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

DEFAULT_MODEL = os.getenv(
    "GROQ_DEFENSE_MODEL",
    "openai/gpt-oss-120b",
)

DEFENSE_SYSTEM_PROMPT = """
You are THESISFORGE AI - DEFENSE MODE.

You are acting as a demanding but fair thesis examiner conducting an oral
thesis defense/viva.

Your goal is to stress-test whether the candidate genuinely understands and
can defend the uploaded thesis.

==================================================
DEFENSE RULES
==================================================

1. Ask ONE question at a time.

2. Questions must be based on the supplied thesis evidence.

3. Start with foundational questions and progressively increase difficulty.

4. Adapt the next question based on:
- previous answers
- weak points
- contradictions
- unsupported claims
- methodological vulnerabilities
- unexplained assumptions
- uncertainty
- areas the candidate failed to defend clearly

5. Do not repeat questions unless intentionally revisiting a contradiction.

6. The defense should generally progress through:
- Stage 1: Research understanding
- Stage 2: Problem and motivation
- Stage 3: Literature and originality
- Stage 4: Methodology
- Stage 5: Data/evidence
- Stage 6: Results and interpretation
- Stage 7: Limitations
- Stage 8: Counterarguments
- Stage 9: Advanced examiner challenge
- Stage 10: Final defense challenge

7. Questions should become harder over time.

8. Challenge the candidate where appropriate:
- Why was this method chosen?
- Why not an alternative?
- What assumptions does this rely on?
- What evidence supports this claim?
- What would invalidate the conclusion?
- What is actually novel here?
- What would a skeptical examiner attack?
- What happens if an assumption fails?
- How generalizable are the results?
- Why should this research matter?

9. Never fabricate thesis content.

10. Never fabricate external research, citations, authors, papers, statistics,
or findings.

11. Do not claim live web verification.

12. Do not expose hidden chain-of-thought.

==================================================
ANSWER EVALUATION
==================================================

Evaluate every answer for:

- correctness
- relevance
- clarity
- confidence
- evidence
- depth
- consistency with the thesis
- ability to handle challenge

Each answer receives an integer score from 1 to 10.

Strong answers:
- directly answer the question
- use thesis evidence
- show genuine understanding
- acknowledge limitations
- handle counterarguments
- stay consistent with the thesis

Weak answers:
- are vague
- avoid the question
- are unsupported
- contradict the thesis
- overclaim
- misunderstand the work
- cannot justify methodology or results

==================================================
ENDING CONDITIONS
==================================================

The defense should normally continue until one of these is true:

A. At least 8 substantial questions have been answered and the candidate has
demonstrated strong command.

B. Around 10 to 12 substantial questions have been answered and enough
evidence exists for a final evaluation.

C. Major weaknesses make further questioning unnecessary.

D. The host application explicitly requests the final report.

Do not end after only 2 or 3 questions unless there is a serious failure.

==================================================
FINAL REPORT
==================================================

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
Explain whether the candidate genuinely appeared to understand the thesis.

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
    """Split thesis text into overlapping chunks."""
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


def _retrieve_thesis_context(
    thesis_text: str,
    query: str,
    top_k: int = 4,
    max_chars: int = 14000,
) -> str:
    """
    Retrieve the most relevant thesis chunks using TF-IDF cosine similarity.
    """
    chunks = _split_thesis(thesis_text)

    if not chunks:
        return ""

    if len(chunks) <= top_k:
        selected = chunks
    else:
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

        selected = [chunks[index] for index in indexes]

    parts: List[str] = []
    used_chars = 0

    for index, chunk in enumerate(selected, start=1):
        labeled = (
            f"\n\n===== RELEVANT THESIS EXCERPT {index} =====\n"
            f"{chunk}"
        )

        if used_chars + len(labeled) > max_chars:
            break

        parts.append(labeled)
        used_chars += len(labeled)

    return "".join(parts).strip()


def initialize_defense_state(thesis_text: str) -> Dict[str, Any]:
    """Create initial state for a fresh thesis defense."""
    if not thesis_text or not thesis_text.strip():
        raise ValueError(
            "No thesis text was provided for Defense Mode."
        )

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


def _history_text(
    state: Dict[str, Any],
    max_items: int = 8,
) -> str:
    """
    Convert recent defense history to a compact text representation.
    """
    history = state.get("history", [])

    if not history:
        return "No previous questions have been asked."

    recent_history = history[-max_items:]

    parts: List[str] = []

    for item in recent_history:
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
""".strip()
        )

    return "\n\n---\n\n".join(parts)


def generate_next_question(
    state: Dict[str, Any],
    model: Optional[str] = None,
) -> str:
    """
    Generate one adaptive viva question.
    """
    if state.get("complete"):
        raise RuntimeError(
            "The defense has already been completed."
        )

    selected_model = model or DEFAULT_MODEL
    client = _get_client()

    next_number = int(
        state.get("question_number", 0)
    ) + 1

    history = _history_text(state)

    retrieval_query = f"""
Generate the next thesis defense question.

Question number:
{next_number}

Previous defense:
{history}

Known weaknesses:
{state.get('weaknesses', [])}

Known contradictions:
{state.get('contradictions', [])}
"""

    relevant_thesis = _retrieve_thesis_context(
        thesis_text=state["thesis_text"],
        query=retrieval_query,
        top_k=4,
        max_chars=12000,
    )

    if not relevant_thesis:
        raise RuntimeError(
            "No relevant thesis context could be retrieved."
        )

    prompt = f"""
Conduct the next step of this thesis defense.

This will be question number {next_number}.

================ THESIS EVIDENCE ================

{relevant_thesis}

=================================================

PREVIOUS DEFENSE HISTORY:

{history}

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
- avoid repeating earlier questions
- do not provide the answer
- do not provide feedback
- output only the question text
"""

    response = client.chat.completions.create(
        model=selected_model,
        messages=[
            {
                "role": "system",
                "content": DEFENSE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.35,
        max_tokens=700,
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "Groq returned an empty defense question."
        )

    state["question_number"] = next_number

    return content.strip()


def evaluate_answer(
    state: Dict[str, Any],
    question: str,
    answer: str,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate one candidate answer and update the defense state.
    """
    if not answer or not answer.strip():
        raise ValueError(
            "The candidate answer is empty."
        )

    selected_model = model or DEFAULT_MODEL
    client = _get_client()

    relevant_thesis = _retrieve_thesis_context(
        thesis_text=state["thesis_text"],
        query=f"""
Defense question:
{question}

Candidate answer:
{answer}
""",
        top_k=4,
        max_chars=12000,
    )

    if not relevant_thesis:
        raise RuntimeError(
            "No relevant thesis context could be retrieved."
        )

    prompt = f"""
Evaluate the candidate's latest defense answer.

================ THESIS EVIDENCE ================

{relevant_thesis}

=================================================

QUESTION:
{question}

CANDIDATE ANSWER:
{answer}

RECENT DEFENSE HISTORY:
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
- should_end should normally remain false until at least 8 questions
- identify contradictions only when genuinely present
- feedback must be concise but meaningful
- use only thesis evidence supplied here
- do not fabricate evidence
- no markdown
- no text outside the JSON
"""

    response = client.chat.completions.create(
        model=selected_model,
        messages=[
            {
                "role": "system",
                "content": DEFENSE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.15,
        max_tokens=1200,
        response_format={
            "type": "json_object",
        },
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "Groq returned an empty answer evaluation."
        )

    try:
        evaluation = json.loads(content)

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Groq returned invalid JSON during answer evaluation: {exc}"
        ) from exc

    score = int(
        evaluation.get("score", 1)
    )

    score = max(
        1,
        min(score, 10),
    )

    strengths = evaluation.get("strengths") or []
    weaknesses = evaluation.get("weaknesses") or []
    contradictions = evaluation.get("contradictions") or []

    if not isinstance(strengths, list):
        strengths = [str(strengths)]

    if not isinstance(weaknesses, list):
        weaknesses = [str(weaknesses)]

    if not isinstance(contradictions, list):
        contradictions = [str(contradictions)]

    state["history"].append(
        {
            "question_number": state["question_number"],
            "question": question,
            "answer": answer,
            "feedback": evaluation.get(
                "feedback",
                "",
            ),
            "score": score,
        }
    )

    state["scores"].append(score)
    state["strengths"].extend(strengths)
    state["weaknesses"].extend(weaknesses)
    state["contradictions"].extend(contradictions)

    should_end = bool(
        evaluation.get(
            "should_end",
            False,
        )
    )

    # Never allow the model to stop too early unless the host forces it.
    if state["question_number"] < 8:
        should_end = False

    # Hard ceiling for the viva.
    if state["question_number"] >= 12:
        should_end = True

    state["complete"] = should_end

    return {
        "feedback": evaluation.get(
            "feedback",
            "",
        ),
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
    """
    Generate the final examiner-style defense report.
    """
    if not state.get("history"):
        raise ValueError(
            "No defense answers are available to evaluate."
        )

    selected_model = model or DEFAULT_MODEL
    client = _get_client()

    scores = state.get(
        "scores",
        [],
    )

    average_score = (
        sum(scores) / len(scores)
        if scores
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
{_history_text(state, max_items=12)}
"""

    relevant_thesis = _retrieve_thesis_context(
        thesis_text=state["thesis_text"],
        query=final_query,
        top_k=5,
        max_chars=14000,
    )

    if not relevant_thesis:
        raise RuntimeError(
            "No relevant thesis context could be retrieved."
        )

    prompt = f"""
Produce the final ThesisForge thesis defense report.

================ THESIS EVIDENCE ================

{relevant_thesis}

=================================================

FULL DEFENSE HISTORY:

{_history_text(state, max_items=12)}

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

Do not simply copy the arithmetic average if the overall quality of the
defense supports a slightly different examiner judgment.

Return markdown.
"""

    response = client.chat.completions.create(
        model=selected_model,
        messages=[
            {
                "role": "system",
                "content": DEFENSE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
        max_tokens=2200,
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "Groq returned an empty final defense report."
        )

    state["complete"] = True

    return content.strip()


def render_defense_mode(
    thesis_text: str,
) -> None:
    """
    Optional standalone Streamlit renderer.

    The main ThesisForge application can instead call the core functions
    directly when /defense is entered.
    """
    import streamlit as st

    st.subheader(
        "ThesisForge AI - Defense Mode"
    )

    st.caption(
        "Adaptive AI viva: one question at a time, increasing difficulty."
    )

    if not thesis_text or not thesis_text.strip():
        st.warning(
            "Upload and process a thesis before running /defense."
        )
        return

    if "defense_state" not in st.session_state:
        st.session_state.defense_state = None

    if "current_defense_question" not in st.session_state:
        st.session_state.current_defense_question = None

    if "final_defense_report" not in st.session_state:
        st.session_state.final_defense_report = None

    if st.session_state.defense_state is None:
        if st.button(
            "Start Thesis Defense",
            type="primary",
        ):
            st.session_state.defense_state = (
                initialize_defense_state(
                    thesis_text
                )
            )

            st.rerun()

        return

    state = st.session_state.defense_state

    for item in state["history"]:
        with st.chat_message("assistant"):
            st.markdown(
                f"**Question {item['question_number']}**"
            )
            st.markdown(
                item["question"]
            )

        with st.chat_message("user"):
            st.markdown(
                item["answer"]
            )

        with st.chat_message("assistant"):
            st.markdown(
                f"**Examiner feedback:** "
                f"{item['feedback']}\n\n"
                f"**Answer score:** "
                f"{item['score']}/10"
            )

    if state.get("complete"):
        if not st.session_state.final_defense_report:
            with st.spinner(
                "Preparing final defense report..."
            ):
                st.session_state.final_defense_report = (
                    generate_final_defense_report(
                        state
                    )
                )

        st.markdown(
            st.session_state.final_defense_report
        )

        if st.button(
            "Start New Defense"
        ):
            st.session_state.defense_state = (
                initialize_defense_state(
                    thesis_text
                )
            )

            st.session_state.current_defense_question = None
            st.session_state.final_defense_report = None

            st.rerun()

        return

    if not st.session_state.current_defense_question:
        with st.spinner(
            "Preparing examiner question..."
        ):
            st.session_state.current_defense_question = (
                generate_next_question(
                    state
                )
            )

    current_question = (
        st.session_state.current_defense_question
    )

    with st.chat_message("assistant"):
        st.markdown(
            f"**Question {state['question_number']}**"
        )
        st.markdown(
            current_question
        )

    answer = st.chat_input(
        "Defend your thesis..."
    )

    if answer:
        with st.spinner(
            "Examiner is evaluating your answer..."
        ):
            evaluation = evaluate_answer(
                state=state,
                question=current_question,
                answer=answer,
            )

        st.session_state.current_defense_question = None

        if evaluation["should_end"]:
            state["complete"] = True

        st.rerun()
