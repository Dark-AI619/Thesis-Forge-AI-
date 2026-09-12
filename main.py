"""
ThesisForge AI - Main Streamlit Application
File: main.py

Project files expected in the same folder:
    main.py
    analyst.py
    defense.py

Install:
    pip install streamlit groq python-dotenv pymupdf

Environment / Streamlit secrets:
    GROQ_API_KEY=your_groq_key

The Analyst and Defense modules created separately use their own model/API
configuration. Keep their required keys in environment variables or
Streamlit secrets as well.

Run locally:
    streamlit run main.py
"""

import os
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from analyst import is_analyst_command, run_analyst_mode
from defense import (
    evaluate_answer,
    generate_final_defense_report,
    generate_next_question,
    initialize_defense_state,
    is_defense_command,
)

load_dotenv()

APP_TITLE = "ThesisForge AI"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

ASSISTANT_SYSTEM_PROMPT = """
You are ThesisForge AI, an academic research assistant.

Your job is to help users understand and improve the thesis/research paper
they uploaded.

You may:
- explain parts of the thesis
- summarize sections
- identify unclear arguments
- discuss methodology
- explain concepts
- help prepare for a defense
- help interpret results
- suggest improvements

Important rules:
- Base thesis-specific claims on the uploaded thesis text.
- Never invent citations, studies, authors, statistics, or findings.
- Clearly distinguish between information found in the thesis and your own
  general academic guidance.
- If the uploaded text does not contain enough information, say so.
- Be direct, specific, and academically useful.
- Do not expose hidden chain-of-thought.

Special commands:
- /analyst starts ThesisForge Analyst Mode.
- /defense starts ThesisForge Defense Mode.
"""


# ---------------------------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 ThesisForge AI")
st.caption("Stress-test your research before your examiner does.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_secret(name: str) -> Optional[str]:
    """
    Read a value from Streamlit secrets first, then environment variables.
    This avoids hardcoding API keys in source code.
    """
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass

    value = os.getenv(name)
    return value if value else None


def extract_pdf_text(uploaded_file) -> str:
    """Extract readable text from an uploaded PDF using PyMuPDF."""
    try:
        pdf_bytes = uploaded_file.getvalue()
        document = fitz.open(stream=pdf_bytes, filetype="pdf")

        pages: List[str] = []

        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append(
                    f"\n\n--- PAGE {page_number} ---\n\n{text}"
                )

        document.close()

        extracted = "".join(pages).strip()

        if not extracted:
            raise ValueError(
                "No readable text was found in this PDF. "
                "The document may be scanned or image-only."
            )

        return extracted

    except Exception as exc:
        raise RuntimeError(f"Could not read the uploaded PDF: {exc}") from exc


def reset_defense() -> None:
    """Clear all state belonging to an active thesis defense."""
    st.session_state.defense_state = None
    st.session_state.current_defense_question = None
    st.session_state.final_defense_report = None


def initialize_session_state() -> None:
    defaults: Dict[str, Any] = {
        "messages": [],
        "thesis_text": "",
        "uploaded_file_id": None,
        "defense_state": None,
        "current_defense_question": None,
        "final_defense_report": None,
        "active_mode": "assistant",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def groq_client() -> Groq:
    api_key = get_secret("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Add it to your .env file "
            "locally or Streamlit secrets when deployed."
        )

    return Groq(api_key=api_key)
def split_thesis_into_chunks(text: str, chunk_size: int = 4000, overlap: int = 400):
    """
    Split thesis into overlapping text chunks.
    Character-based chunking keeps the implementation lightweight.
    """
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def retrieve_relevant_chunks(
    thesis_text: str,
    query: str,
    top_k: int = 4,
) -> str:
    """
    Retrieve only the thesis chunks most relevant to the user's query.
    Uses TF-IDF + cosine similarity.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    chunks = split_thesis_into_chunks(thesis_text)

    if not chunks:
        return ""

    # If the thesis is very small, simply return it.
    if len(chunks) <= top_k:
        return "\n\n---\n\n".join(chunks)

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

    best_indexes = similarities.argsort()[::-1][:top_k]

    selected_chunks = [
        chunks[index]
        for index in best_indexes
    ]

    return "\n\n--- RELEVANT THESIS SECTION ---\n\n".join(
        selected_chunks
    )


def ask_groq(
    user_text: str,
    thesis_text: str,
    previous_messages,
) -> str:

    client = groq_client()

    messages = [
        {
            "role": "system",
            "content": ASSISTANT_SYSTEM_PROMPT,
        }
    ]

    # IMPORTANT:
    # Retrieve only relevant thesis sections.
    if thesis_text:
        relevant_context = retrieve_relevant_chunks(
            thesis_text=thesis_text,
            query=user_text,
            top_k=4,
        )

        messages.append(
            {
                "role": "system",
                "content": (
                    "Below are the most relevant sections retrieved "
                    "from the user's uploaded thesis.\n\n"
                    "Answer thesis-specific questions using this evidence.\n\n"
                    "===== RETRIEVED THESIS CONTEXT =====\n"
                    f"{relevant_context}\n"
                    "===== END CONTEXT ====="
                ),
            }
        )

    # Keep only recent chat history.
    for message in previous_messages[-6:]:
        if message["role"] in {"user", "assistant"}:
            messages.append(
                {
                    "role": message["role"],
                    "content": message["content"],
                }
            )

    messages.append(
        {
            "role": "user",
            "content": user_text,
        }
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=2500,
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "Groq returned an empty response."
        )

    return content



def add_message(role: str, content: str) -> None:
    st.session_state.messages.append(
        {
            "role": role,
            "content": content,
        }
    )


def start_analyst_mode() -> str:
    """Run Analyst Mode against the currently uploaded thesis."""
    if not st.session_state.thesis_text:
        raise ValueError("Upload a thesis PDF before using /analyst.")

    st.session_state.active_mode = "analyst"
    reset_defense()

    return run_analyst_mode(st.session_state.thesis_text)


def start_defense_mode() -> str:
    """Start a fresh adaptive thesis defense and return question one."""
    if not st.session_state.thesis_text:
        raise ValueError("Upload a thesis PDF before using /defense.")

    st.session_state.active_mode = "defense"
    st.session_state.final_defense_report = None

    state = initialize_defense_state(st.session_state.thesis_text)
    st.session_state.defense_state = state

    question = generate_next_question(state)
    st.session_state.current_defense_question = question

    return (
        "## Thesis Defense Started\n\n"
        "I will act as your examiner. I will ask **one question at a time**, "
        "and the questions will become harder as the defense progresses.\n\n"
        f"### Question {state['question_number']}\n\n"
        f"{question}"
    )


def process_defense_answer(answer: str) -> str:
    """Evaluate a defense answer and either ask the next question or finish."""
    state = st.session_state.defense_state
    question = st.session_state.current_defense_question

    if not state or not question:
        raise RuntimeError(
            "No active defense question exists. Type /defense to start again."
        )

    evaluation = evaluate_answer(
        state=state,
        question=question,
        answer=answer,
    )

    feedback = (
        f"**Examiner feedback:** {evaluation['feedback']}\n\n"
        f"**Answer score:** {evaluation['score']}/10"
    )

    if evaluation["should_end"]:
        report = generate_final_defense_report(state)
        st.session_state.final_defense_report = report
        st.session_state.current_defense_question = None
        st.session_state.active_mode = "assistant"

        return (
            f"{feedback}\n\n"
            "---\n\n"
            f"{report}"
        )

    next_question = generate_next_question(state)
    st.session_state.current_defense_question = next_question

    return (
        f"{feedback}\n\n"
        f"### Question {state['question_number']}\n\n"
        f"{next_question}"
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

initialize_session_state()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("ThesisForge")

    uploaded_file = st.file_uploader(
        "Upload thesis",
        type=["pdf"],
        help="Upload a text-based PDF thesis or research paper.",
    )

    if uploaded_file is not None:
        file_id = (
            uploaded_file.name,
            uploaded_file.size,
        )

        if file_id != st.session_state.uploaded_file_id:
            with st.spinner("Reading thesis..."):
                try:
                    extracted_text = extract_pdf_text(uploaded_file)

                    st.session_state.thesis_text = extracted_text
                    st.session_state.uploaded_file_id = file_id
                    st.session_state.messages = []
                    st.session_state.active_mode = "assistant"
                    reset_defense()

                    st.success("Thesis loaded successfully.")

                except Exception as exc:
                    st.session_state.thesis_text = ""
                    st.session_state.uploaded_file_id = None
                    st.error(str(exc))

    if st.session_state.thesis_text:
        st.success("Thesis ready for analysis")
        st.caption(
            f"{len(st.session_state.thesis_text):,} extracted characters"
        )
    else:
        st.info("Upload a thesis PDF to begin.")

    st.divider()

    st.subheader("Commands")
    st.code("/analyst", language=None)
    st.caption("Full five-part thesis evaluation and score out of 100.")

    st.code("/defense", language=None)
    st.caption("Start an adaptive AI thesis viva.")

    st.divider()

    mode_labels = {
        "assistant": "Assistant",
        "analyst": "Analyst",
        "defense": "Defense",
    }

    st.write(
        f"**Current mode:** "
        f"{mode_labels.get(st.session_state.active_mode, 'Assistant')}"
    )

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.active_mode = "assistant"
        reset_defense()
        st.rerun()


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------

if not st.session_state.thesis_text:
    st.info(
        "Upload your thesis PDF from the sidebar. "
        "After that you can chat normally, type **/analyst**, "
        "or type **/defense**."
    )

# Render conversation history.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input(
    "Ask about your thesis, type /analyst, or type /defense..."
)

if user_input:
    add_message("user", user_input)

    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        with st.chat_message("assistant"):

            # /analyst always starts a fresh analyst run.
            if is_analyst_command(user_input):
                with st.spinner(
                    "Analyzing thesis, research quality, novelty, impact, "
                    "modern standards, and weaknesses..."
                ):
                    response_text = start_analyst_mode()

            # /defense always starts a fresh defense.
            elif is_defense_command(user_input):
                with st.spinner("Preparing your thesis defense..."):
                    response_text = start_defense_mode()

            # During an active defense, ordinary messages are candidate answers.
            elif (
                st.session_state.active_mode == "defense"
                and st.session_state.defense_state is not None
            ):
                with st.spinner("Examiner is evaluating your answer..."):
                    response_text = process_defense_answer(user_input)

            # Otherwise use normal Groq-powered assistant mode.
            else:
                st.session_state.active_mode = "assistant"

                with st.spinner("Thinking..."):
                    # Exclude the message just added so ask_groq can append it once.
                    history = st.session_state.messages[:-1]

                    response_text = ask_groq(
                        user_text=user_input,
                        thesis_text=st.session_state.thesis_text,
                        previous_messages=history,
                    )

            st.markdown(response_text)

        add_message("assistant", response_text)

    except Exception as exc:
        error_message = f"**Error:** {exc}"

        with st.chat_message("assistant"):
            st.error(str(exc))

        add_message("assistant", error_message)
