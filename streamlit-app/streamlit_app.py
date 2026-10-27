import os
from uuid import uuid4

import requests
import streamlit as st


API_URL = os.getenv(
    "RAG_API_URL",
    "http://127.0.0.1:8000",
)

st.set_page_config(
    page_title="Document QA",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap'
    );

    html,
    body,
    [class*="css"] {
        font-family: "Inter", sans-serif;
    }

    .stApp,
    [data-testid="stAppViewContainer"] {
        background: #f5f2ec !important;
        color: #1c1c1c !important;
    }

    [data-testid="stHeader"] {
        background: #f5f2ec !important;
    }

    [data-testid="stToolbar"] {
        background: transparent !important;
    }

    [data-testid="stAppViewContainer"] .main .block-container {
        max-width: 1100px !important;
        width: 100% !important;
        margin: 0 auto !important;
        padding: 2rem 3rem 6rem !important;
    }

    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #ddd9d2 !important;
    }

    [data-testid="stSidebar"] .block-container {
        padding: 2rem 1.25rem !important;
    }

    .brand {
        padding-bottom: 1.2rem;
        margin-bottom: 1.2rem;
        border-bottom: 1px solid #e8e4dc;
    }

    .brand-title {
        color: #1c1c1c;
        font-size: 1rem;
        font-weight: 600;
    }

    .brand-subtitle {
        margin-top: 0.2rem;
        color: #8c8780;
        font-size: 0.72rem;
    }

    .section-label,
    .upload-label,
    .thread-label {
        color: #6b6762 !important;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .section-label {
        margin: 1.5rem 0 0.5rem;
    }

    [data-testid="stSidebar"] button {
        border: 1px solid #ddd9d2 !important;
        border-radius: 5px !important;
        background: #ffffff !important;
        color: #2d6a4f !important;
        font-size: 0.8rem !important;
    }

    [data-testid="stSidebar"] button:hover {
        background: #edf4f0 !important;
        border-color: #b7d5c7 !important;
    }

    .page-heading {
        padding-bottom: 1.2rem;
        margin-bottom: 1.8rem;
        border-bottom: 1px solid #ddd9d2;
    }

    .page-title {
        margin: 0;
        color: #1c1c1c;
        font-size: 1.35rem;
        font-weight: 600;
    }

    .page-subtitle {
        margin: 0.35rem 0 0;
        color: #8c8780;
        font-size: 0.82rem;
    }

    .upload-label {
        margin-bottom: 0.45rem;
    }

    [data-testid="stFileUploader"] {
        border: 1px solid #ddd9d2 !important;
        border-radius: 6px !important;
        background: #ffffff !important;
    }

    [data-testid="stFileUploader"] section {
        border: none !important;
        background: #ffffff !important;
    }

    [data-testid="stFileUploader"] button {
        border: 1px solid #ddd9d2 !important;
        background: #ffffff !important;
        color: #2d6a4f !important;
    }

    .document-strip {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.4rem 0.7rem;
        margin: 1rem 0 1.5rem;
        border: 1px solid #b7d5c7;
        border-radius: 5px;
        background: #edf4f0;
        color: #2d6a4f;
        font-size: 0.78rem;
    }

    .document-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #2d6a4f;
    }

    .thread-label {
        padding-bottom: 0.55rem;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid #e8e4dc;
    }

    [data-testid="stChatMessage"] {
        padding: 0.75rem 0 !important;
        border-bottom: 1px solid #e8e4dc !important;
        background: transparent !important;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div {
        color: #1c1c1c !important;
    }

    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] code {
        color: #1c1c1c !important;
    }

    [data-testid="stExpander"] {
        margin-top: 0.7rem;
        border: 1px solid #ddd9d2 !important;
        border-radius: 5px !important;
        background: #faf8f4 !important;
    }

    [data-testid="stExpander"] p,
    [data-testid="stExpander"] span,
    [data-testid="stExpander"] div {
        color: #5f5a55 !important;
    }

    .reference-row {
        padding: 0.45rem 0;
        border-bottom: 1px solid #eeeae3;
        color: #5f5a55 !important;
        font-size: 0.8rem;
    }

    .reference-row:last-child {
        border-bottom: none;
    }

    .empty-state {
        padding: 2.5rem 0;
        color: #8c8780 !important;
        font-size: 0.85rem;
    }

    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottom"] [data-testid="stChatInput"] {
        background: #f5f2ec !important;
        color: #1c1c1c !important;
    }

    [data-testid="stBottom"] {
        padding: 0.6rem 0 !important;
        border-top: 1px solid #ddd9d2 !important;
        box-shadow: 0 -2px 8px rgba(0, 0, 0, 0.04) !important;
    }

    [data-testid="stChatInput"] textarea {
        min-height: 42px !important;
        border: 1px solid #c9c4bc !important;
        border-radius: 6px !important;
        background: #ffffff !important;
        color: #1c1c1c !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInput"] textarea:focus {
        border-color: #2d6a4f !important;
        box-shadow: 0 0 0 2px rgba(45, 106, 79, 0.12) !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #8c8780 !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInput"] button {
        border: 1px solid #c9c4bc !important;
        border-radius: 5px !important;
        background: #ffffff !important;
        color: #2d6a4f !important;
    }

    [data-testid="stChatInput"] button:hover {
        background: #edf4f0 !important;
        border-color: #b7d5c7 !important;
    }

    hr {
        border-color: #e8e4dc !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def create_conversation(document_id=None):
    conversation_id = str(uuid4())

    st.session_state.conversations[conversation_id] = {
        "title": "New conversation",
        "document_id": document_id,
        "messages": [],
    }

    st.session_state.active_conversation_id = conversation_id


if "document_id" not in st.session_state:
    st.session_state.document_id = None

if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "active_conversation_id" not in st.session_state:
    create_conversation()


with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-title">Document QA</div>
            <div class="brand-subtitle">Research assistant</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "+ New session",
        use_container_width=True,
    ):
        create_conversation(
            document_id=st.session_state.document_id
        )
        st.rerun()

    st.markdown(
        '<div class="section-label">Sessions</div>',
        unsafe_allow_html=True,
    )

    visible_conversations = [
        (conversation_id, conversation)
        for conversation_id, conversation in (
            st.session_state.conversations.items()
        )
        if (
            conversation["messages"]
            or conversation_id
            == st.session_state.active_conversation_id
        )
    ]

    if not visible_conversations:
        st.caption("No sessions yet.")

    for conversation_id, conversation in visible_conversations:
        if st.button(
            conversation["title"],
            key=f"conversation_{conversation_id}",
            use_container_width=True,
        ):
            st.session_state.active_conversation_id = (
                conversation_id
            )
            st.rerun()


st.markdown(
    """
    <div class="page-heading">
        <h1 class="page-title">Document Q&amp;A</h1>
        <p class="page-subtitle">
            Upload a PDF or DOCX, then ask questions in natural language.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="upload-label">Source document</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "docx"],
    label_visibility="collapsed",
)


if uploaded_file is not None:
    upload_key = (
        f"{uploaded_file.name}:{uploaded_file.size}"
    )

    if st.session_state.get("indexed_upload_key") != upload_key:
        with st.spinner("Uploading and indexing document..."):
            try:
                response = requests.post(
                    f"{API_URL}/documents/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    },
                    timeout=300,
                )

                response.raise_for_status()
                data = response.json()

                st.session_state.document_id = data["document_id"]
                st.session_state.indexed_upload_key = upload_key

                create_conversation(
                    document_id=data["document_id"]
                )

                st.success(
                    f"Indexed {data['filename']} — "
                    f"{data['chunks_indexed']} chunks ready."
                )

            except requests.RequestException as exc:
                st.error(f"Upload failed: {exc}")


active_conversation = st.session_state.conversations[
    st.session_state.active_conversation_id
]

active_conversation["document_id"] = (
    st.session_state.document_id
)


if st.session_state.document_id:
    document_id = st.session_state.document_id

    st.markdown(
        f"""
        <div class="document-strip">
            <span class="document-dot"></span>
            Active document:
            <strong>{document_id}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="thread-label">Conversation</div>',
        unsafe_allow_html=True,
    )

    for message in active_conversation["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("References"):
                    for index, source in enumerate(
                        message["sources"],
                        start=1,
                    ):
                        title = (
                            source.get("citation")
                            or source.get("filename")
                            or source.get("source")
                            or f"Source {index}"
                        )

                        st.markdown(
                            f'<div class="reference-row">'
                            f"<strong>{index}.</strong> {title}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                        st.caption(
                            source.get("text", "")
                        )

    if not active_conversation["messages"]:
        st.markdown(
            '<div class="empty-state">'
            "No messages yet. Ask a question below to begin."
            "</div>",
            unsafe_allow_html=True,
        )

    question = st.chat_input(
        "Ask something about the uploaded document..."
    )

    if question:
        active_conversation["messages"].append(
            {
                "role": "user",
                "content": question,
            }
        )

        if active_conversation["title"] == "New conversation":
            active_conversation["title"] = question[:40]

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence..."):
                try:
                    response = requests.post(
                        f"{API_URL}/ask",
                        json={
                            "question": question,
                            "document_id": (
                                st.session_state.document_id
                            ),
                            "conversation_id": (
                                st.session_state
                                .active_conversation_id
                            ),
                        },
                        timeout=300,
                    )

                    response.raise_for_status()
                    data = response.json()

                    answer = data["answer"]
                    sources = data.get("sources", [])

                    st.markdown(answer)

                    active_conversation["messages"].append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )

                    if sources:
                        with st.expander("References"):
                            for index, source in enumerate(
                                sources,
                                start=1,
                            ):
                                title = (
                                    source.get("citation")
                                    or source.get("filename")
                                    or source.get("source")
                                    or f"Source {index}"
                                )

                                st.markdown(
                                    f'<div class="reference-row">'
                                    f"<strong>{index}.</strong> {title}"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                                st.caption(
                                    source.get("text", "")
                                )

                except requests.RequestException as exc:
                    st.error(f"Question failed: {exc}")

else:
    st.markdown(
        '<div class="empty-state">'
        "Upload a PDF or DOCX above to begin asking questions."
        "</div>",
        unsafe_allow_html=True,
    )