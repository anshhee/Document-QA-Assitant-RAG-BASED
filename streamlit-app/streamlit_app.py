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


# ── Visual-only CSS ─────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: "Inter", sans-serif;
    }

    /* ── Canvas ───────────────────────────────────────────────────────── */
    .stApp,
    [data-testid="stAppViewContainer"] {
        background: #f5f2ec !important;
        color: #1c1c1c !important;
    }

    [data-testid="stAppViewContainer"] .main .block-container {
        max-width: 1100px !important;
        width: 100% !important;
        margin: 0 auto !important;
        padding: 2rem 3rem 6rem !important;
    }

    /* ── Sidebar ──────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #faf8f5 !important;
        border-right: 1px solid #ddd9d2 !important;
    }

    [data-testid="stSidebar"] .block-container {
        padding: 1.75rem 1.1rem !important;
    }

    /* Brand block */
    .brand {
        padding-bottom: 1rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #e8e4dc;
    }
    .brand-title {
        color: #1c1c1c;
        font-size: 0.95rem;
        font-weight: 600;
    }
    .brand-subtitle {
        margin-top: 0.15rem;
        color: #9c9690;
        font-size: 0.7rem;
    }

    /* Generic label */
    .section-label {
        color: #9c9690;
        font-size: 0.64rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 1.25rem 0 0.4rem;
    }

    /* All sidebar buttons */
    [data-testid="stSidebar"] button {
        border: 1px solid #ddd9d2 !important;
        border-radius: 5px !important;
        background: #ffffff !important;
        color: #2d6a4f !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        transition: background 0.12s, border-color 0.12s !important;
    }
    [data-testid="stSidebar"] button:hover {
        background: #edf4f0 !important;
        border-color: #b7d5c7 !important;
        color: #1e4d39 !important;
    }

    /* Conversation list buttons (non-first) */
    [data-testid="stSidebar"] .stButton:not(:first-of-type) > button {
        background: transparent !important;
        border-color: transparent !important;
        color: #6b6762 !important;
        font-weight: 400 !important;
        text-align: left !important;
        padding: 0.28rem 0.55rem !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    [data-testid="stSidebar"] .stButton:not(:first-of-type) > button:hover {
        background: #f0ede7 !important;
        border-color: #e8e4dc !important;
        color: #1c1c1c !important;
    }

    /* Active conversation highlight */
    .conv-active > div > button {
        background: #edf4f0 !important;
        border-left: 2px solid #2d6a4f !important;
        border-radius: 0 4px 4px 0 !important;
        color: #1c1c1c !important;
        font-weight: 500 !important;
    }

    /* Separator between sidebar items */
    .conv-sep {
        height: 1px;
        background: #ede9e3;
        margin: 0.05rem 0;
    }

    /* ── Page heading ─────────────────────────────────────────────────── */
    .page-heading {
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #ddd9d2;
    }
    .page-title {
        margin: 0;
        color: #1c1c1c;
        font-size: 1.25rem;
        font-weight: 600;
    }
    .page-subtitle {
        margin: 0.3rem 0 0;
        color: #9c9690;
        font-size: 0.8rem;
    }

    /* ── Upload area ──────────────────────────────────────────────────── */
    .upload-label {
        color: #9c9690;
        font-size: 0.64rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
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

    /* ── Document status strip ────────────────────────────────────────── */
    .doc-strip {
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
        padding: 0.65rem 0.85rem;
        margin: 0.75rem 0 0.5rem;
        border: 1px solid #b7d5c7;
        border-radius: 6px;
        background: #edf4f0;
    }
    .doc-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #2d6a4f;
        flex-shrink: 0;
        margin-top: 0.3rem;
    }
    .doc-name {
        font-size: 0.82rem;
        font-weight: 600;
        color: #1e4d39;
        margin: 0 0 0.2rem;
        word-break: break-all;
    }
    .doc-meta {
        font-size: 0.72rem;
        color: #4a8a6a;
        margin: 0;
    }

    /* ── Thread label ─────────────────────────────────────────────────── */
    .thread-label {
        color: #9c9690;
        font-size: 0.64rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding-bottom: 0.5rem;
        margin-top: 1.25rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid #e8e4dc;
    }

    /* ── Chat messages ────────────────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        padding: 0.75rem 0 !important;
        border-bottom: 1px solid #ede9e3 !important;
        background: transparent !important;
        border-radius: 0 !important;
        border-top: none !important;
        border-left: none !important;
        border-right: none !important;
    }

    /* User — very light green tint */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #f0f7f2 !important;
        border: 1px solid #d4e9dc !important;
        border-radius: 6px !important;
        padding: 0.7rem 0.85rem !important;
        margin-bottom: 0.4rem !important;
    }

    /* Assistant — clean off-white */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: #faf8f4 !important;
        border: 1px solid #e8e4dc !important;
        border-radius: 6px !important;
        padding: 0.7rem 0.85rem !important;
        margin-bottom: 0.4rem !important;
    }

    /* ── References expander ──────────────────────────────────────────── */
    [data-testid="stExpander"] {
        margin-top: 0.5rem !important;
        border: 1px solid #ddd9d2 !important;
        border-radius: 5px !important;
        background: #faf8f4 !important;
        box-shadow: none !important;
    }
    [data-testid="stExpander"] summary {
        font-size: 0.74rem !important;
        font-weight: 600 !important;
        color: #9c9690 !important;
        letter-spacing: 0.04em !important;
    }

    .ref-row {
        display: flex;
        gap: 0.5rem;
        padding: 0.4rem 0;
        border-bottom: 1px solid #ede9e3;
        font-size: 0.79rem;
        color: #5a5550;
        line-height: 1.4;
    }
    .ref-row:last-child { border-bottom: none; }

    .ref-num {
        flex-shrink: 0;
        font-size: 0.67rem;
        font-weight: 700;
        color: #2d6a4f;
        border: 1px solid #b7d5c7;
        border-radius: 3px;
        padding: 0.08rem 0.32rem;
        margin-top: 0.12rem;
        letter-spacing: 0.02em;
    }

    .ref-body { flex: 1; }
    .ref-title { font-weight: 500; color: #3a3530; }
    .ref-page {
        font-size: 0.69rem;
        color: #9c9690;
        margin-left: 0.35rem;
    }

    /* ── Empty state ──────────────────────────────────────────────────── */
    .empty-state {
        padding: 2rem 0;
        color: #9c9690;
        font-size: 0.83rem;
    }

    /* ── Alerts ───────────────────────────────────────────────────────── */
    [data-testid="stAlert"] {
        border-radius: 5px !important;
        font-size: 0.82rem !important;
    }

    hr { border-color: #e8e4dc !important; }

    /* ── Chat input footer ────────────────────────────────────────────── */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottom"] [data-testid="stChatInput"] {
        background: #f5f2ec !important;
        color: #1c1c1c !important;
    }
    [data-testid="stBottom"] {
        border-top: 1px solid #ddd9d2 !important;
        box-shadow: 0 -2px 8px rgba(0,0,0,0.04) !important;
    }
    [data-testid="stChatInput"] textarea {
        border: 1px solid #c9c4bc !important;
        border-radius: 6px !important;
        background: #ffffff !important;
        color: #1c1c1c !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #2d6a4f !important;
        box-shadow: 0 0 0 2px rgba(45,106,79,0.1) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #9c9690 !important;
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
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ────────────────────────────────────────────────────────────────
def _fmt_size(n_bytes: int) -> str:
    """Return a human-readable file size string."""
    if n_bytes >= 1_048_576:
        return f"{n_bytes / 1_048_576:.1f} MB"
    if n_bytes >= 1_024:
        return f"{n_bytes / 1_024:.0f} KB"
    return f"{n_bytes} B"


def create_conversation(document_id=None):
    conversation_id = str(uuid4())

    st.session_state.conversations[conversation_id] = {
        "title": "New conversation",
        "document_id": document_id,
        "messages": [],
    }

    st.session_state.active_conversation_id = conversation_id


# ── Session state ───────────────────────────────────────────────────────────
if "document_id" not in st.session_state:
    st.session_state.document_id = None

# Extra display-only metadata (visual only, no logic impact)
if "doc_meta" not in st.session_state:
    st.session_state.doc_meta = {}          # {filename, size_bytes, chunks, filetype}

if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "active_conversation_id" not in st.session_state:
    create_conversation()


# ── Sidebar ─────────────────────────────────────────────────────────────────
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

    # New session button
    if st.button("+ New session", use_container_width=True):
        create_conversation(document_id=st.session_state.document_id)
        st.rerun()

    # Clear conversation button (visual only — clears messages in active conv)
    active_conv_ref = st.session_state.conversations.get(
        st.session_state.active_conversation_id, {}
    )
    if active_conv_ref.get("messages"):
        if st.button(
            "Clear conversation",
            use_container_width=True,
            key="clear_conv",
        ):
            active_conv_ref["messages"] = []
            active_conv_ref["title"] = "New conversation"
            st.rerun()

    st.markdown(
        '<div class="section-label">Sessions</div>',
        unsafe_allow_html=True,
    )

    visible_conversations = [
        (cid, conv)
        for cid, conv in st.session_state.conversations.items()
        if conv["messages"]
        or cid == st.session_state.active_conversation_id
    ]

    if not visible_conversations:
        st.caption("No sessions yet.")

    for i, (conversation_id, conversation) in enumerate(visible_conversations):
        label = conversation["title"]
        is_active = (
            conversation_id == st.session_state.active_conversation_id
        )

        if is_active:
            st.markdown('<div class="conv-active">', unsafe_allow_html=True)

        if st.button(
            label,
            key=f"conversation_{conversation_id}",
            use_container_width=True,
        ):
            st.session_state.active_conversation_id = conversation_id
            st.rerun()

        if is_active:
            st.markdown("</div>", unsafe_allow_html=True)

        if i < len(visible_conversations) - 1:
            st.markdown('<div class="conv-sep"></div>', unsafe_allow_html=True)


# ── Page heading ────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="page-heading">
        <h1 class="page-title">Document Q&amp;A</h1>
        <p class="page-subtitle">Upload a PDF or DOCX, then ask questions in natural language.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Upload section ──────────────────────────────────────────────────────────
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
    upload_key = f"{uploaded_file.name}:{uploaded_file.size}"

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

                # Store display-only metadata
                st.session_state.doc_meta = {
                    "filename": data["filename"],
                    "chunks": data["chunks_indexed"],
                    "size_bytes": uploaded_file.size,
                    "filetype": uploaded_file.type or "",
                }

                create_conversation(document_id=data["document_id"])

            except requests.RequestException as exc:
                st.error(f"Upload failed: {exc}")


# ── "New document" reset action ─────────────────────────────────────────────
# Shown only after a document is loaded; clears doc state so a new file
# can be uploaded. Does not touch conversations or the backend.
if st.session_state.document_id and st.session_state.doc_meta:
    col_strip, col_action = st.columns([6, 1])

    meta = st.session_state.doc_meta
    ext = (meta.get("filetype", "") or "").split("/")[-1].upper() or "FILE"
    size_str = _fmt_size(meta.get("size_bytes", 0))
    chunks = meta.get("chunks", "?")

    with col_strip:
        st.markdown(
            f"""
            <div class="doc-strip">
                <span class="doc-dot"></span>
                <div>
                    <div class="doc-name">{meta.get('filename', 'Untitled')}</div>
                    <div class="doc-meta">{ext} &middot; {size_str} &middot; {chunks} chunks indexed</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_action:
        st.markdown("<div style='padding-top:0.9rem'></div>", unsafe_allow_html=True)
        if st.button("New doc", key="new_doc"):
            st.session_state.document_id = None
            st.session_state.doc_meta = {}
            st.session_state.pop("indexed_upload_key", None)
            st.rerun()


# ── Active conversation ─────────────────────────────────────────────────────
active_conversation = st.session_state.conversations[
    st.session_state.active_conversation_id
]

active_conversation["document_id"] = st.session_state.document_id


if st.session_state.document_id:

    st.markdown(
        '<div class="thread-label">Conversation</div>',
        unsafe_allow_html=True,
    )

    # ── Render conversation history ─────────────────────────────────────
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
                        page = source.get("page") or source.get("page_number")
                        page_tag = (
                            f'<span class="ref-page">p.{page}</span>'
                            if page
                            else ""
                        )

                        st.markdown(
                            f'<div class="ref-row">'
                            f'<span class="ref-num">{index}</span>'
                            f'<span class="ref-body">'
                            f'<span class="ref-title">{title}</span>'
                            f'{page_tag}'
                            f"</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        st.caption(source.get("text", ""))

    # ── Empty state ─────────────────────────────────────────────────────
    if not active_conversation["messages"]:
        st.markdown(
            '<div class="empty-state">No messages yet. Ask a question below to begin.</div>',
            unsafe_allow_html=True,
        )

    # ── Chat input ──────────────────────────────────────────────────────
    question = st.chat_input("Ask something about the uploaded document...")

    if question:
        active_conversation["messages"].append(
            {"role": "user", "content": question}
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
                            "document_id": st.session_state.document_id,
                            "conversation_id": (
                                st.session_state.active_conversation_id
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
                            for index, source in enumerate(sources, start=1):
                                title = (
                                    source.get("citation")
                                    or source.get("filename")
                                    or source.get("source")
                                    or f"Source {index}"
                                )
                                page = (
                                    source.get("page")
                                    or source.get("page_number")
                                )
                                page_tag = (
                                    f'<span class="ref-page">p.{page}</span>'
                                    if page
                                    else ""
                                )

                                st.markdown(
                                    f'<div class="ref-row">'
                                    f'<span class="ref-num">{index}</span>'
                                    f'<span class="ref-body">'
                                    f'<span class="ref-title">{title}</span>'
                                    f'{page_tag}'
                                    f"</span>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                                st.caption(source.get("text", ""))

                except requests.RequestException as exc:
                    st.error(f"Question failed: {exc}")

else:
    st.markdown(
        '<div class="empty-state">Upload a document to begin asking questions.</div>',
        unsafe_allow_html=True,
    )