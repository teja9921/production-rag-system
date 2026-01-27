import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Agentic Medical RAG", layout="wide")

st.title("🩺 Agentic Medical RAG Assistant")

# -------------------------
# Session Initialization
# -------------------------

if "user_id" not in st.session_state:
    r = requests.post(f"{API_URL}/users")
    st.session_state.user_id = r.json()["user_id"]

if "conversations" not in st.session_state:
    r = requests.get(
        f"{API_URL}/users/{st.session_state.user_id}/conversations"
    )
    st.session_state.conversations = r.json()["conversations"]

if "conversation_id" not in st.session_state:
    if st.session_state.conversations:
        st.session_state.conversation_id = (
            st.session_state.conversations[0]["conversation_id"]
        )
    else:
        r = requests.post(
            f"{API_URL}/conversations",
            params={"user_id": st.session_state.user_id},
        )
        convo = r.json()
        st.session_state.conversations.append(convo)
        st.session_state.conversation_id = convo["conversation_id"]

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# Sidebar — Conversations
# -------------------------

with st.sidebar:
    st.header("💬 Conversations")

    if st.button("➕ New Chat"):
        r = requests.post(
            f"{API_URL}/conversations",
            params={"user_id": st.session_state.user_id},
        )
        convo = r.json()
        st.session_state.conversations.append(convo)
        st.session_state.conversation_id = convo["conversation_id"]
        st.session_state.messages = []
        st.rerun()

    for convo in st.session_state.conversations:
        if st.button(convo["conversation_id"][:8]):
            st.session_state.conversation_id = convo["conversation_id"]
            r = requests.get(
                url = f"{API_URL}/conversations/{st.session_state.conversation_id}/messages"
            )
            st.session_state.messages = r.json()
            st.rerun()

# -------------------------
# Chat History Display
# -------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# Chat Input
# -------------------------

query = st.chat_input("Ask a medical question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_answer = ""

        with requests.post(
            f"{API_URL}/conversations/{st.session_state.conversation_id}/stream",
            json={"query": query},
            stream=True,
        ) as r:
            for chunk in r.iter_content(chunk_size=None):
                token = chunk.decode("utf-8")
                full_answer += token
                response_placeholder.markdown(full_answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_answer}
    )
