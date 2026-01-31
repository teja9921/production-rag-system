import streamlit as st
import requests
import uuid
from streamlit_js_eval import streamlit_js_eval
from streamlit_cookies_manager import EncryptedCookieManager

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Agentic Medical RAG", layout="wide")

st.title("🩺 Agentic Medical RAG Assistant")

# -------------------------
# Helper - FIXED VERSION
# -------------------------



cookies = EncryptedCookieManager(
    prefix="medical_rag_",
    password="streamlit"
)

if not cookies.ready():
    st.stop()

def get_or_create_user_id():
    if "user_id" in cookies:
        return cookies["user_id"]
    
    new_id = str(uuid.uuid4())
    cookies["user_id"] = new_id
    cookies.save()
    return new_id

# -------------------------
# Session Initialization - IMPROVED
# -------------------------

if "user_id" not in st.session_state:
    st.session_state.user_id = get_or_create_user_id()
    
    # Create backend user
    try:
        r = requests.post(
            f"{API_URL}/users", 
            json={"user_id": st.session_state.user_id},
            timeout=5
        )
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to create/get user: {e}")
        st.stop()

if "conversations" not in st.session_state:
    try:
        r = requests.get(
            f"{API_URL}/users/{st.session_state.user_id}/conversations",
            timeout=5
        )
        r.raise_for_status()
        st.session_state.conversations = r.json()["conversations"]
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load conversations: {e}")
        st.session_state.conversations = []

if "conversation_id" not in st.session_state:
    if st.session_state.conversations:
        st.session_state.conversation_id = (
            st.session_state.conversations[0]["conversation_id"]
        )
    else:
        try:
            r = requests.post(
                f"{API_URL}/conversations",
                params={"user_id": st.session_state.user_id},
                timeout=5
            )
            r.raise_for_status()
            convo = r.json()
            st.session_state.conversations.append(convo)
            st.session_state.conversation_id = convo["conversation_id"]
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to create conversation: {e}")
            st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Load messages for current conversation if not loaded
if st.session_state.conversation_id and not st.session_state.messages:
    try:
        r = requests.get(
            f"{API_URL}/conversations/{st.session_state.conversation_id}/messages",
            params={"user_id": st.session_state.user_id},
            timeout=5
        )
        if r.status_code == 200:
            st.session_state.messages = r.json()
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not load previous messages: {e}")

# -------------------------
# Sidebar – Conversations - IMPROVED
# -------------------------

with st.sidebar:
    st.header("💬 Conversations")

    if st.button("➕ New Chat"):
        try:
            r = requests.post(
                f"{API_URL}/conversations",
                params={"user_id": st.session_state.user_id},
                timeout=5
            )
            r.raise_for_status()
            convo = r.json()
            st.session_state.conversations.insert(0, convo)  # Add to top
            st.session_state.conversation_id = convo["conversation_id"]
            st.session_state.messages = []
            st.rerun()
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to create new conversation: {e}")

    for idx, convo in enumerate(st.session_state.conversations):
        conv_id = convo["conversation_id"]
        button_label = f"{'🔹' if conv_id == st.session_state.conversation_id else '⚪'} {conv_id[:8]}"
        
        if st.button(button_label, key=f"conv_{idx}"):
            if conv_id != st.session_state.conversation_id:
                st.session_state.conversation_id = conv_id
                
                try:
                    r = requests.get(
                        f"{API_URL}/conversations/{conv_id}/messages",
                        params={"user_id": st.session_state.user_id},
                        timeout=5
                    )
                    r.raise_for_status()
                    st.session_state.messages = r.json()
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to load messages: {e}")
                    st.session_state.messages = []
                
                st.rerun()

# -------------------------
# Chat History Display - IMPROVED
# -------------------------

for msg in st.session_state.messages:
    if isinstance(msg, dict) and "role" in msg and "content" in msg:
        # Filter out NO_ANSWER messages
        if msg["content"] != "NO_ANSWER":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    else:
        st.warning("⚠️ Skipped malformed message")

# -------------------------
# Chat Input - IMPROVED ERROR HANDLING
# -------------------------

query = st.chat_input("Ask a medical question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_answer = ""

        try:
            with requests.post(
                f"{API_URL}/conversations/{st.session_state.conversation_id}/stream",
                params={"user_id": st.session_state.user_id},
                json={"query": query},
                stream=True,
                timeout=30
            ) as r:
                r.raise_for_status()
                
                for chunk in r.iter_content(chunk_size=None):
                    if chunk:
                        token = chunk.decode("utf-8")
                        full_answer += token
                        response_placeholder.markdown(full_answer)
                
                # Handle NO_ANSWER case
                if full_answer == "NO_ANSWER":
                    full_answer = "I couldn't find relevant information to answer your question."
                    response_placeholder.markdown(full_answer)
                
        except requests.exceptions.RequestException as e:
            full_answer = f"Error: Could not get response from server. {e}"
            response_placeholder.error(full_answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_answer}
    )