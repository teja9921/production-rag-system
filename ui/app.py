import streamlit as st
import requests
import uuid
from datetime import datetime, timedelta
from streamlit_cookies_manager import EncryptedCookieManager

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Agentic Medical RAG", layout="wide")

st.title("🩺 Agentic Medical RAG Assistant")

# -------------------------
# Cookie Manager Setup
# -------------------------

cookies = EncryptedCookieManager(
    prefix="medical_rag_",
    password="streamlit"
)

if not cookies.ready():
    st.stop()

def get_or_create_user_id():
    """Get user_id from cookies or create new one"""
    if "user_id" in cookies:
        return cookies["user_id"]
    
    new_id = str(uuid.uuid4())
    cookies["user_id"] = new_id
    cookies.save()
    return new_id

# -------------------------
# Session Initialization
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
# Sidebar – Conversations
# -------------------------

with st.sidebar:
    st.header("💬 Conversations")

    if st.button("➕ New Chat", use_container_width=True):
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

    # Display all conversations
    for idx, convo in enumerate(st.session_state.conversations):
        conv_id = convo["conversation_id"]
        
        # Handle NULL titles gracefully
        title = convo.get("title") or "New Chat"
        
        # Optional: Show "generating..." for very recent chats
        created_at = convo.get("created_at")
        if not convo.get("title") and created_at:
            try:
                created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                if datetime.utcnow() - created < timedelta(seconds=5):
                    title = "✨ Generating title..."
            except (ValueError, AttributeError):
                pass  # Keep default "New Chat" if datetime parsing fails
        
        is_active = conv_id == st.session_state.conversation_id
        
        # Create columns for title and actions
        col1, col2 = st.columns([4, 1])
        
        with col1:
            button_label = f"{'🔹' if is_active else '⚪'} {title}"
            if st.button(button_label, key=f"conv_{idx}", use_container_width=True):
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
        
        with col2:
            # Show edit menu for active conversation only
            if is_active:
                with st.popover("⚙️", use_container_width=True):
                    st.caption("**Conversation Options**")
                    
                    # Regenerate title button
                    if st.button("🔄 Regenerate Title", key=f"regen_{idx}", use_container_width=True):
                        try:
                            r = requests.post(
                                f"{API_URL}/conversations/{conv_id}/title/regenerate",
                                params={"user_id": st.session_state.user_id},
                                timeout=10
                            )
                            if r.status_code == 200:
                                new_title = r.json()["title"]
                                # Update in session state
                                st.session_state.conversations[idx]["title"] = new_title
                                st.success(f"Title updated!")
                                st.rerun()
                            else:
                                st.error("Failed to regenerate title")
                        except Exception as e:
                            st.error(f"Error: {e}")
                    
                    # Custom title input
                    with st.form(key=f"edit_title_{idx}"):
                        new_title = st.text_input(
                            "✏️ Custom Title", 
                            value=title if title != "New Chat" and title != "✨ Generating title..." else "", 
                            max_chars=200,
                            placeholder="Enter a custom title..."
                        )
                        if st.form_submit_button("Save", use_container_width=True):
                            if new_title.strip():
                                try:
                                    r = requests.put(
                                        f"{API_URL}/conversations/{conv_id}/title",
                                        params={"user_id": st.session_state.user_id},
                                        json={"title": new_title},
                                        timeout=5
                                    )
                                    if r.status_code == 200:
                                        st.session_state.conversations[idx]["title"] = new_title
                                        st.success("Title updated!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to update title")
                                except Exception as e:
                                    st.error(f"Error: {e}")
                            else:
                                st.warning("Title cannot be empty")
                    
                    st.divider()
                    
                    # Delete conversation button (placeholder for now)
                    if st.button("🗑️ Delete", key=f"del_{idx}", use_container_width=True, type="secondary"):
                        try:
                            r = requests.delete(
                                f"{API_URL}/conversations/{conv_id}",
                                params={"user_id": st.session_state.user_id},
                                timeout=5
                            )
                            if r.status_code == 200:
                                # Remove from session state
                                st.session_state.conversations.pop(idx)
                                
                                # Switch to another conversation or create new one
                                if st.session_state.conversations:
                                    st.session_state.conversation_id = st.session_state.conversations[0]["conversation_id"]
                                    # Load messages for new conversation
                                    r = requests.get(
                                        f"{API_URL}/conversations/{st.session_state.conversation_id}/messages",
                                        params={"user_id": st.session_state.user_id}
                                    )
                                    st.session_state.messages = r.json() if r.status_code == 200 else []
                                else:
                                    # Create new conversation if none left
                                    r = requests.post(
                                        f"{API_URL}/conversations",
                                        params={"user_id": st.session_state.user_id}
                                    )
                                    convo = r.json()
                                    st.session_state.conversations = [convo]
                                    st.session_state.conversation_id = convo["conversation_id"]
                                    st.session_state.messages = []
                            
                                st.success("Conversation deleted!")
                                st.rerun()
                            else:
                                st.error("Failed to delete conversation")
                        except Exception as e:
                            st.error(f"Error: {e}")
                        

    # Show conversation count
    st.divider()
    st.caption(f"Total conversations: {len(st.session_state.conversations)}")

# -------------------------
# Chat History Display
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
    
    # Refresh conversations list to get updated title

    # (Title is generated on backend after first message)
    try:
        r = requests.get(
            f"{API_URL}/users/{st.session_state.user_id}/conversations",
            timeout=5
        )
        if r.status_code == 200:
            st.session_state.conversations = r.json()["conversations"]
    except requests.exceptions.RequestException:
        pass  # Silent fail - not critical
