import streamlit as st
import requests
import os
import uuid
import time
from datetime import datetime
#from services.cassandra_connector import CassandraConnector

st.set_page_config(page_title="🧠 RAG Agent", layout="wide")

API_HOST = os.getenv("API_HOST", "ragagentbot")
API_PORT = os.getenv("API_PORT", "8000")
BASE_URL = f"http://{API_HOST}:{API_PORT}"

# ---------------- Initialize Session ----------------
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []
if "session_title" not in st.session_state:
    st.session_state.session_title = None
if "waiting_for_response" not in st.session_state:
    st.session_state.waiting_for_response = False
if "sessions" not in st.session_state:
    st.session_state.sessions = []

headers = {"Authorization": f"Bearer {st.session_state.token}"}

# ---------------- Theme Toggle + Styling ----------------
st.sidebar.title("⚙️ Settings")
st.sidebar.toggle("🌗 Dark Mode", key="dark_mode", value=True)

dark_mode = st.session_state.dark_mode
st.markdown("""
    <style>
    body { background-color: %s; color: %s; }
    .chat-message { background-color: %s; color: %s; border-radius: 12px; padding: 10px; }
    .user { background-color: #4caf50; }
    .confidence-bar { height: 5px; border-radius: 5px; margin-top: 4px; }
    .sidebar-item:hover { background-color: %s; border-radius: 6px; padding: 6px; }
    .footer { text-align: center; font-size: 0.85em; margin-top: 40px; color: #888; }
    .nav { background-color: #222; padding: 0.5rem 1rem; font-size: 1.1rem; color: white; font-weight: bold; }
    </style>
""" % ("#0e1117" if dark_mode else "white", "white" if dark_mode else "black",
       "#2c2c2c" if dark_mode else "#f5f5f5", "white" if dark_mode else "black",
       "#333" if dark_mode else "#eee"), unsafe_allow_html=True)

# ---------------- Navbar ----------------
st.markdown("<div class='nav'>🧠 RAG Chatbot</div>", unsafe_allow_html=True)

# ---------------- Login ----------------
if not st.session_state.token:
    st.title("Welcome to RAG Agent 👋")
    with st.form("auth_form"):
        st.subheader("Login or Register")
        email = st.text_input("Username")
        password = st.text_input("Password", type="password")
        is_register = st.checkbox("Register instead?")
        submit = st.form_submit_button("Submit")

        if submit:
            endpoint = f"{BASE_URL}/register" if is_register else f"{BASE_URL}/login"
            data = {"username": email, "email": email, "password": password, "his_job": "user"}
            if not is_register:
                data = {"username": email, "password": password}
            try:
                res = requests.post(endpoint, data=data)
                if res.ok:
                    st.session_state.token = res.json()["access_token"]
                    st.session_state.username = email
                    st.rerun()
                else:
                    st.error(res.text)
            except:
                st.error("❌ Cannot connect to backend.")
    st.stop()

# ---------------- Fetch Sessions ----------------
try:
    res = requests.get(f"{BASE_URL}/sessions?user={st.session_state.username}", headers=headers)
    if res.ok:
        st.session_state.sessions = res.json().get("sessions", [])
except:
    st.warning("⚠️ Could not load sessions.")

# ---------------- Sidebar Sessions ----------------
st.sidebar.title("📚 Your Chats")

if st.sidebar.button("➕ New Conversation"):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.history = []
    st.session_state.session_title = None
    st.rerun()  # ✅ Correct way to refresh the app

# List old sessions
if st.session_state.sessions:
    st.session_state.sessions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    for session in st.session_state.sessions:
        if st.sidebar.button(session["title"]):
            st.session_state.session_id = session["session_id"]
            st.session_state.session_title = session["title"]

            # Load old history
            try:
                res = requests.get(f"{BASE_URL}/history?session_id={session['session_id']}", headers=headers)
                if res.ok:
                    st.session_state.history = res.json().get("history", [])
                    st.success(f"✅ Loaded {len(st.session_state.history)} messages from '{session['title']}'")
                    st.rerun()  # ✅ refresh chat area to reflect old history
            except Exception as e:
                st.warning(f"⚠️ Could not load chat history: {str(e)}")



# ---------------- Upload ----------------
st.subheader("📤 Upload Document")

with st.form("upload_form"):
    file = st.file_uploader("📚 Select a document to upload", type=["pdf", "txt", "csv"])
    upload = st.form_submit_button("Upload")

    if upload:
        if file:
            with st.spinner("⏳ Uploading and processing your document..."):
                try:
                    files = {"file": (file.name, file.read())}
                    res = requests.post(f"{BASE_URL}/upload", files=files, headers=headers, timeout=300)

                    if res.ok:
                        st.toast("✅ File uploaded and processing started!", icon='🎉')
                    else:
                        error_detail = res.json().get("detail", "Upload failed.")
                        st.error(f"❌ Upload failed: {error_detail}")
                except Exception as e:
                    st.error(f"❌ Backend unreachable: {str(e)}")
        else:
            st.warning("⚠️ Please select a file before uploading.")

# ---------------- Replay Old Chat History ----------------
if st.session_state.history:
    for item in st.session_state.history:
        if item["role"] == "user":
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(item["message"])
        elif item["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(item["message"], unsafe_allow_html=True)

# ---------------- Chat Interface ----------------
st.subheader("💬 Ask a Question")

if not st.session_state.waiting_for_response:
    question = st.chat_input("Type your question...")

    if question:
        if not st.session_state.session_title:
            st.session_state.session_title = question[:30] + "..." if len(question) > 30 else question

            # ✅ Save session and re-fetch sidebar after first message
            try:
                requests.post(f"{BASE_URL}/register_session", data={
                    "session_id": st.session_state.session_id,
                    "title": st.session_state.session_title,
                    "username": st.session_state.username
                }, headers=headers)

                # Refresh sessions
                res = requests.get(f"{BASE_URL}/sessions?user={st.session_state.username}", headers=headers)
                if res.ok:
                    st.session_state.sessions = res.json().get("sessions", [])
            except:
                st.warning("⚠️ Failed to register or refresh sessions.")

        st.session_state.waiting_for_response = True
        st.session_state.history.append({"role": "user", "message": question})

        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            with st.spinner("🤖 Thinking..."):
                try:
                    res = requests.post(f"{BASE_URL}/chat", data={"question": question, "session_id": st.session_state.session_id}, headers=headers, timeout=60)

                    if res.ok:
                        data = res.json()
                        answer = data.get("answer", "No answer returned.")
                        st.session_state.history.append({"role": "assistant", "message": answer})

                        # Typing animation
                        for dots in ["", ".", "..", "..."]:
                            placeholder.markdown(f"Assistant is typing{dots}")
                            time.sleep(0.4)
                        placeholder.markdown(answer, unsafe_allow_html=True)
                    else:
                        placeholder.error("❌ Failed to get a response.")
                except Exception as e:
                    placeholder.error(f"❌ Backend unreachable: {str(e)}")

        # Auto-scroll after sending
        st.markdown("""
            <script>
            var chatBox = window.parent.document.querySelector('.element-container');
            if (chatBox) { chatBox.scrollTop = chatBox.scrollHeight; }
            </script>
        """, unsafe_allow_html=True)

        st.session_state.waiting_for_response = False

# ---------------- Footer ----------------
st.markdown("<div class='footer'>© 2025 RAG Agent | Built by Amine</div>", unsafe_allow_html=True)
