import streamlit as st
import requests
import os
import json
from datetime import datetime
from pathlib import Path
import uuid

st.set_page_config(page_title="RAG Agent", layout="wide")

API_HOST = os.getenv("API_HOST", "ragagentbot")
API_PORT = os.getenv("API_PORT", "8000")
BASE_URL = f"http://{API_HOST}:{API_PORT}"

SESSION_ID = st.session_state.get("session_id") or str(uuid.uuid4())
st.session_state.session_id = SESSION_ID
CHAT_HISTORY_FILE = Path(f".history_{SESSION_ID}.json")

# Load chat history
if CHAT_HISTORY_FILE.exists():
    with open(CHAT_HISTORY_FILE, "r") as f:
        st.session_state.history = json.load(f)
else:
    st.session_state.history = []

# ---------------- Theme Toggle + Styling ----------------
st.sidebar.title("⚙️ Settings")
st.sidebar.toggle("🌗 Dark Mode", key="dark_mode", value=True)

dark_mode = st.session_state.dark_mode

theme_css = """
    <style>
    body { background-color: #0e1117; color: white; }
    .chat-message { background-color: #2c2c2c; color: white; border-radius: 12px; padding: 10px; }
    .user { background-color: #4caf50; }
    .confidence-bar { height: 5px; border-radius: 5px; margin-top: 4px; }
    .sidebar-item:hover { background-color: #333; border-radius: 6px; padding: 6px; }
    .footer { text-align: center; font-size: 0.85em; margin-top: 40px; color: #888; }
    .nav { background-color: #222; padding: 0.5rem 1rem; font-size: 1.1rem; color: white; font-weight: bold; }
    </style>
""" if dark_mode else """
    <style>
    body { background-color: white; color: black; }
    .chat-message { background-color: #f5f5f5; color: black; border-radius: 12px; padding: 10px; }
    .user { background-color: #4caf50; color: white; }
    .confidence-bar { height: 5px; border-radius: 5px; margin-top: 4px; }
    .sidebar-item:hover { background-color: #eee; border-radius: 6px; padding: 6px; }
    .footer { text-align: center; font-size: 0.85em; margin-top: 40px; color: #666; }
    .nav { background-color: #f0f0f0; padding: 0.5rem 1rem; font-size: 1.1rem; color: #333; font-weight: bold; }
    </style>
"""
st.markdown(theme_css, unsafe_allow_html=True)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("📚 Past Chats")
    if "username" in st.session_state:
        st.write(f"👋 Welcome, **{st.session_state.username}**")
    for msg in reversed(st.session_state.history[-10:]):
        role = "🧑" if msg["role"] == "user" else "🤖"
        st.markdown(f"<div class='sidebar-item'>{role} {msg['message'][:40]}...</div>", unsafe_allow_html=True)
    st.markdown(f"<br><hr>🔌 <small>Connected to: `{BASE_URL}`</small>", unsafe_allow_html=True)

# ---------------- Navbar ----------------
st.markdown("<div class='nav'>🧠 RAG Chatbot</div>", unsafe_allow_html=True)

# ---------------- Login / Register Flow ----------------
if "token" not in st.session_state:
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
                    st.success("✅ Authenticated!")
                    st.session_state.token = res.json()["access_token"]
                    st.session_state.username = email
                    st.rerun()
                else:
                    st.error(res.text)
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend.")
    st.stop()

# ---------------- Upload ----------------
st.subheader("📤 Upload Document")
with st.form("upload_form"):
    file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    upload = st.form_submit_button("Upload")
    if upload and file:
        files = {"file": (file.name, file.read())}
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        try:
            res = requests.post(f"{BASE_URL}/upload", files=files, headers=headers)
            st.success("✅ File uploaded!") if res.ok else st.error(f"❌ Upload failed: {res.text}")
        except requests.exceptions.RequestException:
            st.error("❌ Backend unreachable.")

# ---------------- Chat UI ----------------
st.subheader("💬 Ask me anything")
question = st.chat_input("Type your question...")

if question:
    st.session_state.history.append({"role": "user", "message": question})
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(st.session_state.history, f)

    with st.chat_message("user", avatar="🧑"):
        st.markdown(f"<div class='chat-message user'>{question}</div>", unsafe_allow_html=True)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            try:
                res = requests.post(f"{BASE_URL}/chat", data={"question": question}, headers=headers)
                if res.ok:
                    data = res.json()
                    answer = data.get("answer", "No answer returned.")
                    st.session_state.history.append({"role": "assistant", "message": answer})
                    with open(CHAT_HISTORY_FILE, "w") as f:
                        json.dump(st.session_state.history, f)
                    st.markdown(f"<div class='chat-message'>{answer}</div>", unsafe_allow_html=True)
                    confidence = data.get("confidence")
                    if confidence is not None:
                        confidence_percent = round(confidence * 100)
                        st.markdown(
                            f"<div class='confidence-bar' style='background:linear-gradient(to right, #4caf50 {confidence_percent}%, #ccc {confidence_percent}%);'></div>",
                            unsafe_allow_html=True
                        )
                    if data.get("sources"):
                        st.caption("📄 Sources: " + ", ".join(f"Page {doc.get('page_label', '?')}" for doc in data["sources"]))
                else:
                    st.error("❌ Failed to get a response.")
            except requests.exceptions.RequestException:
                st.error("❌ Backend API is unreachable.")

# ---------------- Footer ----------------
st.markdown("<div class='footer'>© 2025 RAG Agent | Built by Amine </div>", unsafe_allow_html=True)
