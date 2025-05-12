import streamlit as st
import requests
import os
import uuid
import time
from datetime import datetime
from datetime import timedelta
from cassandra.cluster import Cluster
import pandas as pd


#from services.cassandra_connector import CassandraConnector

st.set_page_config(page_title="🧠 BRI Chat Assistant", layout="wide")

API_HOST = os.getenv("API_HOST", "ragagentbot")
API_PORT = os.getenv("API_PORT", "8000")
BASE_URL = f"http://{API_HOST}:{API_PORT}"

# ---------------- Header with user info & logout ----------------
if st.session_state.get("username"):
    with st.container():
        cols = st.columns([6, 1])
        cols[0].markdown(f"👤 **Logged in as:** `{st.session_state.username}`")
        if cols[1].button("🚪 Logout"):
            try:
                # Call backend to clear cookie if applicable
                res = requests.get(f"{BASE_URL}/logout", headers={"Authorization": f"Bearer {st.session_state.token}"})
            except:
                pass
            # Clear session
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

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
if "auto_loaded" not in st.session_state:
    st.session_state.auto_loaded = False
if "mode" not in st.session_state:
    st.session_state.mode = "chat"

mode = st.sidebar.radio("🧭 Navigation", ["Chat", "Dashboard"])
st.session_state.mode = "chat" if mode == "Chat" else "dashboard"


headers = {"Authorization": f"Bearer {st.session_state.token}"}

# ---------------- Theme Toggle + Styling ----------------
st.sidebar.title("⚙️ Settings")
st.sidebar.toggle("🌗 Dark Mode", key="dark_mode", value=True)

if st.session_state.get("last_uploaded_filename"):
    st.sidebar.caption(f"📄 Chatting with: `{st.session_state.last_uploaded_filename}`")


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

# ➕ New conversation button
if st.sidebar.button("➕ New Conversation"):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.history = []
    st.session_state.session_title = None
    st.session_state.last_uploaded_filename = None
    st.rerun()

# Grouping helper
today = datetime.utcnow().date()
yesterday = today - timedelta(days=1)

grouped_sessions = {
    "Today": [],
    "Yesterday": [],
    "Previous 7 Days": []
}

for session in st.session_state.sessions:
    try:
        ts = datetime.strptime(session["timestamp"], "%Y-%m-%d %H:%M:%S.%f")
    except:
        continue

    if ts.date() == today:
        grouped_sessions["Today"].append(session)
    elif ts.date() == yesterday:
        grouped_sessions["Yesterday"].append(session)
    elif ts.date() >= today - timedelta(days=7):
        grouped_sessions["Previous 7 Days"].append(session)

# Render each group
for label, sessions in grouped_sessions.items():
    if sessions:
        st.sidebar.markdown(f"**{label}**")
        for session in sorted(sessions, key=lambda x: x["timestamp"], reverse=True):
            if st.sidebar.button(session["title"], key=session["session_id"]):
                st.session_state.session_id = session["session_id"]
                st.session_state.session_title = session["title"]

                try:
                    res = requests.get(
                        f"{BASE_URL}/history?session_id={session['session_id']}",
                        headers=headers
                    )
                    if res.ok:
                        st.session_state.history = res.json().get("history", [])
                        st.rerun()
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
                        st.session_state.last_uploaded_filename = file.name
                        
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
        # ⬇️ Save the first question as title and register the session
        is_first_message = not st.session_state.session_title
        if is_first_message:
            st.session_state.session_title = question[:30] + "..." if len(question) > 30 else question

            try:
                requests.post(f"{BASE_URL}/register_session", data={
                    "session_id": st.session_state.session_id,
                    "title": st.session_state.session_title,
                    "username": st.session_state.username
                }, headers=headers)

                # 🔁 Refresh sidebar sessions after saving the session
                res = requests.get(f"{BASE_URL}/sessions?user={st.session_state.username}", headers=headers)
                if res.ok:
                    st.session_state.sessions = res.json().get("sessions", [])
                    st.rerun()  # ⬅️ Force rerun to reload sidebar with new session
            except:
                st.warning("⚠️ Failed to register or refresh sessions.")

        #regular chat flow
        st.session_state.waiting_for_response = True
        st.session_state.history.append({"role": "user", "message": question})

        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            with st.spinner("🤖 Thinking..."):
                try:
                    res = requests.post(f"{BASE_URL}/chat", data={
                        "question": question,
                        "session_id": st.session_state.session_id,
                        "source_filename": st.session_state.get("last_uploaded_filename")
                    }, headers=headers, timeout=60)
                    
                    if res.ok:
                        data = res.json()
                        answer = data.get("answer", "No answer returned.")
                        st.session_state.history.append({"role": "assistant", "message": answer})

                        # ✅ Display full markdown (with sources)
                        placeholder.markdown(answer, unsafe_allow_html=True)

                    else:
                        error_detail = res.json().get("answer", "Error occurred.")
                        placeholder.error(error_detail)

                except Exception as e:
                    placeholder.error(f"❌ Backend unreachable: {str(e)}")

        st.session_state.waiting_for_response = False

# ---------------- RENDER MAIN VIEW ----------------
if st.session_state.mode == "chat":
    # Your existing chat logic already defined above (nothing to change here)
    st.markdown("<div class='footer'>© 2025 RAG Agent | Built by Amine</div>", unsafe_allow_html=True)

elif st.session_state.mode == "dashboard":
    st.title("📊 Chat Dashboard")

    try:
        from dotenv import load_dotenv
        load_dotenv()
        CASSANDRA_HOST = os.getenv("CASSANDRA_HOSTS", "cassandra")
        CASSANDRA_PORT = int(os.getenv("CASSANDRA_PORT", "9042"))
        KEYSPACE = os.getenv("CASSANDRA_KEYSPACE", "rag_keyspace")
        cluster = Cluster(contact_points=[CASSANDRA_HOST], port=CASSANDRA_PORT)
        cass_session = cluster.connect(KEYSPACE)
        
        #cluster = Cluster(contact_points=["localhost"])
        #cass_session = cluster.connect("rag_keyspace")
        #db = CassandraConnector()
        #cass_session = db.session

        days = st.sidebar.slider("📅 Show data from last N days", min_value=1, max_value=30, value=7)
        since = datetime.utcnow() - timedelta(days=days)

        # Uploads
        st.subheader("📄 Uploaded Documents")
        query_docs = """
            SELECT doc_id, filename, uploaded_by, domain, content, timestamp
            FROM documents WHERE timestamp >= %s ALLOW FILTERING
        """
        rows = cass_session.execute(query_docs, (since,))
        doc_df = pd.DataFrame(rows, columns=["doc_id", "filename", "uploaded_by", "domain", "content", "timestamp"])

        if doc_df.empty:
            st.info("No uploads found.")
        else:
            st.metric("🗂️ Total Uploads", len(doc_df))
            st.metric("👤 Unique Uploaders", doc_df['uploaded_by'].nunique())
            st.dataframe(doc_df[["filename", "uploaded_by", "domain", "timestamp"]])

        # Sessions
        st.subheader("💬 Chat Sessions")
        query_sess = """
            SELECT session_id, username, title, timestamp
            FROM sessions WHERE timestamp >= %s ALLOW FILTERING
        """
        sessions = cass_session.execute(query_sess, (since,))
        sess_df = pd.DataFrame(sessions, columns=["session_id", "username", "title", "timestamp"])

        if sess_df.empty:
            st.info("No sessions found.")
        else:
            st.metric("💬 Total Sessions", len(sess_df))
            st.metric("👥 Active Users", sess_df['username'].nunique())

            count_df = sess_df['username'].value_counts().rename_axis("User").reset_index(name="Session Count")
            st.bar_chart(count_df.set_index("User"))
            st.dataframe(sess_df[["username", "title", "timestamp"]])

    except Exception as e:
        st.error(f"❌ Error loading dashboard: {e}")


