import streamlit as st
import requests

st.set_page_config(page_title="RAG Agent", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        .chat-message {
            padding: 10px;
            border-radius: 8px;
            margin: 5px 0;
        }
        .user { background-color: #4caf50; color: white; text-align: right; }
        .bot { background-color: #2c2c2c; color: white; text-align: left; }
    </style>
""", unsafe_allow_html=True)

# Sidebar: Chat history
with st.sidebar:
    st.title("📚 Chat History")
    if "history" not in st.session_state:
        st.session_state.history = []

    for i, entry in enumerate(st.session_state.history):
        st.button(f"🗨️ {entry[:20]}...", key=f"hist_{i}")

# Login/Register Flow
if "token" not in st.session_state:
    st.title("🧠 RAG Agent Chatbot")
    with st.form("auth_form"):
        st.write("### Login or Register")
        email = st.text_input("Username")
        password = st.text_input("Password", type="password")
        is_register = st.checkbox("Register instead?")
        submit = st.form_submit_button("Submit")

        if submit:
            endpoint = "http://ragagentbot:8000/register" if is_register else "http://ragagentbot:8000/login"
            data = {"username": email, "email": email, "password": password, "his_job": "user"}
            if not is_register:
                data = {"username": email, "password": password}
            res = requests.post(endpoint, data=data)

            if res.ok:
                st.success("✅ Authenticated!")
                st.session_state.token = res.json()["access_token"]
                st.rerun()
            else:
                st.error(res.text)
    st.stop()

# Upload & Chat Tabs
tab1, tab2 = st.tabs(["📄 Upload", "💬 Chat"])

with tab1:
    st.subheader("Upload a Document")
    with st.form("upload_form"):
        uploaded_file = st.file_uploader("Choose a PDF or document", type=["pdf", "txt"])
        domain = st.text_input("Domain (e.g., finance, legal, tech)")
        upload_btn = st.form_submit_button("Upload")

        if upload_btn and uploaded_file and domain:
            files = {"file": uploaded_file.getvalue()}
            data = {"domain": domain}
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            res = requests.post("http://ragagentbot:8000/upload", files=files, data=data, headers=headers)

            if res.ok:
                st.success("✅ File uploaded successfully.")
            else:
                st.error("❌ Upload failed.")

with tab2:
    st.subheader("Ask a Question")
    question = st.chat_input("Type your message...")
    if question:
        st.session_state.history.append(question)
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                headers = {"Authorization": f"Bearer {st.session_state.token}"}
                res = requests.post("http://ragagentbot:8000/chat", data={"question": question}, headers=headers)
                if res.ok:
                    answer = res.json()
                    st.markdown(answer["answer"], unsafe_allow_html=True)
                    st.caption(f"💬 Source pages: " + ", ".join(f"📄 Page {doc.get('page_label', '?')}" for doc in answer.get("sources", [])))
                else:
                    st.error("❌ Failed to get a response.")
