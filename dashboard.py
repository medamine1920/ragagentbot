import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Prevent memory issues

from dotenv import load_dotenv
load_dotenv()  # ⬅️ Load .env variables

import streamlit as st
import pandas as pd
from cassandra.cluster import Cluster
from datetime import datetime, timedelta
from io import BytesIO
import plotly.express as px

# ✅ Read connection info from .env
CASSANDRA_HOST = os.getenv("CASSANDRA_HOSTS", "cassandra")
CASSANDRA_PORT = int(os.getenv("CASSANDRA_PORT", "9042"))
KEYSPACE = os.getenv("CASSANDRA_KEYSPACE", "rag_keyspace")

# ✅ Connect to Cassandra
cluster = Cluster(contact_points=[CASSANDRA_HOST], port=CASSANDRA_PORT)
session = cluster.connect(KEYSPACE)

# ------------------ Streamlit UI ------------------

st.set_page_config(page_title="📊 RAG Dashboard", layout="wide")
st.title("📊 Chat & Upload Dashboard")

# Date range filter
st.sidebar.header("📅 Date Filter")
days = st.sidebar.slider("Show data from the last N days", min_value=1, max_value=30, value=7)
since_date = datetime.utcnow() - timedelta(days=days)

# ------------------ Uploaded Documents ------------------
st.subheader("📄 Uploaded Documents")
try:
    query = """
        SELECT doc_id, filename, uploaded_by, domain, content, timestamp
        FROM documents
        WHERE timestamp >= %s ALLOW FILTERING
    """
    rows = session.execute(query, (since_date,))
    df = pd.DataFrame(rows, columns=["doc_id", "filename", "uploaded_by", "domain", "content", "timestamp"])

    if df.empty:
        st.info("No documents uploaded in the selected date range.")
    else:
        st.metric("🗂️ Total Uploads", len(df))
        st.metric("👤 Unique Uploaders", df['uploaded_by'].nunique())
        st.dataframe(df[["filename", "uploaded_by", "domain", "timestamp"]].sort_values(by="timestamp", ascending=False))

        # CSV export
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV", data=csv, file_name="uploads.csv", mime="text/csv")

except Exception as e:
    st.error(f"❌ Error loading document data: {e}")

# ------------------ Chat Sessions ------------------
st.subheader("💬 Chat Sessions")
try:
    session_query = """
        SELECT session_id, username, title, timestamp
        FROM sessions
        WHERE timestamp >= %s ALLOW FILTERING
    """
    sessions = session.execute(session_query, (since_date,))
    s_df = pd.DataFrame(sessions, columns=["session_id", "username", "title", "timestamp"])

    if s_df.empty:
        st.info("No chat sessions found.")
    else:
        st.metric("💬 Total Sessions", len(s_df))
        st.metric("👥 Unique Users", s_df["username"].nunique())

        chart_df = s_df["username"].value_counts().rename_axis("User").reset_index(name="Session Count")
        st.bar_chart(chart_df.set_index("User"))

        st.dataframe(s_df[["username", "title", "timestamp"]].sort_values(by="timestamp", ascending=False))

except Exception as e:
    st.error(f"❌ Error loading session data: {e}")

# ------------------ Filters ------------------
if not df.empty:
    st.sidebar.markdown("### Filters")
    selected_domains = st.sidebar.multiselect("Filter by Domain", df["domain"].unique(), default=df["domain"].unique())
    selected_uploaders = st.sidebar.multiselect("Filter by Uploader", df["uploaded_by"].unique(), default=df["uploaded_by"].unique())
    df = df[df["domain"].isin(selected_domains) & df["uploaded_by"].isin(selected_uploaders)]

# ------------------ Pie Chart ------------------
if not df.empty:
    st.subheader("📊 Documents by Domain")
    domain_counts = df["domain"].value_counts().rename_axis("Domain").reset_index(name="Count")
    st.plotly_chart(px.pie(domain_counts, names="Domain", values="Count", title="Documents per Domain"), use_container_width=True)

# ------------------ Excel Export ------------------
excel_buffer = BytesIO()
with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
    if not df.empty:
        df.to_excel(writer, sheet_name="Documents", index=False)
    if not s_df.empty:
        s_df.to_excel(writer, sheet_name="Sessions", index=False)

st.download_button(
    label="📥 Export Dashboard to Excel",
    data=excel_buffer.getvalue(),
    file_name="dashboard_export.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Load categorized data
#rows = session.execute("SELECT question, response, category FROM chat_history WHERE timestamp >= %s ALLOW FILTERING", (since_date,))
#df = pd.DataFrame(rows, columns=["question", "response", "category"])

# Show chart
#if not df.empty:
#    st.subheader("🧠 Message Categories")
#    cat_counts = df['category'].value_counts().rename_axis('Category').reset_index(name='Count')
#    st.bar_chart(cat_counts.set_index('Category'))