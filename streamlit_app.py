import streamlit as st
import requests
import os
from uuid import uuid4
import uuid
import time
from datetime import datetime
from datetime import timedelta
from cassandra.cluster import Cluster
import pandas as pd
import sys
import json,re
import fitz  # PyMuPDF
import asyncio
from pathlib import Path
try:
    from services.document_processor import process_law_pdf as _process_law_pdf
except Exception:
    _process_law_pdf = None  # fallback to cached loader only


CACHE_FILE = "/app/data/law_translation.json"

# Add backend/APP to sys.path so 'services' can be imported
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICES_DIR = os.path.join(BASE_DIR, "backend", "APP")
if SERVICES_DIR not in sys.path:
    sys.path.insert(0, SERVICES_DIR)

from services.llm_service import GeminiService
law_data_cache = None 

sys.path.append("APP")
from dotenv import load_dotenv
load_dotenv()

CASSANDRA_HOST = os.getenv("CASSANDRA_HOSTS", "cassandra")
CASSANDRA_PORT = int(os.getenv("CASSANDRA_PORT", "9042"))
KEYSPACE = os.getenv("CASSANDRA_KEYSPACE", "rag_keyspace")
cluster = Cluster(contact_points=[CASSANDRA_HOST], port=CASSANDRA_PORT)
cass_session = cluster.connect(KEYSPACE)



st.set_page_config(page_title="BRI Chat Assistant LAW", layout="wide")
st.markdown("""
    <style>
    .topnav {
        background-color: #0e1117;
        overflow: hidden;
        position: sticky;
        top: 0;
        width: 100%;
        z-index: 9999;
        padding: 0.75rem 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #333;
    }
    .topnav h1 {
        font-size: 1.3rem;
        color: #fff;
        margin: 0;
    }
    .topnav .user-info {
        font-size: 0.9rem;
        color: #bbb;
    }
    .topnav .admin-badge {
        background-color: #ff1744;
        color: white;
        font-size: 0.7rem;
        padding: 2px 6px;
        border-radius: 6px;
        margin-left: 10px;
    }
    .logout-button {
        color: #fff;
        background-color: #333;
        padding: 5px 12px;
        border-radius: 6px;
        border: none;
        cursor: pointer;
        margin-left: 20px;
    }
    </style>
""", unsafe_allow_html=True)


API_HOST = os.getenv("API_HOST", "ragagentbot")
API_PORT = os.getenv("API_PORT", "8000")
BASE_URL = f"http://{API_HOST}:{API_PORT}"

cass_session = cluster.connect(KEYSPACE)

# --- make backend/APP importable for Streamlit ---
ROOT = os.path.dirname(__file__)
APP_DIR = os.path.join(ROOT, "backend", "APP")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)


def auth_headers():
    """
    Returns a valid Authorization header for API requests.
    """
    token = st.session_state.get("token")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def get_legal_recommendations_gemini_strict(gemini, law_text_en: str, client: dict) -> dict:
    """
    Ask Gemini for STRICT JSON ONLY. Parse robustly and validate.
    """
    prompt = f"""
You are a compliance assistant for **Tunisian leasing** decisions.

Return **STRICT JSON ONLY**. Do not include any prose before or after the JSON.

Use this schema exactly:
{{
  "eligible": true,
  "summary": "string (<=160 chars, plain text)",
  "reasons": ["string", "string"],
  "recommendations": [
    {{
      "label": "Option name",
      "upfront_needed_tnd": 0,
      "safe_car_value_tnd": 0,
      "monthly_payment_tnd": 0,
      "duration_months": 48
    }}
  ]
}}

Rules:
- Keep summary short and client-friendly.
- Reasons must be short compliance bullets (max 6–10 words).
- Numbers must be pure numbers (no currency symbols).
- NO markdown, NO tables, NO extra commentary.

CLIENT:
- Age: {client["age"]}
- Monthly Income (TND): {client["monthly_income"]}
- Existing Loans: {client["existing_loans"]}
- Employment Status: {client["employment_status"]}
- Requested Car Value (TND): {client["requested_car_value"]}
- Desired Duration (months): {client["desired_duration"]}
- Credit Score: {client["credit_score"]}

LAW CONTEXT (name only): Tunisian leasing framework (Law n°2016-48; Law n°94-89).
(Do not quote law text. Use it as domain context only.)
"""
    raw = asyncio.run(gemini.generate(prompt))
    text = raw if isinstance(raw, str) else str(raw)
    data = _extract_json(text)
    if not data:
        return {}  # let caller decide fallback
    return _validate_legal_payload(data)


def render_law_explanation(
    *,
    age,
    monthly_income,
    existing_loans,
    employment_status,
    requested_car_value,
    desired_duration,
    credit_score,
    translated_law_en: str
):
    """Renders a concise law-based explanation for the decision."""
    if not translated_law_en or not translated_law_en.strip():
        return

    try:
        gemini = GeminiService()
        prompt = f"""
You are a Tunisian leasing law assistant. Using ONLY the translated law text below (no external knowledge),
give a brief justification of the eligibility decision for this client. Write a short paragraph followed by
3 clear bullet points that reference specific law items if relevant (e.g., "Law 2016-48, Art. X - ...").

CLIENT
- Age: {age}
- Monthly Income (TND): {monthly_income}
- Existing Loans: {existing_loans}
- Employment Status: {employment_status}
- Requested Car Value (TND): {requested_car_value}
- Desired Duration (months): {desired_duration}
- Credit Score: {credit_score}

TRANSLATED LAW (English):
{translated_law_en[:8000]}
"""
        text = asyncio.run(gemini.generate(prompt))
        text = text if isinstance(text, str) else str(text)
        st.subheader("📚 Law Explanation")
        st.markdown(text, unsafe_allow_html=True)
    except Exception as e:
        st.caption(f"Law explanation unavailable right now: {e}")


# --- Professional top bar ----------------------------------------------------
def render_topbar():
    st.markdown(
        """
        <style>
        .topbar {
            position: sticky; top: 0; z-index: 1000;
            backdrop-filter: blur(8px);
            background: rgba(14,17,23,0.85);
            border-bottom: 1px solid rgba(255,255,255,0.07);
            padding: 10px 14px; margin: -1rem -1rem 0 -1rem;
        }
        .topbar-row { display: flex; align-items: center; justify-content: space-between; }
        .brand { font-weight: 600; letter-spacing: .2px; }
        .user-chip {
            display: inline-flex; align-items: center; gap: 10px; padding: 6px 10px;
            border-radius: 999px; border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.04);
        }
        .role-badge {
            font-size: .75rem; padding: 2px 8px; border-radius: 999px;
            background: #334155; color: #e2e8f0; border: 1px solid rgba(148,163,184,.3);
        }
        .avatar {
            width: 28px; height: 28px; border-radius: 50%;
            background: linear-gradient(135deg,#64748b,#0ea5e9);
            display: inline-block;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    with st.container():
        st.markdown('<div class="topbar"><div class="topbar-row">', unsafe_allow_html=True)
        c1, c2 = st.columns([1,1], gap="large")
        with c1:
            st.markdown('<div class="brand">🧠 RAG Chatbot</div>', unsafe_allow_html=True)
        with c2:
            # Right side: user chip + menu
            u = st.session_state.get("username") or "Guest"
            role = st.session_state.get("role", "user")
            col_a, col_b = st.columns([3,2])
            with col_a:
                st.markdown(
                    f'<div class="user-chip"><span class="avatar"></span>'
                    f'<span>{u}</span><span class="role-badge">{role}</span></div>',
                    unsafe_allow_html=True
                )
            with col_b:
                # Clean, professional dropdown using st.popover (Streamlit 1.31+)
                with st.popover("Account ▾", use_container_width=True):
                    st.write("**Account**")
                    st.caption(f"Signed in as: `{u}`")
                    st.divider()
                    if st.button("Switch account (Logout)", type="secondary", use_container_width=True):
                        st.session_state["show_logout_confirm"] = True
        st.markdown('</div></div>', unsafe_allow_html=True)

    # Confirm dialog (lightweight)
    if st.session_state.get("show_logout_confirm"):
        with st.container(border=True):
            st.markdown("### Confirm logout")
            st.write("You will be signed out and returned to the login screen.")
            c_ok, c_cancel = st.columns(2)
            with c_ok:
                if st.button("Logout now", type="primary", use_container_width=True):
                    st.session_state.pop("show_logout_confirm", None)
                    do_logout()
            with c_cancel:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.pop("show_logout_confirm", None)

# Render top bar on every page
render_topbar()


def _extract_json(text: str) -> dict:
    """
    Try hard to extract a single JSON object from any LLM text.
    Returns {} if not found or invalid.
    """
    if not text:
        return {}
    # 1) direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) find the largest {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        blob = text[start:end+1]
        try:
            return json.loads(blob)
        except Exception:
            # 3) last resort: strip code fences/backticks and retry simple patterns
            blob = re.sub(r"```(json)?", "", blob).strip()
            try:
                return json.loads(blob)
            except Exception:
                return {}
    return {}

# --- Logout / session helpers -----------------------------------------------
def _keep_keys_from_session(keep_keys=("dark_mode",)):
    keep = {k: st.session_state.get(k) for k in keep_keys if k in st.session_state}
    st.session_state.clear()
    for k, v in keep.items():
        st.session_state[k] = v

def revoke_backend_token():
    """Try to invalidate the token on the API (non-blocking)."""
    try:
        requests.post(f"{BASE_URL}/logout", headers=auth_headers(), timeout=5)
    except Exception:
        pass  # never block UI

def do_logout():
    # 1) server-side revoke (best effort)
    revoke_backend_token()

    # 2) client-side wipe everything except theme preference
    keep = {"dark_mode": st.session_state.get("dark_mode", True)}
    st.session_state.clear()
    st.session_state.update(keep)

    # 3) rerun to show the login screen
    st.rerun()



def _is_amine_user() -> bool:
    u = (st.session_state.get("username") or "").strip().lower()
    # if login uses an email, compare the local-part
    if "@" in u:
        u = u.split("@", 1)[0]
    return u == "amine"


def _validate_legal_payload(data: dict) -> dict:
    """
    Ensure required keys exist with reasonable defaults.
    """
    safe = {
        "eligible": bool(data.get("eligible", False)),
        "summary": str(data.get("summary", "")).strip(),
        "reasons": data.get("reasons") if isinstance(data.get("reasons"), list) else [],
        "recommendations": data.get("recommendations") if isinstance(data.get("recommendations"), list) else [],
    }

    # Normalize recommendation items
    norm = []
    for rec in safe["recommendations"]:
        if not isinstance(rec, dict): 
            continue
        norm.append({
            "label": str(rec.get("label", "Option")).strip(),
            "upfront_needed_tnd": float(rec.get("upfront_needed_tnd", 0) or 0),
            "safe_car_value_tnd": float(rec.get("safe_car_value_tnd", 0) or 0),
            "monthly_payment_tnd": float(rec.get("monthly_payment_tnd", 0) or 0),
            "duration_months": int(rec.get("duration_months", 0) or 0),
        })
    safe["recommendations"] = norm
    return safe

def _fallback_recommendations(monthly_income: float, requested_car_value: float, desired_duration: int):
    """
    Generate sane options when LLM output is unusable.
    Uses a 40% income allocation rule-of-thumb.
    """
    max_pay = max(0.0, monthly_income * 0.40)
    # Current plan estimate (rough)
    est_pay = requested_car_value / max(12, desired_duration)
    # Option A: keep duration, compute upfront needed to fit max_pay
    afford_value = max_pay * desired_duration
    upfront_needed = max(0.0, requested_car_value - afford_value)

    return {
        "eligible": est_pay <= max_pay,
        "summary": "Decision based on internal affordability guardrail (40% income rule).",
        "reasons": (["Debt-to-income within threshold."] if est_pay <= max_pay else
                    ["Estimated payment exceeds 40% of monthly income."]),
        "recommendations": [
            {
                "label": "Current Request (Estimate)",
                "upfront_needed_tnd": 0.0,
                "safe_car_value_tnd": requested_car_value,
                "monthly_payment_tnd": round(est_pay, 2),
                "duration_months": desired_duration
            },
            {
                "label": "Fit 40% Income (Same Duration)",
                "upfront_needed_tnd": round(upfront_needed, 2),
                "safe_car_value_tnd": round(afford_value, 2),
                "monthly_payment_tnd": round(min(est_pay, max_pay), 2),
                "duration_months": desired_duration
            },
            {
                "label": "Extend to Lower Payment (+12 mo)",
                "upfront_needed_tnd": 0.0,
                "safe_car_value_tnd": requested_car_value,
                "monthly_payment_tnd": round(requested_car_value / (desired_duration + 12), 2),
                "duration_months": desired_duration + 12
            },
        ]
    }
def render_legal_decision_panel(
    *, eligible, summary, score, monthly_payment, interest_rate, duration_months,
    reasons, recommendations, law_excerpt=""
):
    st.markdown("## 📜 Legal Decision & Recommendations")
    left, right = st.columns([1,3])
    with left:
        st.markdown(
            f"<div style='padding:10px;border-radius:10px;"
            f"background:{'#13381b' if eligible else '#3a0e0e'};color:white;text-align:center;font-weight:600;'>"
            f"{'✅ Eligible' if eligible else '❌ Not Eligible'}</div>",
            unsafe_allow_html=True
        )
    with right:
        st.markdown(f"**Summary:** {summary}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Credit Score", f"{score}")
    m2.metric("Monthly Payment", f"{monthly_payment:,.2f} TND")
    m3.metric("Interest Rate", f"{interest_rate*100:.2f}%")
    m4.metric("Duration", f"{max(1, int(round(duration_months/12)))} yrs")


    if reasons:
        st.markdown("### ⚠️ Reasons")
        st.markdown("\n".join([f"- {r}" for r in reasons]))

    if recommendations:
        st.markdown("### 📊 Options to Present")
        df = pd.DataFrame(recommendations).rename(columns={
            "label":"Option","upfront_needed_tnd":"Upfront Needed (TND)",
            "safe_car_value_tnd":"Safe Car Value (TND)",
            "monthly_payment_tnd":"Monthly Payment (TND)",
            "duration_months":"Duration (mo)"
        })
        st.dataframe(
            df.style.format({
                "Upfront Needed (TND)":"{:,.2f}",
                "Safe Car Value (TND)":"{:,.2f}",
                "Monthly Payment (TND)":"{:,.2f}",
            }),
            use_container_width=True
        )

    if law_excerpt:
        with st.expander("📚 Law basis (excerpt)"):
            st.markdown(law_excerpt[:1000] + ("…" if len(law_excerpt) > 1000 else ""))


def get_bot_response(message: str) -> str:
    """
    Calls your FastAPI /chat endpoint and returns the assistant's answer.
    """
    try:
        res = requests.post(
            f"{BASE_URL}/chat",
            data={"question": message, "session_id": st.session_state.session_id},
            headers={"Authorization": f"Bearer {st.session_state.token}"},
            timeout=60
        )
        if res.ok:
            return res.json().get("answer", "No answer returned.")
        return f"⚠️ Error from backend: {res.text}"
    except Exception as e:
        return f"❌ Backend unreachable: {e}"

# Use your connector everywhere (avoid duplicate Cluster() connects)
def get_cassandra_session():
    """
    Returns a Cassandra session connected to the configured keyspace.
    """
    global cass_session
    return cass_session

# Small helper: try to load & cache the translated law text (non-blocking if missing)
@st.cache_data(show_spinner=False)
def try_load_translated_law(law_path: str):
    try:
        import fitz
        from services.llm_service import GeminiService

        doc = fitz.open(law_path)
        fr = "\n".join([p.get_text("text") for p in doc]).strip()
        gemini = GeminiService()
        eng = asyncio.run(gemini.generate(
            "Translate the following French legal text into clear, professional English:\n\n" + fr
        ))
        return {"french_text": fr, "english_text": str(eng).strip()}
    except Exception as e:
        return {"french_text": "", "english_text": "", "error": str(e)}

# --- Silent load of translated law (no UI) ---
LAW_FILE_PATH = "/app/data/loi_2016_48.pdf"

# Fast path: cached loader you already have
LAW = try_load_translated_law(LAW_FILE_PATH)  # returns {"english_text": "..."} in your code
TRANSLATED_LAW_EN = LAW.get("english_text", "") or LAW.get("translated_english", "") or ""

# Fallback once: only if cache was empty AND the async processor is available
if not TRANSLATED_LAW_EN and _process_law_pdf is not None:
    try:
        law_data = asyncio.run(_process_law_pdf(LAW_FILE_PATH))
        # tolerate either key name
        TRANSLATED_LAW_EN = (
            law_data.get("english_text", "") or
            law_data.get("translated_english", "") or
            ""
        )
    except Exception as e:
        # keep the app running even if translation fails
        TRANSLATED_LAW_EN = ""
        st.warning(f"Law translation unavailable right now: {e}")


def get_clients_df():
    """
    Returns leasing_clients as a pandas DataFrame (safe if empty).
    """
    try:
        rows = cass_session.execute("SELECT * FROM leasing_clients ALLOW FILTERING")
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"❌ Error loading leasing applications: {e}")
        return pd.DataFrame()

def get_client_match_from_text(message: str):
    """
    Tries to find a client by name or UUID mentioned in a message.
    """
    df = get_clients_df()
    if df.empty:
        return None

    msg = message.lower()
    # UUID hit
    for _, r in df.iterrows():
        if str(r.get("client_id", "")).lower() in msg:
            return r.to_dict()

    # Name hit (simple contains)
    for _, r in df.iterrows():
        name = str(r.get("full_name", "")).lower()
        if name and name in msg:
            return r.to_dict()

    return None

# ---------------- Loan Rules ----------------
LOAN_RULES = {
    "max_income_ratio": 0.4,  # Max % of income for monthly payment
    "min_credit_score": 50,
    "employment_weights": {
        "permanent": 25,
        "contract": 15,
        "self-employed": 10,
        "unemployed": 0
    },
    "duration_penalties": {
        12: 0,
        24: -5,
        36: -10,
        48: -15,
        60: -20,
        72: -30
    }
}


# ---------------- Process Law PDF ----------------
law_data_cache = None  # global cache

async def process_law_pdf(file_path: str):
    """
    Reads a French law PDF, translates it into English using Gemini,
    and stores both versions in memory + disk cache.
    """
    global law_data_cache

    # ✅ Check disk cache first
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            law_data_cache = json.load(f)
        return law_data_cache

    # ✅ Check memory cache
    if law_data_cache:
        return law_data_cache

    # --- Extract French text ---
    doc = fitz.open(file_path)
    french_text = "\n".join([page.get_text("text") for page in doc]).strip()

    # --- Translate with Gemini ---
    gemini = GeminiService()
    english_text = await gemini.generate(
        f"Translate the following French legal text into clear, professional English:\n\n{french_text}"
    )
    english_text = english_text.strip() if isinstance(english_text, str) else str(english_text)

    # --- Store in cache ---
    law_data_cache = {"french_text": french_text, "english_text": english_text}

    # ✅ Save to disk cache so we don’t reprocess on restart
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(law_data_cache, f, ensure_ascii=False, indent=2)

    return law_data_cache

# ---------------- Streamlit UI ----------------
# st.subheader("⚖️ Law File Processing")


# ---------------- Header with user info & logout ----------------
#if st.session_state.get("username"):
#    user_display = f"{st.session_state.username}"
#    is_admin = st.session_state.get("role") == "admin"
    
username = st.session_state.get("username")
role = st.session_state.get("role", "user")
user_display = username or "Guest"
is_admin = (role == "admin") if username else False



def fetch_client_info_from_message(message: str):
    rows = cass_session.execute("SELECT * FROM leasing_clients")
    for row in rows:
        if row.full_name.lower() in message.lower() or str(row.client_id) in message:
            return {
                "Name": row.full_name,
                "National ID": row.national_id,
                "Employment": row.employment_status,
                "Monthly Income": row.monthly_income,
                "Existing Loans": row.existing_loans,
                "Credit Score": row.credit_score,
                "Requested Car Value": row.requested_car_value,
                "Loan Duration (Months)": row.desired_duration_months,
            }
    return None



# ---------------- Initialize Session ----------------
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None
if "role" not in st.session_state:
    st.session_state.role = "user"
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
st.sidebar.title("⚙️ Settings")
if st.sidebar.button("🚪 Logout / Switch account", use_container_width=True):
    do_logout()
if "mode" not in st.session_state:
    st.session_state.mode = "chat"
if st.session_state.token:
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    nav_items = ["Dashboard", "Chat Assistant"]
    if _is_amine_user():
        nav_items.append("All Leasing Applications")
    menu = st.sidebar.radio("Navigation", nav_items)
    st.session_state["menu"] = menu
    
    if menu == "Dashboard":
        st.subheader("📊 Dashboard")
        try:
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
        st.stop()
    
    
    
    elif menu == "Chat Assistant":
        
        # ---------------- Upload ----------------
        # st.subheader("📤 Upload Document")
        # headers = {"Authorization": f"Bearer {st.session_state.token}"}
        # with st.form("upload_form"):
        #     file = st.file_uploader("📚 Select a document to upload", type=["pdf", "txt", "csv"])
        #     upload = st.form_submit_button("Upload")

        #     if upload:
        #         if file:
        #             with st.spinner("⏳ Uploading and processing your document..."):
        #                 try:
        #                     files = {"file": (file.name, file.read())}
        #                     res = requests.post(f"{BASE_URL}/upload", files=files, headers=headers, timeout=300)

        #                     if res.ok:
        #                         st.toast("✅ File uploaded and processing started!", icon='🎉')
        #                         st.session_state.last_uploaded_filename = file.name
                                
        #                     else:
        #                         error_detail = res.json().get("detail", "Upload failed.")
        #                         st.error(f"❌ Upload failed: {error_detail}")
        #                 except Exception as e:
        #                     st.error(f"❌ Backend unreachable: {str(e)}")
        #         else:
        #             st.warning("⚠️ Please select a file before uploading.")

        st.markdown("""
        <style>
        /* Dark app background + center container */
        .app-wrapper { max-width: 1100px; margin: 0 auto; }
        body { background:#0e1117 !important; }
        /* Hero headline (when empty) */
        .hero {
            text-align:center; padding: 11vh 0 5vh 0; color:#e5e7eb;
        }
        .hero h1 {
            font-size: 40px; font-weight: 700; letter-spacing:.2px; margin:0 0 18px 0;
        }
        .hero p { color:#94a3b8; margin:0; }
        /* Composer row like ChatGPT */
        .composer { position: sticky; bottom: 0; z-index: 20; padding: 16px 0 22px; background: linear-gradient(180deg, rgba(14,17,23,0) 0%, rgba(14,17,23,1) 26%); }
        .bar {
            display:flex; align-items:center; gap:10px;
            background:#1f2937; border:1px solid rgba(148,163,184,.2);
            border-radius: 999px; padding: 8px 10px; box-shadow: inset 0 1px 0 rgba(255,255,255,.03);
        }
        .plus {
            width:38px; height:38px; border-radius:999px; display:flex; align-items:center; justify-content:center;
            border:1px solid rgba(148,163,184,.25); background:#0b1324; color:#cbd5e1; font-size:18px; cursor:pointer;
        }
        .attach-chip {
            display:inline-flex; align-items:center; gap:.45rem; padding:.25rem .6rem;
            border-radius:999px; border:1px solid rgba(148,163,184,.25);
            background:#0b1324; color:#cbd5e1; font-size:.85rem;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="app-wrapper">', unsafe_allow_html=True)

        # --- Show a GPT-like hero if no messages yet ---
        if not st.session_state.history:
            st.markdown(
                '<div class="hero"><h1>What\'s on the agenda today?</h1>'
                '<p>Welcome Mr. {{ user.username }}</p></div>',
                unsafe_allow_html=True
            )

        # --- Render chat history (only here) ---
        if st.session_state.history:
            for item in st.session_state.history:
                if item["role"] == "user":
                    with st.chat_message("user", avatar="🧑‍💻"):
                        st.markdown(item["message"])
                elif item["role"] == "assistant":
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(item["message"], unsafe_allow_html=True)

        # --- Composer with ＋ attach (popover) + st.chat_input ---
        c_left, c_right = st.columns([0.12, 0.88], vertical_alignment="bottom")
        with c_left:
            with st.popover("＋", help="Attach a document", use_container_width=True):
                up = st.file_uploader("Attach a file (PDF/TXT/CSV)", type=["pdf","txt","csv"])
                if up is not None:
                    import requests
                    try:
                        files = {"file": (up.name, up.getvalue())}
                        r = requests.post(f"{BASE_URL}/upload", files=files, headers=auth_headers(), timeout=300)
                        if r.ok:
                            st.session_state.last_uploaded_filename = up.name
                            st.toast(f"✅ Attached: {up.name}", icon="📎")
                        else:
                            st.error(r.json().get("detail", r.text))
                    except Exception as e:
                        st.error(f"❌ Backend unreachable: {e}")

            if st.session_state.get("last_uploaded_filename"):
                st.markdown(f"<div class='attach-chip'>📎 {st.session_state.last_uploaded_filename}</div>", unsafe_allow_html=True)

        with c_right:
            question = st.chat_input("Ask anything…")

        # --- Send message ---
        if question:
            is_first = not st.session_state.session_title
            if is_first:
                st.session_state.session_title = question[:30] + "..." if len(question) > 30 else question
                try:
                    requests.post(
                        f"{BASE_URL}/register_session",
                        data={
                            "session_id": st.session_state.session_id,
                            "title": st.session_state.session_title,
                            "username": st.session_state.username
                        },
                        headers=auth_headers()
                    )
                    res = requests.get(f"{BASE_URL}/sessions?user={st.session_state.username}", headers=auth_headers())
                    if res.ok:
                        st.session_state.sessions = res.json().get("sessions", [])
                except Exception:
                    pass

            st.session_state.history.append({"role": "user", "message": question})
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(question)

            with st.chat_message("assistant", avatar="🤖"):
                placeholder = st.empty()
                with st.spinner("Thinking…"):
                    try:
                        res = requests.post(
                            f"{BASE_URL}/chat",
                            data={
                                "question": question,
                                "session_id": st.session_state.session_id,
                                "source_filename": st.session_state.get("last_uploaded_filename")
                            },
                            headers=auth_headers(),
                            timeout=60
                        )
                        if res.ok:
                            data = res.json()
                            answer = data.get("answer", "No answer returned.")
                            st.session_state.history.append({"role": "assistant", "message": answer})
                            placeholder.markdown(answer, unsafe_allow_html=True)
                        else:
                            placeholder.error(res.json().get("answer", "Error occurred."))
                    except Exception as e:
                        placeholder.error(f"❌ Backend unreachable: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

        # Stop here so nothing else (legacy) renders below
        st.stop()



    elif menu == "All Leasing Applications":
        if not _is_amine_user():
            st.error("⛔ Not authorized to view this page.")
            st.stop()

        st.subheader("📁 All Leasing Applications")
        try:
            rows = cass_session.execute("SELECT * FROM leasing_clients")
            st.dataframe(rows)
        except Exception as e:
            st.error(f"❌ Error loading leasing applications: {e}")

        # ---------------- Loan helpers ----------------
        def calculate_credit_score(age, income, has_loan, employment_status, car_price, duration_months):
            score = 0
            # Age
            if 25 <= age <= 45:
                score += 20
            elif 18 <= age < 25 or 45 < age <= 60:
                score += 10
            else:
                score += 5
            # Employment stability
            score += LOAN_RULES["employment_weights"].get(employment_status.lower(), 0)
            # Existing loans
            if has_loan:
                score -= 15
            # Income allocation vs. rough estimate
            max_payment_allowed = income * LOAN_RULES["max_income_ratio"]
            if duration_months <= 0:
                est_monthly_payment = float("inf")
            else:
                est_monthly_payment = car_price / duration_months
            if est_monthly_payment > max_payment_allowed:
                score -= 30
            else:
                score += 15
            # Duration penalty (first bucket that fits)
            for max_months, penalty in LOAN_RULES["duration_penalties"].items():
                if duration_months <= max_months:
                    score += penalty
                    break
            return max(score, 0)

        def calculate_monthly_plan(car_value, duration_months, score):
            if duration_months <= 0:
                return 0.0, 0.0  # guard
            if score >= 80:
                interest_rate = 0.03
            elif score >= 65:
                interest_rate = 0.04
            else:
                interest_rate = 0.06
            monthly_interest = interest_rate / 12.0
            monthly_payment = (car_value * monthly_interest) / (1 - (1 + monthly_interest) ** -duration_months)
            return round(monthly_payment, 2), interest_rate

        def _compute_score_and_plan(age, income, existing_loans, employment_status, car_value, duration_months):
            score = calculate_credit_score(
                age=age,
                income=income,
                has_loan=bool(existing_loans),
                employment_status=employment_status,
                car_price=car_value,
                duration_months=duration_months
            )
            monthly_payment, applied_rate = calculate_monthly_plan(car_value, duration_months, score)
            band = "High" if score >= 80 else ("Medium" if score >= 65 else "Low")
            return score, band, monthly_payment, applied_rate

        def decide_loan_approval(credit_score: float, monthly_payment: float, income: float):
            """
            Approve if:
            - credit_score >= LOAN_RULES['min_credit_score']  (0..100 scale here)
            - monthly_payment / income <= LOAN_RULES['max_income_ratio']
            """
            reasons = []
            if not income or income <= 0:
                return False, ["Invalid income provided (must be > 0)"]
            if credit_score < LOAN_RULES["min_credit_score"]:
                reasons.append(f"Credit score below minimum ({credit_score:.0f} < {LOAN_RULES['min_credit_score']})")
            income_ratio = monthly_payment / income if income else 1.0
            if income_ratio > LOAN_RULES["max_income_ratio"]:
                reasons.append(f"Debt-to-income too high ({income_ratio:.2%} > {LOAN_RULES['max_income_ratio']:.0%})")
            return len(reasons) == 0, reasons

        # ---------------- Eligibility form ----------------
        st.markdown("---")
        with st.expander("🚗 Leasing Eligibility Form", expanded=True):
            with st.form("eligibility_form"):
                full_name = st.text_input("Full name")
                national_id = st.text_input("National ID")

                col1, col2, col3 = st.columns(3)
                with col1:
                    age = st.number_input("Age", min_value=18, max_value=75, value=30, step=1)
                with col2:
                    employment_status_label = st.selectbox(
                        "Employment status",
                        ["Permanent", "Contract", "Self-Employed", "Unemployed"],
                        index=0
                    )
                with col3:
                    existing_loans = st.checkbox("Has existing loans?")

                col4, col5, col6 = st.columns(3)
                with col4:
                    monthly_income = st.number_input("Monthly income (TND)", min_value=0.0, value=2000.0, step=100.0)
                with col5:
                    requested_car_value = st.number_input("Requested car value (TND)", min_value=0.0, value=50000.0, step=500.0)
                with col6:
                    desired_duration = st.selectbox("Duration (months)", [12, 24, 36, 48, 60, 72], index=3)

                submitted = st.form_submit_button("📊 Evaluate Eligibility")

            # map UI label -> rules key
            emp_map = {
                "Permanent": "permanent",
                "Contract": "contract",
                "Self-Employed": "self-employed",
                "Unemployed": "unemployed"
            }
            employment_status = emp_map[employment_status_label]

            if submitted:
                validation_errors = []
                if not full_name.strip():
                    validation_errors.append('full name')
                if not national_id.strip():
                    validation_errors.append('national ID')
                if monthly_income <= 0:
                    validation_errors.append('monthly income (must be > 0)')
                if requested_car_value <= 0:
                    validation_errors.append('requested car value (must be > 0)')
                if validation_errors:
                    st.error('Please complete the following before evaluating: ' + ', '.join(validation_errors))
                else:
                    # 1) Compute
                    score, band, monthly_payment, applied_rate = _compute_score_and_plan(
                        age, monthly_income, existing_loans, employment_status,
                        requested_car_value, desired_duration
                    )
    
                    # 2) Decide
                    approval, reasons = decide_loan_approval(score, monthly_payment, monthly_income)
    
                    # 3) Save to DB (adjust types if your schema differs)
                    try:
                        get_cassandra_session().execute(
                            """
                            INSERT INTO leasing_clients (
                                client_id, full_name, national_id, employment_status,
                                monthly_income, existing_loans, credit_score,
                                requested_car_value, desired_duration_months, created_at
                            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            """,
                            (
                                uuid.uuid4(), full_name, national_id, employment_status,
                                float(monthly_income), bool(existing_loans), int(score),
                                float(requested_car_value), int(desired_duration), datetime.utcnow()
                            )
                        )
                    except Exception as e:
                        # If your column existing_loans is INT, try int(existing_loans) instead of bool(existing_loans)
                        st.error(f"❌ Error saving application: {e}")
    
                    # 4) Strict legal recs (LLM) → panel (fallback if LLM fails)
                    gemini = GeminiService()
                    client_payload = {
                        "age": age,
                        "monthly_income": monthly_income,
                        "existing_loans": existing_loans,
                        "employment_status": employment_status,
                        "requested_car_value": requested_car_value,
                        "desired_duration": desired_duration,
                        "credit_score": score,
                    }
                    legal = get_legal_recommendations_gemini_strict(gemini, TRANSLATED_LAW_EN, client_payload)
                    if not legal or (not legal.get("summary") and not legal.get("recommendations")):
                        legal = _fallback_recommendations(monthly_income, requested_car_value, desired_duration)
    
                    render_legal_decision_panel(
                        eligible=bool(legal.get("eligible")),
                        summary=str(legal.get("summary", "")),
                        score=int(score),
                        monthly_payment=float(monthly_payment),
                        interest_rate=float(applied_rate),
                        duration_months=int(desired_duration),
                        reasons=list(legal.get("reasons", [])),
                        recommendations=list(legal.get("recommendations", [])),
                        law_excerpt=""
                    )
    
                    # 5) Short law explanation (uses the SINGLE top-level function)
                    render_law_explanation(
                        age=age,
                        monthly_income=monthly_income,
                        existing_loans=existing_loans,
                        employment_status=employment_status,
                        requested_car_value=requested_car_value,
                        desired_duration=desired_duration,
                        credit_score=score,
                        translated_law_en=TRANSLATED_LAW_EN
                    )
    
                    # 6) Quick schedule (approx)
                    payment_df = pd.DataFrame({
                        "Month": list(range(1, desired_duration + 1)),
                        "Payment (TND)": [monthly_payment] * desired_duration,
                        "Interest Rate (%)": [applied_rate * 100] * desired_duration,
                        "Remaining Balance (TND)": [
                            max(0, requested_car_value - (monthly_payment * m))
                            for m in range(desired_duration)
                        ],
                    })
                    st.subheader("📅 Payment Schedule (approx.)")
                    st.dataframe(
                        payment_df.style.format({
                            "Payment (TND)": "{:,.2f}",
                            "Interest Rate (%)": "{:.2f}%",
                            "Remaining Balance (TND)": "{:,.2f}",
                        }),
                        use_container_width=True
                    )
    
                    # 7) Final banner
                    if approval:
                        st.success("🎉 Client is eligible for leasing.")
                    else:
                        st.error("❌ Client is not eligible.")
                        if reasons:
                            st.markdown("**Reasons:**")
                            for r in reasons:
                                st.markdown(f"- {r}")
                        suggested_payment = monthly_income * LOAN_RULES["max_income_ratio"]
                        possible_durations = [m for m in [12, 24, 36, 48, 60, 72] if (requested_car_value / m) <= suggested_payment]
                        suggested_duration = min(possible_durations) if possible_durations else 72
                        upfront_needed = max(0.0, requested_car_value - (suggested_payment * desired_duration))
                        st.info(
                            f"💡 Suggestion: Pay **{upfront_needed:,.2f} TND** upfront or choose **{suggested_duration} months** to qualify."
                        )
    
                
                
        # ===== end facelift =====
        def render_law_explanation(*, age, monthly_income, existing_loans, employment_status,
                                requested_car_value, desired_duration, credit_score,
                                translated_law_en: str):
            """One-shot law narrative (no duplicates, robust to missing law)."""
            if not translated_law_en:
                return  # silently skip if we have no law text loaded

            try:
                gemini = GeminiService()
                prompt = f"""
                            You are an expert in Tunisian leasing law.
                            Here is the English translation of the law (context only, do not quote excessively):
                            {translated_law_en}

                            Client profile:
                            - Age: {age}
                            - Monthly Income: {monthly_income} TND
                            - Existing Loans: {existing_loans}
                            - Employment Status: {employment_status}
                            - Requested Car Value: {requested_car_value} TND
                            - Desired Duration: {desired_duration} months
                            - Credit Score: {credit_score}

                            Explain clearly, in 1 short paragraph, whether the client is eligible per this framework,
                            then provide a concise, readable table with:
                            - Required upfront amount (if any) to qualify,
                            - Safe maximum car value,
                            - Realistic monthly plan aligned with the law.

                            Keep it professional. No JSON. No code fences.
                            """
                text = asyncio.run(gemini.generate(prompt))
                text = text if isinstance(text, str) else str(text)

                st.subheader("📚 Law Explanation")
                st.markdown(text, unsafe_allow_html=True)
            except Exception as e:
                # Don't break the page if the LLM call fails
                st.caption(f"Law explanation unavailable right now: {e}")



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
    st.title("Welcome to AI Assistant 👋")
    with st.form("auth_form"):
        st.subheader("Login")
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
                    data = res.json()
                    st.session_state.token = data["access_token"]
                    st.session_state.username = data.get("the_user", email)
                    st.session_state.role = data.get("his_job", "user")   # "admin" or "user"
                else:
                    st.error(res.text)
            except:
                st.error("❌ Cannot connect to backend.")
    st.stop()

# ---------------- Fetch Sessions ----------------
try:
    res = requests.get(f"{BASE_URL}/sessions?user={st.session_state.username}", headers=auth_headers())
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

# st.subheader("💬 Ask a Question")

# if not st.session_state.waiting_for_response:
#     question = st.chat_input("Type your question...")

#     if question:
#         # ⬇️ Save the first question as title and register the session
#         is_first_message = not st.session_state.session_title
#         if is_first_message:
#             st.session_state.session_title = question[:30] + "..." if len(question) > 30 else question

#             try:
#                 requests.post(f"{BASE_URL}/register_session", data={
#                     "session_id": st.session_state.session_id,
#                     "title": st.session_state.session_title,
#                     "username": st.session_state.username
#                 }, headers=headers)

#                 # 🔁 Refresh sidebar sessions after saving the session
#                 res = requests.get(f"{BASE_URL}/sessions?user={st.session_state.username}", headers=auth_headers())
#                 if res.ok:
#                     st.session_state.sessions = res.json().get("sessions", [])
#                     st.rerun()  # ⬅️ Force rerun to reload sidebar with new session
#             except:
#                 st.warning("⚠️ Failed to register or refresh sessions.")

#         #regular chat flow
#         st.session_state.waiting_for_response = True
#         st.session_state.history.append({"role": "user", "message": question})

#         with st.chat_message("user", avatar="🧑‍💻"):
#             st.markdown(question)

#         with st.chat_message("assistant", avatar="🤖"):
#             placeholder = st.empty()
#             with st.spinner("🤖 Thinking..."):
#                 try:
#                     res = requests.post(f"{BASE_URL}/chat", data={
#                         "question": question,
#                         "session_id": st.session_state.session_id,
#                         "source_filename": st.session_state.get("last_uploaded_filename")
#                     }, headers=headers, timeout=60)
                    
#                     if res.ok:
#                         data = res.json()
#                         answer = data.get("answer", "No answer returned.")
#                         st.session_state.history.append({"role": "assistant", "message": answer})

#                         # ✅ Display full markdown (with sources)
#                         placeholder.markdown(answer, unsafe_allow_html=True)

#                     else:
#                         error_detail = res.json().get("answer", "Error occurred.")
#                         placeholder.error(error_detail)

#                 except Exception as e:
#                     placeholder.error(f"❌ Backend unreachable: {str(e)}")

#         st.session_state.waiting_for_response = False

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
        
    except Exception as e:
        st.error(f"❌ Error: {e}")
        



# ---------------- Leasing Applications Page ----------------
def leasing_applications_page():
    st.title("📂 All Leasing Applications")
    try:
        query = "SELECT * FROM leasing_clients ALLOW FILTERING"
        rows = cass_session.execute(query)
        client_df = pd.DataFrame(rows)
        if not client_df.empty:
            st.dataframe(
                client_df[[
                    "full_name", "employment_status", "monthly_income",
                    "credit_score", "requested_car_value",
                    "desired_duration_months", "created_at"
                ]].sort_values("created_at", ascending=False)
            )
        else:
            st.info("No client applications found.")
    except Exception as e:
        st.error(f"❌ Error loading leasing applications: {e}")

# ---------------- Helper: Fetch client for Chatbot ----------------
def get_client_info_by_name(name: str):
    try:
        query = "SELECT * FROM leasing_clients WHERE full_name = %s ALLOW FILTERING"
        row = cass_session.execute(query, (name,)).one()
        if row:
            return {
                "full_name": row.full_name,
                "employment_status": row.employment_status,
                "monthly_income": row.monthly_income,
                "credit_score": row.credit_score,
                "requested_car_value": row.requested_car_value,
                "desired_duration_months": row.desired_duration_months,
                "created_at": row.created_at
            }
    except Exception as e:
        st.error(f"⚠️ Error fetching client: {e}")
    return None
