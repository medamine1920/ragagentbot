from fastapi import FastAPI, HTTPException, Query, Depends, UploadFile, File, Form, Response, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse, RedirectResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from services.auth_service import AuthService, UserInDB, Token, verify_token
from services.cassandra_connector import CassandraConnector
from rag_agent import RAGAgent
from services.auth_service import hash_password
from services.semantic_cache import SemanticCache  # Make sure this is imported
from utils.text_classification import guess_domain_from_text
from services.cassandra_connector import CassandraConnector
import re, uuid

from loan_eligibility import (calculate_credit_score,decide_loan_approval,)

from datetime import datetime
import uuid
from uuid import UUID,uuid4
import logging
import os
from pydantic import BaseModel
#from decision_engine import is_eligible, calculate_monthly_payment
#from loan_eligibility import calculate_monthly_plan
#from services.decision_engine import is_eligible, calculate_monthly_payment, calculate_credit_score
#from decision_engine import (is_eligible,calculate_monthly_payment,calculate_credit_score,)


# Setup DB + Auth
db = CassandraConnector()
auth_service = AuthService(db)
rag_agent = RAGAgent()
semantic_cache = SemanticCache(rag_agent.hf_embedding)  # Initialize with same embedder used in rag_agent

# App initialization
app = FastAPI(title="RAG Agent Bot", version="1.1")
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

logger = logging.getLogger("ragagent")
logging.basicConfig(level=logging.INFO)

#categorizer = ChatCategorizer()
# --- Compatibility helpers for credit scoring / plan ---

def _score_to_band(score: float) -> str:
    if score >= 700: return "Excellent"
    if score >= 650: return "Good"
    if score >= 600: return "Fair"
    if score >= 500: return "Weak"
    return "High Risk"

def _islamic_monthly_plan(amount: float, months: int, score: float):
    """
    Sharia-aligned profit rate tiers. Adjust the tiers if you already have bank-approved ones.
    """
    if months <= 0 or amount <= 0:
        return 0.0, 0.0
    if score >= 700:
        annual_profit = 0.03
    elif score >= 650:
        annual_profit = 0.04
    elif score >= 600:
        annual_profit = 0.05
    else:
        annual_profit = 0.06
    m = annual_profit / 12.0
    payment = (amount * m) / (1 - (1 + m) ** (-months))
    return round(payment, 2), annual_profit

def _compute_score_and_plan(age, monthly_income, existing_loans, employment_status,
                            requested_car_value, desired_duration):
    """
    Normalizes both versions of calculate_credit_score:
      - v1 returns float score
      - v2 returns (score, breakdown: {monthly_payment, profit_rate, ...}, band)
    Always returns: score, band, monthly_payment, applied_rate
    """
    try:
        out = calculate_credit_score(
            age, monthly_income, existing_loans, employment_status,
            requested_car_value, desired_duration
        )
    except TypeError:
        # Legacy keyword version
        out = calculate_credit_score(
            age=age, monthly_income=monthly_income, existing_loans=existing_loans,
            employment_status=employment_status, car_price=requested_car_value,
            duration_months=desired_duration
        )

    # v2: tuple (score, breakdown, band)
    if isinstance(out, tuple) and len(out) == 3:
        score, breakdown, band = out
        monthly_payment = float(breakdown.get("monthly_payment")
                                if breakdown.get("monthly_payment") is not None
                                else 0.0)
        applied_rate = float(breakdown.get("profit_rate")
                             if breakdown.get("profit_rate") is not None
                             else 0.0)
        # If monthly_payment not in breakdown, compute it:
        if monthly_payment <= 0.0:
            monthly_payment, applied_rate = _islamic_monthly_plan(
                requested_car_value, desired_duration, score
            )
        # If band missing, derive it:
        band = band or _score_to_band(score)
        return float(score), str(band), float(monthly_payment), float(applied_rate)

    # v1: just a score
    score = float(out)
    band = _score_to_band(score)
    monthly_payment, applied_rate = _islamic_monthly_plan(
        requested_car_value, desired_duration, score
    )
    return score, band, monthly_payment, applied_rate



@app.get("/", response_class=HTMLResponse)
async def home(request: Request, current_user: UserInDB = Depends(auth_service.get_current_user)):
    return templates.TemplateResponse("interface.html", {"request": request, "user": current_user})

@app.post("/login", response_model=Token)
async def login_user(
    request: Request,  # To get IP & User-Agent
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends()
):
    ip_address = request.client.host
    user_agent = request.headers.get("user-agent", "unknown")
    try:
        token = await auth_service.login_for_access_token(response, form_data)
        # ✅ Log successful login
        db.log_login_attempt(
            username=form_data.username,
            ip_address=ip_address,
            user_agent=user_agent,
            successful=True
        )
        return token
    except Exception as e:
        # ❌ Log failed login
        db.log_login_attempt(
            username=form_data.username,
            ip_address=ip_address,
            user_agent=user_agent,
            successful=False
        )
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/register")
async def register(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    his_job: str = Form(...),
):
    print(f"✅ Registering {username}")
    try:
        hashed_pw = hash_password(password)
        #db.insert_user(username, email, his_job, hashed_pw)
        # Insert user into Cassandra
        print(f"🔥 About to call insert_user with: {username}, {email}, {his_job}")
        user_id = db.insert_user(username=username, email=email, his_job=his_job, password=hashed_pw)
        print(f"✅ Registered user_id: {user_id}")


        
        # Determine if request came from Postman (JSON headers)
        if request.headers.get("accept") == "application/json":
            return JSONResponse(status_code=200, content={"message": "User registered successfully."})
        
        # Otherwise, render HTML template
        return templates.TemplateResponse("interface.html", {"request": request})

    except Exception as e:
        if request.headers.get("accept") == "application/json":
            return JSONResponse(status_code=500, content={"error": str(e)})
        return templates.TemplateResponse("interface.html", {"request": request, "error": str(e)})

@app.get("/logout")
async def logout(response: Response):
    await auth_service.logout(response)
    return RedirectResponse(url="/", status_code=302)

@app.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    current_user: UserInDB = Depends(auth_service.get_current_user)
):
    content = await file.read()
    filename = file.filename

    # 🧠 Guess domain if not passed explicitly
    raw_text = content.decode("utf-8", errors="ignore")[:1000]  # Limit for speed
    domain = guess_domain_from_text(raw_text)

    user_context = {
        "username": current_user.username,
        "email": current_user.email
    }

    success = await rag_agent.process_document(content, filename, domain, user_context)
    return {"success": success, "domain": domain}



def extract_national_id(question: str):
    import re
    match = re.search(r"\b\d{8}\b", question)
    return match.group(0) if match else None

UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b")
NID_RE  = re.compile(r"\b\d{8}\b")  # Tunisian CIN like 8 digits

def _extract_uuid(text: str) -> str | None:
    m = UUID_RE.search(text or "")
    return m.group(0) if m else None

def _extract_nid(text: str) -> str | None:
    m = NID_RE.search(text or "")
    return m.group(0) if m else None

def _row_to_dict(row):
    # works for both namedtuple Row and dict-like
    try:
        return row._asdict()
    except Exception:
        return dict(row)

@app.post("/chat")
async def chat_post(
    request: Request,
    question: str = Form(...),
    session_id: str = Form(None),
    source_filename: str = Form(None),
    current_user: UserInDB = Depends(auth_service.get_current_user)
):
    rag_agent = RAGAgent()

    try:
        # === Step 0: quick intent – eligibility by client id / national id
        q_lower = (question or "").lower()
        client_uuid_txt = _extract_uuid(question)
        national_id_txt = _extract_nid(question)

        if ("eligible" in q_lower or "eligibility" in q_lower) and (client_uuid_txt or national_id_txt):
            # --- fetch client by UUID first, else by national_id
            row = None
            if client_uuid_txt:
                try:
                    row = db.session.execute(
                        "SELECT * FROM leasing_clients WHERE client_id = %s",
                        (uuid.UUID(client_uuid_txt),)
                    ).one()
                except Exception as e:
                    logger.warning(f"UUID lookup failed: {e}")
            if (row is None) and national_id_txt:
                try:
                    row = db.session.execute(
                        "SELECT * FROM leasing_clients WHERE national_id = %s ALLOW FILTERING",
                        (national_id_txt,)
                    ).one()
                except Exception as e:
                    logger.warning(f"NID lookup failed: {e}")

            if row is None:
                return {
                    "question": question,
                    "answer": f"❌ No client found for "
                            f"{'client id ' + client_uuid_txt if client_uuid_txt else 'national id ' + national_id_txt}.",
                    "sources": [],
                    "confidence": 1.0,
                    "source_type": "rule"
                }

            r = _row_to_dict(row)

            # Required fields
            age = int(r.get("age") or r.get("client_age") or 0)
            monthly_income = float(r.get("monthly_income") or 0.0)
            existing_loans = bool(r.get("existing_loans") or False)
            employment_status = str(r.get("employment_status") or "")
            requested_car_value = float(r.get("requested_car_value") or 0.0)
            desired_duration = int(r.get("desired_duration_months") or 0)

            # --- unified scoring + plan (no calculate_monthly_plan; no undefined score/band)
            score, band, monthly_payment, applied_rate = _compute_score_and_plan(
                age, monthly_income, existing_loans, employment_status,
                requested_car_value, desired_duration
            )

            approval, reasons = decide_loan_approval(score, monthly_payment, monthly_income)
            total_cost = round(monthly_payment * desired_duration, 2)
            name = r.get("full_name") or "Client"

            answer = (
                f"{'✅ **Eligible for Leasing**' if approval else '❌ **Not Eligible for Leasing**'}\n\n"
                f"- **Name:** {name}\n"
                f"- **Client ID:** `{client_uuid_txt or r.get('client_id')}`\n"
                f"- **Credit Score:** `{score:.0f} ({band})`\n"
                f"- **Monthly Payment:** `{monthly_payment:,.2f} TND`\n"
                f"- **Duration:** `{desired_duration} months`\n"
                f"- **Total Cost:** `{total_cost:,.2f} TND`\n"
                + ("" if approval else ("\n**Reasons:**\n" + "\n".join([f"- {x}" for x in reasons])))
            )

            return {
                "question": question, "answer": answer, "sources": [],
                "confidence": 1.0, "source_type": "rule"
            }


        # === Step 1: semantic cache
        cached_answer = await semantic_cache.search(question)
        if cached_answer:
            logger.info(f"✅ Semantic Cache HIT for: {question[:50]}...")
            return {
                "question": question,
                "answer": cached_answer,
                "sources": [],
                "confidence": 1.0,
                "source_type": "cache"
            }

        # === Step 2: (optional) your older national-id "loan can i get" rule – keep if still needed
        if "loan" in q_lower and "can i get" in q_lower:
            nid = _extract_nid(question)
            if not nid:
                return {
                    "question": question,
                    "answer": "❗ Please provide your 8-digit national ID so I can check your loan eligibility.",
                    "sources": [],
                    "confidence": 1.0,
                    "source_type": "rule"
                }

            row = db.session.execute(
                "SELECT * FROM leasing_clients WHERE national_id = %s ALLOW FILTERING",
                (nid,)
            ).one()
            if not row:
                return {
                    "question": question,
                    "answer": f"❌ No client record found for ID {nid}.",
                    "sources": [],
                    "confidence": 1.0,
                    "source_type": "rule"
                }

            r = _row_to_dict(row)
            age = int(r.get("age") or r.get("client_age") or 0)
            monthly_income = float(r.get("monthly_income") or 0.0)
            existing_loans = bool(r.get("existing_loans") or False)
            employment_status = str(r.get("employment_status") or "")
            requested_car_value = float(r.get("requested_car_value") or 0.0)
            desired_duration = int(r.get("desired_duration_months") or 0)

            credit_score = calculate_credit_score(
                age=age, income=monthly_income, has_loan=existing_loans,
                employment_status=employment_status, car_price=requested_car_value,
                duration_months=desired_duration,
            )
            monthly_payment, applied_rate = calculate_monthly_plan(
                requested_car_value, desired_duration, credit_score
            )
            approval, reasons = decide_loan_approval(
                credit_score, monthly_payment, monthly_income
            )
            total = round(monthly_payment * desired_duration, 2)

            if approval:
                answer = f"""
✅ **Eligible for Leasing**

- **Name:** {r.get("full_name")}
- **Monthly Payment:** `{monthly_payment:,.2f} TND`
- **Duration:** `{desired_duration} months`
- **Total Cost:** `{total:,.2f} TND`
- **Decision basis:** {", ".join(reasons) or "Within income threshold"}
""".strip()
            else:
                answer = f"""
❌ **Not Eligible for Leasing**

- **Name:** {r.get("full_name")}
- **Reasons:** {", ".join(reasons) if reasons else "Does not meet guardrails"}
""".strip()

            return {
                "question": question,
                "answer": answer,
                "sources": [],
                "confidence": 1.0,
                "source_type": "rule"
            }

        # === Step 3: default to RAG
        user_context = {
            "name": current_user.username,
            "role": current_user.his_job,
            "session_id": session_id,
            "source_filename": source_filename
        }
        response_data = await rag_agent.generate_response(question, user_context)

        # === Step 4: cache the answer
        await semantic_cache.store(question, response_data["answer"])

        # === Step 5: return
        return {
            "question": question,
            "answer": response_data["answer"],
            "sources": response_data.get("sources", []),
            "confidence": float(response_data.get("confidence", 0.7)),
            "source_type": response_data.get("source_type", "llm")
        }

    except Exception as e:
        logger.error(f"❌ Error in chat_post: {e}")
        return {
            "question": question,
            "answer": "<div class='error'>Sorry, something went wrong.</div>",
            "sources": [],
            "confidence": 0.0,
            "source_type": "error"
        }
        
        


@app.get("/debug")
async def debug_uploaded_chunks():
    try:
        retriever = rag_agent.astra_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        results = retriever.get_relevant_documents("test")  # use a dummy query
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/history")
async def get_history(session_id: str, current_user: UserInDB = Depends(auth_service.get_current_user)):
    history = db.get_chat_history(session_id)
    return {"history": history}

@app.get("/sessions")
async def get_sessions(user: str = Query(...)):
    query = """
    SELECT session_id, title, timestamp FROM sessions WHERE username = %s ALLOW FILTERING
    """
    #rows = CassandraConnector.session.execute(query, (user,))
    db = CassandraConnector()
    rows = db.session.execute(query, (user,))

    sessions = []
    for row in rows:
        sessions.append({
            "session_id": str(row.session_id),
            "title": row.title,
            "timestamp": str(row.timestamp)
        })
    
    return {"sessions": sessions}


@app.post("/register_session")
async def register_session(
    session_id: str = Form(...),
    title: str = Form(...),
    username: str = Form(...)
):
    logger = logging.getLogger(__name__)
    
    timestamp = datetime.utcnow()

    try:
        query = """
        INSERT INTO sessions (session_id, title, username, timestamp)
        VALUES (%s, %s, %s, %s)
        """

        cassandra = CassandraConnector()
        cassandra.session.execute(query, (UUID(session_id), title, username, timestamp))
        logger.info(f"✅ Session saved for {username}: {title}")
        return {"message": "Session registered successfully"}

    except Exception as e:
        logger.error(f"❌ Failed to register session: {e}")
        return {"detail": f"Session registration failed: {str(e)}"}
    
    
    
class ClientInput(BaseModel):
    full_name: str
    national_id: str
    employment_status: str
    monthly_income: float
    existing_loans: bool
    credit_score: int
    requested_car_value: float
    desired_duration_months: int

@app.post("/add_client")
def add_client(data: ClientInput):
    client_id = uuid4()
    created_at = datetime.utcnow()
    query = """
    INSERT INTO leasing_clients (
        client_id, full_name, national_id, employment_status,
        monthly_income, existing_loans, credit_score,
        requested_car_value, desired_duration_months, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    db.session.execute(query, (
        client_id, data.full_name, data.national_id, data.employment_status,
        data.monthly_income, data.existing_loans, data.credit_score,
        data.requested_car_value, data.desired_duration_months, created_at
    ))
    return {"message": "Client added", "client_id": str(client_id)}



@app.get("/client_plan")
def client_plan(national_id: str, user: dict = Depends(verify_token)):
    query = "SELECT * FROM leasing_clients WHERE national_id = %s ALLOW FILTERING"
    row = db.session.execute(query, (national_id,)).one()

    if not row:
        raise HTTPException(status_code=404, detail="Client not found")

    if user["role"] != "admin":
        return {"message": "❌ Access restricted: only admin can view full client analysis."}

    eligible, reason = is_eligible(row)
    monthly_payment = calculate_monthly_payment(row.requested_car_value, row.desired_duration_months)

    return {
        "client": row.full_name,
        "eligible": eligible,
        "reason": reason,
        "credit_score": row.credit_score,
        "monthly_payment_estimate": monthly_payment,
        "car_value": row.requested_car_value,
        "duration": row.desired_duration_months
    }

#@app.post("/nlp/update")
#def update_categories():
#    categorizer.predict_and_save()
#    return {"message": "Categories updated successfully"}





