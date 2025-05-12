from fastapi import FastAPI, HTTPException, Query, Depends, UploadFile, File, Form, Response, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse, RedirectResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from services.auth_service import AuthService, UserInDB, Token
from services.cassandra_connector import CassandraConnector
from rag_agent import RAGAgent
from services.auth_service import hash_password
from services.semantic_cache import SemanticCache  # Make sure this is imported
from utils.text_classification import guess_domain_from_text
from services.cassandra_connector import CassandraConnector
from nlp_processor import ChatCategorizer
from datetime import datetime
import uuid
from uuid import UUID
import logging
import os


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

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, current_user: UserInDB = Depends(auth_service.get_current_user)):
    return templates.TemplateResponse("interface.html", {"request": request, "user": current_user})

@app.post("/login", response_model=Token)
async def login_user(
    request: Request,  # 👈 To get IP & User-Agent
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


@app.post("/chat")
async def chat_post(
    request: Request,
    question: str = Form(...),
    session_id: str = Form(None),  # allow optional
    source_filename: str = Form(None),  # accept uploaded file name
    current_user: UserInDB = Depends(auth_service.get_current_user)
):
    rag_agent = RAGAgent()
    try:
        # ✅ Step 1: Try semantic cache first
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

        # ✅ Step 2: Fallback to RAG
        user_context = {
            "name": current_user.username,
            "role": current_user.his_job,
            "session_id": session_id,
            "source_filename": source_filename  # Very important
        }

        response_data = await rag_agent.generate_response(
            question,
            user_context
        )

        # ✅ Step 3: Store answer in cache
        await semantic_cache.store(question, response_data["answer"])

        # ✅ Step 4: Return clean result
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
    
    
    
#@app.post("/nlp/update")
#def update_categories():
#    categorizer.predict_and_save()
#    return {"message": "Categories updated successfully"}





