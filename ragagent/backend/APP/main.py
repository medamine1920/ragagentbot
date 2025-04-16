from fastapi import FastAPI, HTTPException, Query, Depends, UploadFile, File, Form, Response, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse, RedirectResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from services.auth_service import AuthService, UserInDB, Token
from services.cassandra_connector import CassandraConnector
from rag_agent import RAGAgent
from services.auth_service import hash_password

import logging
import os

# Setup DB + Auth
db = CassandraConnector()
auth_service = AuthService(db)
rag_agent = RAGAgent()

# App initialization
app = FastAPI(title="RAG Agent Bot", version="1.1")
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

logger = logging.getLogger("ragagent")
logging.basicConfig(level=logging.INFO)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, current_user: UserInDB = Depends(auth_service.get_current_user)):
    return templates.TemplateResponse("interface.html", {"request": request, "user": current_user})

@app.post("/login", response_model=Token)
async def login_user(
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends()
):
    token = await auth_service.login_for_access_token(response, form_data)
    # 👇 Return the token in the JSON so Postman can grab it
    return token

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
    domain: str = Form(...),
    current_user: UserInDB = Depends(auth_service.get_current_user)
):
    content = await file.read()
    filename = file.filename
    user_context = {
        "username": current_user.username,
        "email": current_user.email
    }
    success = await rag_agent.process_document(content, filename, domain, user_context)
    return {"success": success}


@app.post("/chat")
async def chat_post(
    request: Request,
    question: str = Form(...),
    current_user: UserInDB = Depends(auth_service.get_current_user)
):
    try:
        response_data = await rag_agent.generate_response(
            question,
            {"name": current_user.username, "role": current_user.his_job}
        )
        return {
            "question": question,
            "answer": response_data["answer"],
            "sources": response_data.get("sources", []),
            "confidence": response_data.get("confidence"),
            "source_type": response_data.get("source_type")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    
    
@app.get("/debug")
async def debug_uploaded_chunks():
    try:
        retriever = rag_agent.astra_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        results = retriever.get_relevant_documents("test")  # use a dummy query
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
    except Exception as e:
        return {"error": str(e)}

