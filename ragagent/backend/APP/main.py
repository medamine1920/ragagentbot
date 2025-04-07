from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path
from rag_agent import RAGAgent
import os


class ChatRequest(BaseModel):
    query: str
    
@app.post("/api/chat")
async def chat(request: ChatRequest):
    return await rag_agent.generate_response(request.query)
app = FastAPI()
rag_agent = RAGAgent()

# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# Replace the static files mount with:
#static_dir = Path("static")
#if static_dir.exists():
#    app.mount("/static", StaticFiles(directory="static"), name="static")
#else:
#    print("Warning: Static directory not found - skipping static files mount")

@app.on_event("startup")
async def startup():
    # Process any existing documents
    doc_dir = "documents"
    if os.path.exists(doc_dir):
        for filename in os.listdir(doc_dir):
            path = os.path.join(doc_dir, filename)
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    await rag_agent.process_document(f.read(), filename)

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    with open("templates/index.html") as f:
        return HTMLResponse(content=f.read())

#@app.post("/api/chat")
#async def chat(query: str):
#    return await rag_agent.generate_response(query)



@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if file.size > Config.MAX_DOCUMENT_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, "File too large")
    
    success = await rag_agent.process_document(await file.read(), file.filename)
    return {"status": "success" if success else "error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)