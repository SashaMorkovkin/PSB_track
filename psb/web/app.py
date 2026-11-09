"""Веб-приложение юридического ассистента. Использует тот же RAG-конвейер, что и сабмит."""
from __future__ import annotations
import os, zipfile
from pathlib import Path
from threading import Lock
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from solution import DATA_DIR, find_file, load_kb, Retriever, answer_question, BASE_URL, API_KEY

app=FastAPI(title="ПСБ — юридический ассистент", version="2.0.0")
_lock=Lock()
_state={"retriever":None,"chunks":None}
class ChatRequest(BaseModel):
    question:str=Field(min_length=2,max_length=5000)
    history:list[dict]=Field(default_factory=list)

def get_kb_path():
    return find_file("knowledge-base.zip","knowledge_base.zip","kb.zip")

def get_retriever():
    with _lock:
        if _state["retriever"] is None:
            chunks=load_kb(get_kb_path())
            retriever=Retriever(chunks)
            if BASE_URL and API_KEY:
                with httpx.Client(timeout=httpx.Timeout(120,connect=15)) as client: retriever.build_dense(client)
            _state.update(retriever=retriever,chunks=chunks)
        return _state["retriever"]

@app.get("/",response_class=HTMLResponse)
async def index():
    p=Path(__file__).parent/"static"/"index.html"
    if not p.exists(): raise HTTPException(500,"UI file missing")
    return HTMLResponse(p.read_text(encoding="utf-8"))

@app.get("/api/health")
async def health():
    return {"status":"ok","llm_configured":bool(BASE_URL and API_KEY)}

@app.get("/api/documents")
async def documents():
    try:
        with zipfile.ZipFile(get_kb_path()) as z:
            items=[]
            for name in sorted(z.namelist()):
                if name.endswith("/") or name.startswith("__MACOSX/"): continue
                if Path(name).suffix.lower() not in (".txt",".md",".docx"): continue
                items.append({"name":name,"title":Path(name).stem.replace("__"," — "),"size_kb":round(z.getinfo(name).file_size/1024,1)})
            return {"documents":items}
    except FileNotFoundError: return {"documents":[]}

@app.get("/api/document/{doc_name:path}")
async def document(doc_name:str):
    try:
        with zipfile.ZipFile(get_kb_path()) as z:
            if doc_name not in z.namelist(): raise HTTPException(404,"Документ не найден")
            raw=z.read(doc_name)
            if Path(doc_name).suffix.lower()==".docx":
                from solution import read_docx
                content=read_docx(raw)
            else:
                from solution import decode_text
                content=decode_text(raw,Path(doc_name).suffix.lower())
            return {"name":doc_name,"content":content}
    except FileNotFoundError: raise HTTPException(404,"База знаний не найдена")

def run_chat(question,history):
    if not BASE_URL or not API_KEY:
        raise HTTPException(503,"Не настроены LLM_BASE_URL и LLM_API_KEY. Добавьте их в .env.")
    retriever=get_retriever()
    with httpx.Client(timeout=httpx.Timeout(120,connect=15)) as client:
        return answer_question(client,retriever,question,history)

@app.post("/api/chat")
async def chat(req:ChatRequest):
    return await run_in_threadpool(run_chat,req.question,req.history)

if __name__=="__main__":
    import uvicorn
    uvicorn.run("web.app:app",host="0.0.0.0",port=8000)
