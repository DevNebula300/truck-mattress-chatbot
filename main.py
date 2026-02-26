"""FastAPI app: /chat for the widget, /ingest to re-index knowledge base."""
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import settings
from rag import answer, ingest


def require_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")):
    """If API_KEY is set in env, require it in the X-API-Key header. Otherwise allow all."""
    if not settings.api_key:
        return
    if not x_api_key or x_api_key.strip() != settings.api_key.strip():
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Vercel's deployment filesystem is read-only; ChromaDB needs write access.
    # Copy the committed chroma_db to /tmp (writable) on cold start.
    if os.environ.get("VERCEL"):
        src = Path(__file__).parent / "chroma_db"
        dst = Path("/tmp/chroma_db")
        if src.exists() and not dst.exists():
            shutil.copytree(str(src), str(dst))
        settings.chroma_persist_dir = dst
    yield


app = FastAPI(
    title="Mattress & Truck RAG API",
    description="RAG-powered chat for mattress catalog and truck specs. Embed the chat widget on your site.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve widget files so /test page can load the chat
_widget_dir = Path(__file__).parent / "widget"
if _widget_dir.exists():
    app.mount("/widget", StaticFiles(directory=str(_widget_dir), html=True), name="widget")


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []  # [{"role": "user"|"assistant", "content": "..."}]


class ChatResponse(BaseModel):
    reply: str


@app.get("/")
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/widget/test.html", status_code=302)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, _: None = Depends(require_api_key)):
    if not settings.openai_api_key:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured. Set OPENAI_API_KEY.")
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    try:
        reply = answer(req.message, chat_history=req.history)
        return ChatResponse(reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test")
def test_page_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/widget/test.html", status_code=302)


@app.post("/ingest")
def run_ingest(_: None = Depends(require_api_key)):
    """Re-load and index all data from the data/ folder. Call this after adding/updating catalog, trucks, FAQs."""
    data_dir = Path(__file__).parent / "data"
    try:
        count = ingest(data_dir=data_dir)
        return {"status": "ok", "documents_indexed": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=True,
    )
