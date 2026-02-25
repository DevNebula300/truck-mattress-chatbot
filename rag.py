"""RAG: load data, embed, store in Chroma, and answer with GPT-4o-mini."""
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings


def get_embeddings():
    return OpenAIEmbeddings(
        api_key=settings.openai_api_key or None,
        model=settings.embedding_model,
    )


def get_llm():
    return ChatOpenAI(
        api_key=settings.openai_api_key or None,
        model=settings.model_name,
        temperature=0.2,
        max_tokens=300,
    )


def get_vector_store(embeddings=None):
    embeddings = embeddings or get_embeddings()
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name="knowledge",
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_persist_dir),
    )


def load_documents_from_data_dir(data_dir: Path) -> list[Document]:
    """Load all supported files under data_dir into LangChain Documents."""
    documents: list[Document] = []
    data_dir = Path(data_dir)

    if not data_dir.exists():
        return documents

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for path in data_dir.rglob("*"):
        if path.is_dir():
            continue
        suf = path.suffix.lower()
        try:
            if suf == ".txt" or suf == ".md":
                text = path.read_text(encoding="utf-8", errors="replace")
                docs = text_splitter.create_documents([text], metadatas=[{"source": str(path)}])
                documents.extend(docs)
            elif suf == ".json":
                import json
                raw = path.read_text(encoding="utf-8", errors="replace")
                data = json.loads(raw)
                # Flatten to text for RAG (support list of objects or single object)
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        content = _dict_to_text(item)
                        if content:
                            documents.append(
                                Document(page_content=content, metadata={"source": str(path), "index": i})
                            )
                else:
                    content = _dict_to_text(data)
                    if content:
                        documents.append(Document(page_content=content, metadata={"source": str(path)}))
            elif suf == ".csv":
                import csv
                text = path.read_text(encoding="utf-8", errors="replace")
                reader = csv.DictReader(text.splitlines())
                rows = list(reader)
                for i, row in enumerate(rows):
                    content = _dict_to_text(row)
                    if content:
                        documents.append(
                            Document(page_content=content, metadata={"source": str(path), "row": i})
                        )
        except Exception as e:
            print(f"Warning: could not load {path}: {e}")

    return documents


def _dict_to_text(d: dict) -> str:
    parts = []
    for k, v in d.items():
        if v is None or v == "":
            continue
        parts.append(f"{k}: {v}")
    return "\n".join(parts)


def ingest(data_dir: Optional[Path] = None) -> int:
    """Load data from data_dir, chunk, embed, and store in Chroma. Returns doc count."""
    data_dir = data_dir or Path(__file__).parent / "data"
    docs = load_documents_from_data_dir(data_dir)
    if not docs:
        return 0
    embeddings = get_embeddings()
    store = get_vector_store(embeddings)
    try:
        store._client.delete_collection("knowledge")
    except Exception:
        pass
    store = get_vector_store(embeddings)
    store.add_documents(docs)
    return len(docs)


def answer(question: str, chat_history: list[dict] | None = None) -> str:
    """Answer using RAG: retrieve context from vector store, then GPT-4o-mini."""
    chat_history = chat_history or []
    embeddings = get_embeddings()
    store = get_vector_store(embeddings)
    retriever = store.as_retriever(search_kwargs={"k": 4})
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant for Estee Bedding Company. We sell four truck mattresses: Rest Stop, Long Haul, Dreamliner, and Heavy Hauler. Recommend ONE based on the customer's needs.
Use ONLY the retrieved context. Keep answers brief (2 to 4 sentences). Use simple formatting so key info stands out: put a short heading on one line starting with ## (e.g. ## Recommendation). Put mattress names and prices in double asterisks, e.g. **Rest Stop** or **$257.78**. Use normal punctuation. Only recommend these four models."""),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    response = chain.invoke(question)
    return response.content if hasattr(response, "content") else str(response)


def _format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)
