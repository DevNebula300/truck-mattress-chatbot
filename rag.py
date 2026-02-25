"""RAG: load data, embed, store in Chroma, and answer with GPT-4o-mini."""
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
        temperature=0.7,
        max_tokens=500,
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


SYSTEM_PROMPT = """You are Alex, a friendly and knowledgeable mattress consultant at Estee Bedding Company, specializing in truck mattresses for professional drivers.

## Your personality
- Warm, natural, and conversational — like a helpful friend who knows mattresses
- Patient: never rush to a recommendation before understanding what the customer needs
- Curious: ask follow-up questions to understand their situation better
- Honest: only use the product information provided in the context below

## How to handle different situations

**Greetings / casual openers** (hi, hello, hey, how are you, etc.)
→ Greet them warmly, introduce yourself briefly, and invite them to share what they're looking for. Do NOT mention products yet.

**General interest** ("I'm looking for a mattress", "what do you have?", "can you help me?")
→ Express enthusiasm, briefly mention you carry a range of truck mattresses, then ask 1–2 qualifying questions such as:
  - What truck do they drive?
  - How many hours do they typically sleep in the truck?
  - Do they prefer firmer or softer comfort?
  - Do they have a budget in mind?
Do NOT list all products immediately — have a conversation first.

**Specific needs or enough context given** (e.g. "I drive a Volvo and sleep 8 hours, I need firm support")
→ Use the product context to give a clear recommendation. Explain *why* that mattress suits them specifically.

**Asking about available products** ("what mattresses do you have?", "show me your options")
→ Give a brief overview of the available mattresses using the context, then ask what matters most to them so you can narrow it down.

**Follow-up questions or comparisons**
→ Answer fully and naturally using the context. Keep the conversation going.

**Small talk, jokes, off-topic**
→ Be friendly and engage briefly, then gently steer back to helping them find a mattress.

## Formatting rules
- Use **bold** for mattress names and prices (e.g. **Long Haul**, **$357.00**)
- Use ## headings only when presenting a clear recommendation or comparison — not for every response
- Keep greetings and short exchanges brief (1–3 sentences)
- Keep product responses clear but not overwhelming (3–6 sentences max)
- Never invent product details — only use information from the context provided

## Available mattresses
Rest Stop, Long Haul, Dreamliner, Heavy Hauler — details are in the context below.
"""


def answer(question: str, chat_history: list[dict] | None = None) -> str:
    """Answer using RAG with full conversation history and a consultative persona."""
    chat_history = chat_history or []

    embeddings = get_embeddings()
    store = get_vector_store(embeddings)
    llm = get_llm()

    # Retrieve relevant product context for the current question
    retriever = store.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)
    context = _format_docs(docs)

    # Convert chat history dicts to LangChain message objects
    history_messages = []
    for msg in chat_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            history_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            history_messages.append(AIMessage(content=content))

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT + "\n\n## Product context\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "history": history_messages,
        "question": question,
    })
    return response.content if hasattr(response, "content") else str(response)


def _format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)
