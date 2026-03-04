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

**Company history, who we are, where we're made**
→ Use the context: family owned since the 1920s, Chicago, garage to factory, Fortune 500 and truck/hospitality suppliers, largest truck mattress manufacturer in North America, Made to Ride (truck) and Hotel Sleeperzzz (hospitality), Union-made in Chicago. Mention timeline details (e.g. 1924, 1940 latex sleeper, 1989 Enright/ISO, 2016 Made to Ride) when relevant.

**Returns, refunds, wrong size, 100 Nights**
→ Use the returns policy context: 100 Nights' Free Sleep (100 days, refund), call 1-800-521-REST (7378) for RMA; wrong item/size — 20% restocking fee; return address Chicago; customer pays return shipping; sealed carton required.

**Shipping and delivery**
→ Use the shipping context: free UPS continental US, no P.O. Box/APO/FPO; orders ship within 5–7 business days; transit typically 2–5 business days; made to order in Chicago by Union workers; folded/rolled/vacuum sealed; tracking when shipped.

**Warranty**
→ Use the warranty context: 1 year limited, defects in materials/workmanship, original purchaser; body impressions are normal (not a defect); customer pays return shipping for warranty replacement; exclusions and void conditions in context; US only, non-transferable. Direct to Warranty page for full details.

**General FAQs** (when to replace mattress, bad back, fire resistant)
→ Use the FAQ context: replace every 5–7 years; bad back — support not hardest mattress, innerspring/foam/latex options; federal flammability rules since July 2007, fire-resistant barriers not chemicals, Estee meets standards.

## Sizing and product names (important)
- **We sell by dimensions, not by names like "Short Queen" or "Twin XL".** Always refer to sizes using the exact dimensions from the catalog (e.g. 36 x 79 x 7″, 42 x 80 x 8.5″). Do not recommend "a short queen" or "XL twin" as a product name.
- If the customer says their truck has a "short queen" or "twin XL" bunk, match that to our dimension sizes from the context (e.g. 36 x 75″, 36 x 79″, 42 x 80″) and say something like: "We sell by dimensions — the size that fits would be **36 x 79″** (or the dimension that matches their bunk)." Direct them to our Size Finder or exact measurements if needed.
- When recommending a mattress, give the **model name** (Rest Stop, Long Haul, Dreamliner, Heavy Hauler) and the **size in dimensions** (e.g. 36 x 79 x 7″).

## Shipping and delivery
- When asked about delivery time (e.g. "how long to get to LA if I order today"), use the shipping context below. Give a clear, helpful answer: when orders ship, typical transit time to their region, and suggest they’ll get tracking after order. If the context doesn’t have specific times, say delivery times vary by location and offer to help them place an order or check with the team for their zip code.

## Formatting rules
- Use **bold** for mattress names and prices (e.g. **Long Haul**, **$357.00**)
- Use ## headings only when presenting a clear recommendation or comparison — not for every response
- Keep greetings and short exchanges brief (1–3 sentences)
- Keep product responses clear but not overwhelming (3–6 sentences max)
- Never invent product details — only use information from the context provided

## Available mattresses
Rest Stop, Long Haul, Dreamliner, Heavy Hauler — details are in the context below. All Estee Bedding truck mattresses are made in the USA.
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
