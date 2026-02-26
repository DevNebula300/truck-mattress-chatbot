# Mattress & Truck RAG AI Agent

RAG-based chat agent that answers from your **mattress catalog**, **truck specifications**, and **FAQs/policies**. Uses **GPT-4o-mini** and an embeddable **chat widget** for your website.

## What you need to add later

When the client provides the data, add it here:

| Data | Location | Format |
|------|----------|--------|
| **Mattress catalog** | `data/mattress_catalog/` | CSV or JSON (models, sizes, pricing, materials, recommendations) |
| **Truck specs** | `data/trucks/` | CSV or JSON (makes, models, sleeper dimensions, fit notes) |
| **FAQs, policies, shipping, upsell** | `data/faqs_policies/` | Markdown or `.txt` |

See `data/README.md` and the README in each subfolder for schemas and examples. Placeholder sample files are included so the system runs before real data exists.

## Quick start

### 1. Environment

```bash
cp .env.example .env
```

Edit `.env` and set:

- **`OPENAI_API_KEY`** – Your OpenAI API key (required for chat and embeddings).
- **`API_KEY`** – (Optional) If set, callers must send this value in the `X-API-Key` header when calling `/chat` (and `/ingest`). Use a long random string and share it only with trusted frontends. Leave empty to allow unauthenticated requests.
- **`ALLOWED_ORIGINS`** – Your website origin(s), comma-separated (e.g. `https://your-site.com`), so the widget can call the API.

### 2. Install and run API

```bash
pip install -r requirements.txt
python main.py
```

API runs at **http://localhost:8000**.  
- **http://localhost:8000/docs** – Swagger UI.  
- **POST /chat** – Chat endpoint used by the widget.  
- **POST /ingest** – Re-index the knowledge base after adding/updating data.

### 3. Index the knowledge base

First time (and after any data changes):

```bash
python scripts/ingest.py
```

Or:

```bash
curl -X POST http://localhost:8000/ingest
```

### 4. Embed the chat widget on your site

**Option A – Same domain**  
Serve `widget/widget.js` from your site and add:

```html
<script
  src="/path/to/widget.js"
  data-api-base="https://your-api-domain.com"
  data-widget-title="Mattress & Truck Help"
></script>
```

**Option B – Local demo**  
Open `widget/index.html` in a browser (or serve it). Ensure `data-api-base` in the script tag points to your running API (e.g. `http://localhost:8000`).

**Widget attributes**

- `data-api-base` – Base URL of this API (no trailing slash).
- `data-widget-title` – Title shown in the chat header.

### Frontend integration (custom app)

If a separate frontend app will call this API, give the developer the base URL and (if you use one) the API key. Full details: **`FRONTEND-API.md`** — endpoint, request/response format, and a small code example.

## Project layout

```
matteres-ai-agent/
├── main.py           # FastAPI app: /chat, /ingest, /health
├── rag.py            # RAG: load data, embed, Chroma, GPT-4o-mini
├── config.py         # Settings from .env
├── requirements.txt
├── .env.example
├── data/             # Knowledge base (add your data here)
│   ├── mattress_catalog/
│   ├── trucks/
│   └── faqs_policies/
├── widget/           # Embeddable chat widget
│   ├── widget.js
│   └── index.html    # Demo page
├── scripts/
│   └── ingest.py     # Re-index from data/
└── chroma_db/        # Created on first ingest (vector store)
```

## When you get the real data

1. **Replace or add files** in `data/mattress_catalog/`, `data/trucks/`, and `data/faqs_policies/` (CSV, JSON, or Markdown as described in each folder’s README).
2. **Re-run ingest**: `python scripts/ingest.py` or `POST /ingest`.
3. Keep **OpenAI API key** and **ALLOWED_ORIGINS** (website URL) set in `.env` for the live site.

No code changes are required; the RAG pipeline picks up whatever you put in `data/`.
