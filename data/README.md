# Knowledge base data

Add your real data here. The RAG system will load and index everything under this folder.

## Folder structure

- **mattress_catalog/** – Product catalog (CSV or JSON)
  - See `mattress_catalog/README.md` and the example schema.
- **trucks/** – Truck specifications (CSV or JSON)
  - See `trucks/README.md` and the example schema.
- **faqs_policies/** – FAQs, shipping, policies, upsell rules (Markdown or text)
  - One file per topic or one combined file; both work.

## After adding data

Run the ingest API to re-index:

```bash
curl -X POST http://localhost:8000/ingest
```

Or use the script:

```bash
python scripts/ingest.py
```
