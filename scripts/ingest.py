"""Load data from data/ and re-index the RAG knowledge base. Run after adding/updating catalog, trucks, or FAQs."""
import sys
from pathlib import Path

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from rag import ingest

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data"
    n = ingest(data_dir=data_dir)
    print(f"Ingest complete. Indexed {n} document chunks.")
