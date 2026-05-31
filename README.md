# RAG Pipeline with LangChain, Pinecone & Claude

A learning project to understand Retrieval-Augmented Generation (RAG).

## Stack
- **LangChain** — orchestration framework
- **Pinecone** — managed vector database
- **Claude (Anthropic)** — LLM for answer generation
- **sentence-transformers** — local embeddings

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and add your API keys:
   ```bash
   cp .env.example .env
   ```

4. Add some documents to the `data/` folder (PDFs or text files).

5. Run ingestion:
   ```bash
   python src/ingest.py
   ```

6. Query the pipeline:
   ```bash
   python src/query.py "Your question here"
   ```
