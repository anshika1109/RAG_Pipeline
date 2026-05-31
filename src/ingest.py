"""
Step 2: Document Ingestion

This script:
1. Loads documents from the data/ folder (PDFs and text files)
2. Splits them into chunks using LangChain's text splitter
3. Prints the chunks so you can see what's happening

We'll add embedding + Pinecone storage in the next step.
"""
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_documents():
    """
    Load all documents from the data/ directory.
    Supports PDF and .txt files.
    """
    # Load PDFs
    pdf_loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )

    # Load text files
    txt_loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
    )

    documents = []
    documents.extend(pdf_loader.load())
    documents.extend(txt_loader.load())

    print(f"\n📄 Loaded {len(documents)} document pages/files")
    return documents


def chunk_documents(documents):
    """
    Split documents into smaller chunks.

    Why chunk?
    - LLMs have limited context windows
    - Smaller chunks = more precise retrieval
    - Overlap ensures we don't cut sentences in half

    RecursiveCharacterTextSplitter tries to split on:
    1. Paragraphs (\\n\\n)
    2. Sentences (\\n)
    3. Words (spaces)
    4. Characters (last resort)
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],  # Priority order for splitting
    )

    chunks = text_splitter.split_documents(documents)
    print(f"✂️  Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def main():
    # Step 1: Load documents
    print("=" * 50)
    print("DOCUMENT INGESTION")
    print("=" * 50)

    documents = load_documents()

    if not documents:
        print("\n⚠️  No documents found! Add PDFs or .txt files to the data/ folder.")
        return

    # Step 2: Chunk documents
    chunks = chunk_documents(documents)

    # Step 3: Preview some chunks
    print("\n" + "=" * 50)
    print("SAMPLE CHUNKS (first 3)")
    print("=" * 50)

    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i + 1} ---")
        print(f"Source: {chunk.metadata.get('source', 'unknown')}")
        print(f"Length: {len(chunk.page_content)} chars")
        print(f"Content: {chunk.page_content[:200]}...")
        print()

    # Step 4: Embed and store in Pinecone
    from embeddings import get_embedding_model, embed_chunks, store_in_pinecone

    print("=" * 50)
    print("EMBEDDING + PINECONE STORAGE")
    print("=" * 50)

    model = get_embedding_model()
    embeddings = embed_chunks(model, chunks)
    store_in_pinecone(chunks, embeddings)


if __name__ == "__main__":
    main()
