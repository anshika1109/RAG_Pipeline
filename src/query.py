"""
Step 4: Query Pipeline

This script:
1. Takes a user question
2. Embeds the question into a vector
3. Searches Pinecone for the most similar chunks
4. Sends the question + retrieved chunks to Claude
5. Returns a grounded answer

This is the "RAG" in action — Retrieval-Augmented Generation.
"""
import sys
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
)


def search_pinecone(query_embedding, top_k=3):
    """
    Search Pinecone for the most similar chunks to our query.

    top_k = how many results to return (3 is a good default for learning).
    Returns chunks ranked by cosine similarity.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True,  # We need the text back!
    )

    return results["matches"]


def build_context(matches):
    """
    Combine retrieved chunks into a single context string.
    This becomes the "knowledge" we give to Claude.
    """
    context_parts = []
    for i, match in enumerate(matches):
        score = match["score"]
        text = match["metadata"]["text"]
        context_parts.append(f"[Chunk {i+1} | Similarity: {score:.3f}]\n{text}")

    return "\n\n---\n\n".join(context_parts)


def ask_claude(question, context):
    """
    Send the question + retrieved context to Claude.

    The prompt template tells Claude to:
    - Only use the provided context
    - Admit if the context doesn't contain the answer
    - Cite which chunks it used
    """
    llm = ChatAnthropic(
        model=CLAUDE_MODEL,
        api_key=ANTHROPIC_API_KEY,
        temperature=0,  # Deterministic answers for consistency
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based ONLY on the provided context.

Rules:
- Only use information from the context below to answer
- If the context doesn't contain enough information, say so
- Keep answers concise and clear
- Mention which parts of the context support your answer"""),
        ("human", """Context:
{context}

Question: {question}

Answer:"""),
    ])

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content


def main():
    # Get the question from command line or use a default
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What is RAG and why does it matter?"

    print("=" * 50)
    print("RAG QUERY PIPELINE")
    print("=" * 50)
    print(f"\n❓ Question: {question}")

    # Step 1: Embed the question
    print("\n🧠 Embedding question...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode([question])[0]

    print(f"   Embedding dimension: {len(query_embedding)}")
    print(f"   First 5 values: {query_embedding[:5]}")
    print(f"   Min: {query_embedding.min():.4f} | Max: {query_embedding.max():.4f}")

    # Step 2: Search Pinecone for relevant chunks
    print("🔍 Searching Pinecone for relevant chunks...")
    matches = search_pinecone(query_embedding, top_k=3)

    print(f"\n📎 Retrieved {len(matches)} chunks:")
    for i, match in enumerate(matches):
        print(f"   {i+1}. Score: {match['score']:.3f} | {match['metadata']['text'][:80]}...")

    # Step 3: Build context from retrieved chunks
    context = build_context(matches)

    # Step 4: Ask Claude with the context
    print("\n🤖 Asking Claude...")
    answer = ask_claude(question, context)

    print("\n" + "=" * 50)
    print("ANSWER")
    print("=" * 50)
    print(f"\n{answer}")


if __name__ == "__main__":
    main()
