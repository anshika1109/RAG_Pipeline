"""
Step 5: Interactive Chat Loop

A REPL (Read-Eval-Print Loop) that lets you ask multiple questions
without restarting the script each time.

The embedding model is loaded once at startup, then reused for every question.
"""
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


class RAGChatbot:
    """
    Encapsulates the RAG pipeline so we can reuse connections
    across multiple questions.
    """

    def __init__(self):
        print("🚀 Initializing RAG Chatbot...")

        # Load embedding model once (takes a few seconds)
        print("   Loading embedding model...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        # Connect to Pinecone once
        print("   Connecting to Pinecone...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = pc.Index(PINECONE_INDEX_NAME)

        # Initialize Claude
        print("   Setting up Claude...")
        self.llm = ChatAnthropic(
            model=CLAUDE_MODEL,
            api_key=ANTHROPIC_API_KEY,
            temperature=0,
        )

        self.prompt = ChatPromptTemplate.from_messages([
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

        self.chain = self.prompt | self.llm

        stats = self.index.describe_index_stats()
        print(f"\n✅ Ready! ({stats['total_vector_count']} vectors in index)")

    def ask(self, question, top_k=3, verbose=True):
        """
        Run the full RAG pipeline for a single question.
        """
        # Embed the question
        query_embedding = self.embedding_model.encode([question])[0]

        # Search Pinecone
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
        )
        matches = results["matches"]

        # Show retrieved chunks if verbose
        if verbose:
            print(f"\n🔍 Retrieved {len(matches)} chunks:")
            for i, match in enumerate(matches):
                print(f"   {i+1}. [Score: {match['score']:.3f}] {match['metadata']['text'][:80]}...")

        # Build context
        context_parts = []
        for i, match in enumerate(matches):
            context_parts.append(
                f"[Chunk {i+1} | Similarity: {match['score']:.3f}]\n{match['metadata']['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # Ask Claude
        response = self.chain.invoke({"context": context, "question": question})
        return response.content


def main():
    chatbot = RAGChatbot()

    print("\n" + "=" * 50)
    print("💬 RAG CHAT (type 'quit' to exit)")
    print("=" * 50)
    print("Ask questions about your documents.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! 👋")
            break

        answer = chatbot.ask(question)
        print(f"\n🤖 Claude: {answer}\n")


if __name__ == "__main__":
    main()
