"""
RAG Chatbot — Streamlit Web UI

A simple chat interface for the RAG pipeline.
Run with: streamlit run src/app.py
"""
import streamlit as st

# This MUST be the first Streamlit command
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load .env file for local dev
load_dotenv()

# Get keys from environment (works locally with .env)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "rag-learning"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CLAUDE_MODEL = "claude-sonnet-4-20250514"


@st.cache_resource
def load_embedding_model():
    """Load embedding model once and cache it across reruns."""
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def connect_pinecone():
    """Connect to Pinecone once and cache the connection."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)


def search_pinecone(index, query_embedding, top_k=3):
    """Search for similar chunks in Pinecone."""
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True,
    )
    return results["matches"]


def ask_claude(question, context):
    """Send question + context to Claude and get an answer."""
    llm = ChatAnthropic(
        model=CLAUDE_MODEL,
        api_key=ANTHROPIC_API_KEY,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based ONLY on the provided context.

Rules:
- Only use information from the context below to answer
- If the context doesn't contain enough information, say so
- Keep answers concise and clear"""),
        ("human", """Context:
{context}

Question: {question}

Answer:"""),
    ])

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content


# --- Streamlit UI ---

st.title("🤖 RAG Chatbot")
st.caption("Ask questions about your documents — powered by LangChain, Pinecone & Claude")

# Sidebar with settings
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Number of chunks to retrieve", 1, 10, 3)
    show_sources = st.checkbox("Show retrieved sources", value=True)

    st.divider()
    st.header("📊 Index Info")
    index = connect_pinecone()
    stats = index.describe_index_stats()
    st.metric("Vectors in index", stats["total_vector_count"])
    st.text(f"Model: {EMBEDDING_MODEL}")
    st.text(f"LLM: {CLAUDE_MODEL}")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📎 Sources"):
                for source in message["sources"]:
                    st.markdown(f"**Score: {source['score']:.3f}**")
                    st.text(source["text"][:200])
                    st.divider()

# Chat input
if question := st.chat_input("Ask a question about your documents..."):
    # Show user message
    st.chat_message("user").markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Load model and search
            model = load_embedding_model()
            query_embedding = model.encode([question])[0]

            matches = search_pinecone(index, query_embedding, top_k=top_k)

            # Build context
            context_parts = []
            sources = []
            for i, match in enumerate(matches):
                context_parts.append(match["metadata"]["text"])
                sources.append({
                    "score": match["score"],
                    "text": match["metadata"]["text"],
                })
            context = "\n\n---\n\n".join(context_parts)

            # Get answer from Claude
            answer = ask_claude(question, context)

        st.markdown(answer)

        # Show sources if enabled
        if show_sources:
            with st.expander("📎 Sources"):
                for source in sources:
                    st.markdown(f"**Similarity: {source['score']:.3f}**")
                    st.text(source["text"][:200])
                    st.divider()

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources if show_sources else [],
    })
