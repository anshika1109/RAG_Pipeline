"""
Configuration settings for the RAG pipeline.
Supports both local (.env) and Streamlit Cloud (st.secrets) environments.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file (local dev)
load_dotenv()


def get_secret(key):
    """Get a secret from Streamlit Cloud or local .env"""
    # Try Streamlit secrets first (for deployed app)
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    # Fall back to environment variable (local dev)
    return os.getenv(key)


# API Keys
ANTHROPIC_API_KEY = get_secret("ANTHROPIC_API_KEY")
PINECONE_API_KEY = get_secret("PINECONE_API_KEY")

# Pinecone settings
PINECONE_INDEX_NAME = "rag-learning"  # Name of your Pinecone index

# Embedding settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Local model, 384 dimensions
EMBEDDING_DIMENSION = 384              # Must match the model's output size

# Chunking settings
CHUNK_SIZE = 500        # Number of characters per chunk
CHUNK_OVERLAP = 50      # Overlap between chunks to preserve context

# LLM settings
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# Data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
