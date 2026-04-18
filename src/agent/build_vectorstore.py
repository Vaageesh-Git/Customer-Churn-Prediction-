"""
Run this script once to build the local Chroma vector store from the knowledge base PDFs.
Usage: python -m src.agent.build_vectorstore
"""

import os
from dotenv import load_dotenv

load_dotenv()

from src.agent.embedder import create_and_persist_vector_store

if __name__ == "__main__":
    use_cloud = os.getenv("USE_CHROMA_CLOUD", "false").lower() == "true"
    print(f"Building vector store ({'Chroma Cloud' if use_cloud else 'local'})...")

    kb_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "knowledge_base"
    )

    store = create_and_persist_vector_store(
        kb_path=kb_path,
        use_cloud=use_cloud,
        cloud_api_key=os.getenv("CHROMA_API_KEY") if use_cloud else None,
    )

    if store:
        print("Vector store built successfully.")
    else:
        print(
            "Failed to build vector store. Check that knowledge_base/ contains PDF files."
        )
