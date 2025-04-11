# module2/store_in_chroma.py


import pickle
from pathlib import Path

from llama_index.vector_stores.chroma import ChromaVectorStore

from chromadb import PersistentClient
from llama_index.core import VectorStoreIndex

from ..module2.utils import ensure_persist_dir

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import ServiceContext
from tqdm import tqdm 
from llama_index.core import Settings

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
Settings.embed_model = embed_model

def main():
    # Load the parsed nodes from the previous step.
    try:
        with open("parsed_nodes.pkl", "rb") as f:
            nodes = pickle.load(f)
        print(f"✅ Loaded {len(nodes)} parsed nodes from file.")
    except Exception as e:
        print(f"❌ Error loading parsed nodes: {e}")
        return

    # Initialize a persistent Chroma client and collection.
    try:
        chroma_path = Path("processed_data\storage\chroma_db")
        chroma_client = PersistentClient(path=str(chroma_path))
        collection = chroma_client.get_or_create_collection("document_chunks")
        vector_store = ChromaVectorStore(chroma_client, collection_name="document_chunks")

        # Add each node to the collection.
        for i, node in tqdm(enumerate(nodes), total=len(nodes), desc="Storing & embedding nodes"):
            collection.add(
                ids=[str(i)],
                documents=[node.text],
                metadatas=[node.metadata]
            )
        print("✅ Nodes stored in ChromaDB.")
    except Exception as e:
        print(f"❌ Error storing nodes into ChromaDB: {e}")
        return

    # Create the vector store index from the nodes and persist it.
    try:
        index = VectorStoreIndex(nodes, vector_store=vector_store)
        persist_dir = ensure_persist_dir("persisted_index")
        index.storage_context.persist(persist_dir=str(persist_dir))
        print(f"✅ Vector store index created and persisted to {persist_dir}")
    except Exception as e:
        print(f"❌ Error creating or persisting VectorStoreIndex: {e}")
        return

if __name__ == '__main__':
    main()
