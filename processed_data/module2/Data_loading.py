# Import necessary libraries
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter, SentenceSplitter
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb import PersistentClient
import torch
from transformers import AutoModel, AutoTokenizer
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import BaseSynthesizer

# ------------------------------
# SET UP HUGGING FACE EMBEDDINGS WITHOUT EXPLICIT MODEL/TOKENIZER
# ------------------------------
try:
    # Setting the embedding model directly without passing model and tokenizer
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    Settings.embed_model = HuggingFaceEmbedding(model_name=model_name)
    print(f"✅ Model {model_name} loaded successfully.")
except Exception as e:
    print(f"❌ Error loading embedding model: {e}")
    exit()

Settings.llm = None  # No LLM required

# ------------------------------
# LOAD DOCUMENTS
# ------------------------------
doc_path = "/Users/liteshperumalla/Desktop/Files/masters/Smart AI Tutor/Module 2"
try:
    reader = SimpleDirectoryReader(doc_path, required_exts=['.pptx', '.ipynb']).load_data()
    if not reader:
        print("❌ No documents found! Check the path and file extensions.")
        exit()
    print(f"✅ Loaded {len(reader)} documents from {doc_path}")
except Exception as e:
    print(f"❌ Error loading documents: {e}")
    exit()

# ------------------------------
# INITIALIZE CHROMADB VECTOR STORE
# ------------------------------
chroma_path = "./chroma_db"

try:
    chroma_client = PersistentClient(path=chroma_path)
    vector_store = ChromaVectorStore(chroma_client, collection_name="document_chunks")
    print("✅ ChromaDB initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing ChromaDB: {e}")
    exit()

# ------------------------------
# DOCUMENT PROCESSING PIPELINE
# ------------------------------
pipeline = IngestionPipeline(
    transformations=[
        TokenTextSplitter(),
        SentenceSplitter(chunk_size=500, chunk_overlap=0),
    ],
    vector_store=vector_store
)

# Run pipeline
try:
    nodes = pipeline.run(documents=reader)
    if not nodes:
        print("❌ No nodes were created. Check document parsing.")
        exit()
    print(f"✅ {len(nodes)} document nodes created and stored in ChromaDB.")
except Exception as e:
    print(f"❌ Error during ingestion pipeline: {e}")
    exit()

# ------------------------------
# CREATE VECTOR STORE INDEX
# ------------------------------
try:
    index = VectorStoreIndex(nodes, vector_store=vector_store)
    print("✅ Vector store index created successfully.")
except Exception as e:
    print(f"❌ Error creating VectorStoreIndex: {e}")
    exit()

# ------------------------------
# CREATE CHAT ENGINE
# ------------------------------
import re

# Function to preprocess the query
def preprocess_query(query):
    # Remove extra spaces and standardize case
    query = " ".join(query.split())  # Remove extra spaces
    query = query.strip().lower()  # Trim and convert to lowercase
    
    # Additional preprocessing can be added here, e.g., spelling correction, handling synonyms
    return query

# Preprocess the query before sending to the engine
query = "What is Python Indentation?"
processed_query = preprocess_query(query)

# Debug: Check the content of the processed query
print("Processed Query:", processed_query)

# Check if query is valid (e.g., not too short)
if len(processed_query) < 3:
    print("❌ Query is too short. Please provide a more detailed question.")
else:
    # Call the chat engine with the preprocessed query
    chat_engine = index.as_chat_engine()
    response = chat_engine.stream_chat(processed_query)

    # Check response before streaming
    print("Streaming Response:")
    if response.response_gen:
        for token in response.response_gen:
            print(token, end="")
    else:
        print("❌ No relevant results found.")

    # Optionally, print out the response at the end:
    print("\nFinal response:", response)

