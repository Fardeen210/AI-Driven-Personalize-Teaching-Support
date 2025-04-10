# module2/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the .env file at the project root.
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    print("Warning: .env file not found. Using default configuration.")

class Config:
    DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
    LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama3.1:latest")
    REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 120.0))
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    # PERSIST_DIR is stored as a path (using forward slashes for portability)
    PERSIST_DIR = Path(os.getenv("PERSIST_DIR", "processed_data/module2/persisted_index"))