# module2/data_integration.py
from module2.config import Config
from module2.utils import setup_settings, ensure_persist_dir, get_storage_context, RAGQueryEngine
from llama_index.core import load_index_from_storage, get_response_synthesizer

def main():
    setup_settings(Config)
    ensure_persist_dir(Config.PERSIST_DIR)
    
    storage_context = get_storage_context(Config.PERSIST_DIR)
    try:
        index = load_index_from_storage(storage_context)
    except Exception as e:
        print(f"Error loading index: {e}")
        return

    retriever = index.as_retriever()
    synthesizer = get_response_synthesizer(response_mode="compact")
    query_engine = RAGQueryEngine(retriever=retriever, response_synthesizer=synthesizer)

    response = query_engine.query("What is python Indentation?")
    print(response)

if __name__ == '__main__':
    main()
