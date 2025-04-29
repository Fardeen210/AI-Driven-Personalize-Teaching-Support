# module2/context_retrieval.py
import argparse

from processed_data.module2.config import Config
from processed_data.module2.utils import setup_settings, ensure_persist_dir, get_storage_context, RAGQueryEngine
from llama_index.core import load_index_from_storage, get_response_synthesizer

def parse_args():
    parser = argparse.ArgumentParser(description="Smart AI Tutor CLI")
    subparsers = parser.add_subparsers(dest='command')
    
    ingestion_parser = subparsers.add_parser('ingest', help="Ingest data into the index")
    ingestion_parser.add_argument('data_path', type=str, help="Path to the data to ingest")
    
    query_parser = subparsers.add_parser('query', help="Query the RAG model")
    query_parser.add_argument('query_text', type=str, help="Query text for the RAG model")
    
    subparsers.add_parser('chat', help="Interactive chat with the AI tutor")
    return parser.parse_args()

def chat():
    print("Welcome to Smart AI Tutor! Type 'exit' to quit the chat.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        print(f"Running query: {user_input}")
        try:
            storage_context = get_storage_context(Config.PERSIST_DIR)
            index = load_index_from_storage(storage_context)
            print("Index loaded successfully.")
        except Exception as e:
            print(f"Error loading index: {e}")
            continue

        retriever = index.as_retriever()
        synthesizer = get_response_synthesizer(response_mode="compact")
        query_engine = RAGQueryEngine(retriever=retriever, response_synthesizer=synthesizer)
        response = query_engine.query(user_input)
        print("Assistant:", response)

def run_ingestion(data_path):
    print(f"Ingestion function not implemented. Data path provided: {data_path}")

def run_query(query_text):
    print(f"Query function not implemented. Query text provided: {query_text}")

def main():
    setup_settings(Config)
    ensure_persist_dir(Config.PERSIST_DIR)
    
    args = parse_args()
    if args.command == 'ingest':
        run_ingestion(args.data_path)
    elif args.command == 'query':
        run_query(args.query_text)
    elif args.command == 'chat':
        chat()
    else:
        print("Invalid command. Use -h for help.")

if __name__ == '__main__':
    main()  