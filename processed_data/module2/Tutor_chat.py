# processed_data/module2/Tutor_chat.py
import os
import time
import streamlit as st

# Import application configuration and utilities.
# (Ensure your PYTHONPATH includes the parent folder so that these imports resolve correctly.)
from .config import Config
from .utils import setup_settings, ensure_persist_dir, get_storage_context, RAGQueryEngine
from llama_index.core import load_index_from_storage, get_response_synthesizer

# --- INITIALIZATION FUNCTIONS ---

def initialize_app():
    """
    Set up environment settings, ensure the persist directory exists,
    and perform any initialization logging.
    """
    setup_settings(Config)
    ensure_persist_dir(Config.PERSIST_DIR)
    st.write("Initialization complete.")

def load_index():
    """
    Attempt to load the persisted index from storage.
    Returns the index or None if loading fails.
    """
    try:
        storage_context = get_storage_context(Config.PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        return index
    except Exception as e:
        st.error(f"Error loading index: {e}")
        return None

# --- CHAT INTERFACE ---

def run_chat():
    """
    Main interactive chat interface using Streamlit.
    This uses the built-in st.chat_input and st.chat_message components (Streamlit 1.18+)
    for a native chat UI experience.
    """
    # Set page configuration before rendering any components.
    st.set_page_config(page_title="Smart AI Tutor", page_icon="🎓", layout="wide")
    st.title("🤖 Smart AI Tutor Chat")

    # Initialize conversation history if not already present.
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display previous messages using the new chat message containers.
    for entry in st.session_state.chat_history:
        if entry["role"] == "user":
            with st.chat_message("user"):
                st.markdown(entry["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(entry["content"])

    # Accept user input using st.chat_input (or fall back to st.text_input if unavailable).
    # (Note: st.chat_input is available in recent Streamlit versions.)
    user_input = st.chat_input("Ask your question...")

    if user_input:
        # Append the user message to session state and display it.
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Use a spinner while generating the tutor's response.
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                index = load_index()
                if index:
                    retriever = index.as_retriever()
                    synthesizer = get_response_synthesizer(response_mode="compact")
                    query_engine = RAGQueryEngine(retriever=retriever, response_synthesizer=synthesizer)
                    response = query_engine.query(user_input)
                    # Simulate streaming by revealing the response word by word.
                    response_placeholder = st.empty()
                    streamed_response = ""
                    for word in str(response).split():
                        streamed_response += word + " "
                        response_placeholder.markdown(streamed_response + "▌")
                        time.sleep(0.03)
                    response_placeholder.markdown(streamed_response.strip())
                    st.session_state.chat_history.append({"role": "assistant", "content": streamed_response.strip()})
                else:
                    st.error("Index could not be loaded. Please try again later.")

        # Rerun to update the interface with new messages.
        st.experimental_rerun()

if __name__ == "__main__":
    initialize_app()
    run_chat()
