# module2/utils.py
from pathlib import Path

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, StorageContext, load_index_from_storage, get_response_synthesizer, PromptTemplate
from sentence_transformers import SentenceTransformer

def setup_settings(config):
    """Initialize embedding and LLM settings using configuration."""
    # Initialize embedding model (if needed directly)
    SentenceTransformer(config.EMBEDDING_MODEL)
    Settings.embed_model = HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL)
    Settings.llm = Ollama(model=config.LLAMA_MODEL, request_timeout=config.REQUEST_TIMEOUT)

def ensure_persist_dir(persist_dir):
    """Make sure the persist directory exists and return it as a Path object."""
    if not isinstance(persist_dir, Path):
        persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return persist_dir

def get_storage_context(persist_dir):
    """Return a storage context from a given persist directory."""
    return StorageContext.from_defaults(persist_dir=str(persist_dir))

# Common prompt template used across modules:
DEFAULT_PROMPT_TEMPLATE = (
    "Given the context information and not prior knowledge, "
    "you are a Teaching Assistant designed to assist users in answering queries. "
    "Explain concepts, solve coding doubts, and provide relevant resources from course modules. "
    "Also, provide a simple example to help the student understand the concept.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)

def get_prompt_template(template_str=None):
    if template_str is None:
        template_str = DEFAULT_PROMPT_TEMPLATE
    return PromptTemplate(template_str)

# A unified RAGQueryEngine that all modules can use.
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.retrievers import BaseRetriever

class RAGQueryEngine(CustomQueryEngine):
    """Custom query engine for retrieval-augmented generation."""
    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        # Collect text from nodes that have a get_text() method
        context_str = "\n".join([
            node.get_text() for node in nodes 
            if hasattr(node, 'get_text') and node.get_text()
        ])
        prompt_template = get_prompt_template()
        formatted_prompt = prompt_template.format(context_str=context_str, query_str=query_str)
        response_obj = self.response_synthesizer.synthesize(query=formatted_prompt, nodes=nodes)
        return response_obj

    def query(self, query_str: str):
        return self.custom_query(query_str)
