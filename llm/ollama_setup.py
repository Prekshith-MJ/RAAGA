from langchain_community.llms import Ollama
import logging

# Configure logging
logging.basicConfig(filename="ollama.log", level=logging.INFO, format="%(asctime)s - %(message)s")

def initialize_llm(model_name: str = "mistral"):
    """
    Initialize Mistral 7B LLM via Ollama.

    Args:
        model_name (str): The name of the model to use (default: "mistral").

    Returns:
        Ollama: An initialized Ollama LLM instance.

    Raises:
        RuntimeError: If the Ollama server is not running or the model fails to load.
    """
    try:
        logging.info(f"Initializing LLM with model: {model_name}")
        llm = Ollama(model=model_name, base_url="http://localhost:11434")
        # Test the connection by invoking a simple prompt
        llm.invoke("Test connection")
        logging.info(f"Successfully initialized LLM with model: {model_name}")
        return llm
    except Exception as e:
        logging.error(f"Failed to initialize LLM with model {model_name}: {str(e)}")
        raise RuntimeError(f"Failed to initialize LLM with model {model_name}: {str(e)}")