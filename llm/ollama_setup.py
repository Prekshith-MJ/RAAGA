from langchain.llms import Ollama

def initialize_llm(model_name: str = "navarasa"):
    """
    Initialize Navarasa 2.0 LLM via Ollama.
    """
    llm = Ollama(model=model_name, base_url="http://localhost:11434")
    return llm

if __name__ == "__main__":
    llm = initialize_llm()
    test_prompt = "Hello, how can I assist with legal queries in Karnataka?"
    response = llm(test_prompt)
    print(f"LLM Response: {response}")