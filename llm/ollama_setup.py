from langchain_community.llms import Ollama

def setup_llm():
    llm = Ollama(model="mistral", base_url="http://localhost:11434")
    return llm

llm = setup_llm()