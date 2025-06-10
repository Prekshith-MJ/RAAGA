from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from typing import Dict, Any, TypedDict
from tools.web_search_tool import get_web_search_tool

def initialize_hybrid_agent():
    """
    Initialize the hybrid RAG agent with vector store and web search using LangGraph.
    """
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"}
    )
    
    # Initialize vector store
    vector_db = Chroma(
        collection_name="legal_docs",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    
    # Initialize LLM (Mistral)
    llm = Ollama(model="mistral", base_url="http://localhost:11434")
    
    # Define prompt template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="You are a legal assistant for rural Karnataka. Answer in simple terms (ELI5) in {language}. Context: {context}\nQuestion: {question}\nAnswer:"
    )
    
    # Define LangGraph state
    class AgentState(TypedDict):
        query: str
        language: str
        answer: str
        sources: list
    
    # Define nodes
    def retrieve_from_vector_db(state: AgentState) -> AgentState:
        docs = vector_db.similarity_search(state["query"], k=3)
        if docs and max(docs, key=lambda x: x.metadata.get("score", 0)).metadata.get("score", 0) > 0.7:
            context = "\n".join([doc.page_content for doc in docs])
            prompt_text = prompt.format(context=context, question=state["query"], language=state["language"])
            answer = llm.invoke(prompt_text)
            return {
                "query": state["query"],
                "language": state["language"],
                "answer": answer,
                "sources": [doc.metadata["source"] for doc in docs]
            }
        return {
            "query": state["query"],
            "language": state["language"],
            "answer": None,
            "sources": []
        }
    
    def web_search(state: AgentState) -> AgentState:
        if not state["answer"]:
            web_tool = get_web_search_tool()
            context = web_tool.func(state["query"])
            prompt_text = prompt.format(context=context, question=state["query"], language=state["language"])
            answer = llm.invoke(prompt_text)
            return {
                "query": state["query"],
                "language": state["language"],
                "answer": answer,
                "sources": ["web"]
            }
        return state
    
    # Build LangGraph workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_from_vector_db)
    workflow.add_node("web_search", web_search)
    workflow.add_edge("retrieve", "web_search")
    workflow.add_edge("web_search", END)
    workflow.set_entry_point("retrieve")
    
    agent = workflow.compile()
    return agent

if __name__ == "__main__":
    agent = initialize_hybrid_agent()
    result = agent.invoke({"query": "What are Karnataka land laws?", "language": "en"})
    print(f"Answer: {result['answer']}\nSources: {result['sources']}")