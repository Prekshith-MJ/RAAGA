from tavily import TavilyClient
from langchain.tools import Tool
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

load_dotenv()

def tavily_search(query: str, max_results: int = 5) -> str:
    """
    Perform web search using Tavily API.
    """
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = tavily.search(query=query, max_results=max_results)
    content = "\n".join([f"[{r['url']}]: {r['content']}" for r in results["results"]])
    return content

def duckduckgo_search(query: str, max_results: int = 5) -> str:
    """
    Fallback search using DuckDuckGo.
    """
    url = f"https://html.duckduckgo.com/html/?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    results = soup.find_all("a", class_="result__a", limit=max_results)
    content = "\n".join([r.text for r in results])
    return content

def get_web_search_tool():
    """
    Return a LangChain tool for web search.
    """
    return Tool(
        name="WebSearch",
        func=lambda query: tavily_search(query) if os.getenv("TAVILY_API_KEY") else duckduckgo_search(query),
        description="Search the web for up-to-date legal information."
    )

if __name__ == "__main__":
    query = "Karnataka land laws 2025"
    result = tavily_search(query) if os.getenv("TAVILY_API_KEY") else duckduckgo_search(query)
    print(f"Web Search Result: {result}")