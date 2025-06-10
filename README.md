Rural Karnataka Legal RAG-Based AI Agent
Overview
This project builds an AI agent to provide legal assistance for rural Karnataka residents, supporting English and Kannada. It uses a Retrieval-Augmented Generation (RAG) framework with local legal documents and live web search, running offline with Ollama and ChromaDB.
Setup

Install Dependencies:pip install -r requirements.txt


Install Ollama:
Follow instructions at Ollama to install and run Navarasa 2.0.
Run ollama pull navarasa to download the model.


Set Environment Variables:
Create a .env file with your Tavily API key:TAVILY_API_KEY=your_api_key




Prepare Legal Documents:
Place Karnataka legal PDFs in ./legal_documents/.


Initialize Vector DB:python vectorstore/init_vector_db.py


Run Streamlit App:streamlit run ui/streamlit_app.py



Usage

Access the app at http://localhost:8501.
Select language (English/Kannada), upload documents, and ask legal questions.
Enable voice output for audio responses.
Provide feedback to improve the system.

Features

Hybrid RAG + web search
Multilingual support (English, Kannada)
Document upload (PDFs)
ELI5-style legal explanations
Taluk/pincode personalization
Offline capability
Feedback loop

Directory Structure
legal-ai-rag/
├── data_ingestion/
│   └── load_docs.py
├── vectorstore/
│   └── init_vector_db.py
├── agents/
│   └── hybrid_agent.py
├── llm/
│   └── ollama_setup.py
├── tools/
│   └── web_search_tool.py
├── ui/
│   └── streamlit_app.py
├── requirements.txt
└── README.md

