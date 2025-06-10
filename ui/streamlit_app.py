import streamlit as st
from agents.hybrid_agent import initialize_hybrid_agent
from data_ingestion.load_docs import load_and_chunk_documents
from vectorstore.init_vector_db import initialize_vector_db
from gtts import gTTS
import os
from pathlib import Path

st.set_page_config(page_title="Karnataka Legal Assistant", layout="wide")

def main():
    st.title("Karnataka Legal Assistant")
    
    # Initialize agent
    agent = initialize_hybrid_agent()
    
    # Language selection
    language = st.selectbox("Select Language", ["English", "Kannada"])
    lang_code = "en" if language == "English" else "kn"
    
    # Document upload
    uploaded_file = st.file_uploader("Upload Legal Document (PDF)", type="pdf")
    if uploaded_file:
        data_dir = "./legal_documents"
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        initialize_vector_db(data_dir)
        st.success("Document uploaded and indexed!")
    
    # Chat interface
    st.subheader("Ask a Legal Question")
    user_query = st.text_input("Your Question", placeholder="What are Karnataka land laws?")
    
    if st.button("Submit"):
        if user_query:
            result = agent.invoke({"query": user_query, "language": lang_code})
            st.write(f"**Answer:** {result['answer']}")
            st.write(f"**Sources:** {', '.join(result['sources'])}")
            
            # Voice output
            if st.checkbox("Enable Voice Output"):
                tts = gTTS(text=result['answer'], lang=lang_code)
                tts.save("output.mp3")
                st.audio("output.mp3")
    
    # Feedback
    feedback = st.slider("Rate this answer (1-5)", 1, 5)
    if st.button("Submit Feedback"):
        with open("feedback.txt", "a") as f:
            f.write(f"Query: {user_query}, Rating: {feedback}\n")
        st.success("Feedback submitted!")

if __name__ == "__main__":
    main()