import streamlit as st
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from legal_agents.hybrid_agent import initialize_hybrid_agent
from vectorstore.init_vector_db import initialize_vector_db
from datetime import datetime

st.set_page_config(page_title="Karnataka Legal Assistant", layout="wide")

def main():
    st.title("Karnataka Legal Assistant")
    
    # Display indexed PDFs
    indexed_pdfs = [f for f in os.listdir("./legal_documents") if f.endswith(".pdf")]
    if indexed_pdfs:
        st.write(f"Indexed PDFs: {', '.join(indexed_pdfs)}")
    else:
        st.write("No PDFs indexed yet.")
    
    # Initialize agent
    if "agent" not in st.session_state:
        st.session_state.agent = initialize_hybrid_agent()
    
    # Language selection
    language = st.selectbox("Select Language", ["English", "Kannada"])
    
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
            with st.spinner("Processing..."):
                result = st.session_state.agent.process_query(user_query, language)
                st.write(f"**Answer:** {result['answer']}")
                st.write("**Sources:**")
                for source in result["sources"]:
                    st.write(f"- {source}")
        else:
            st.error("Please enter a question.")
    
    # Feedback
    st.subheader("Feedback")
    feedback = st.slider("Rate this answer (1-5)", 1, 5)
    comment = st.text_area("Comments", placeholder="Any additional feedback?")
    if st.button("Submit Feedback"):
        with open("feedback.txt", "a") as f:
            f.write(f"{datetime.now()}: Query: {user_query}, Rating: {feedback}, Comment: {comment}\n")
        st.success("Feedback submitted!")

if __name__ == "__main__":
    main()