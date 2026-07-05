# Import libraries
import streamlit as st
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import Chroma

# Page configuration
st.set_page_config(
    page_title="Multi PDF Chatbot",
    page_icon="📚"
)

st.title("📚 Multi PDF Chatbot")

# API key input (Make sure this is in your Streamlit secrets!)
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# 1. Initialize Session State for Memory and Vectorstore
if "messages" not in st.session_state:
    st.session_state.messages = [] # Stores chat history

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None # Prevents re-embedding PDFs on every chat

# Upload PDFs
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

# Process PDFs ONLY if they haven't been processed yet
if uploaded_files and st.session_state.vectorstore is None:
    with st.spinner("Processing PDFs..."):
        documents = []
        for uploaded_file in uploaded_files:
            # Save temporary PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            # Load PDF
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # Store source filename
            for doc in docs:
                doc.metadata["source_file"] = uploaded_file.name
            documents.extend(docs)

        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)

        # Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )
        
        # Vector DB saved to session state
        st.session_state.vectorstore = Chroma.from_documents(
            chunks,
            embeddings
        )
        st.success("PDFs processed successfully!")

# Chat Interface
if st.session_state.vectorstore is not None:
    
    # 2. Display previous chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    retriever = st.session_state.vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # 3. Chat Input
    if question := st.chat_input("Ask your question about the PDFs"):
        
        # Add user message to UI and history
        st.chat_message("user").markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        # Format chat history for the prompt
        formatted_history = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages[:-1]]
        )

        # Retrieve documents
        retrieved_docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # 4. Updated Prompt with Chat History
        prompt = f"""
        Answer using only the context below. Consider the chat history for context.

        Chat History:
        {formatted_history}

        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        # Generate response
        response = llm.invoke(prompt)

        # Add assistant message to UI and history
        with st.chat_message("assistant"):
            st.markdown(response.content)
            
            # Display sources gracefully
            shown = set()
            source_text = "\n\n**Sources:**\n"
            for doc in retrieved_docs:
                source = (doc.metadata["source_file"], doc.metadata["page"])
                if source not in shown:
                    source_text += f"* 📄 {doc.metadata['source_file']} (Page {doc.metadata['page'] + 1})\n"
                    shown.add(source)
            st.caption(source_text)
            
        st.session_state.messages.append({"role": "assistant", "content": response.content})
