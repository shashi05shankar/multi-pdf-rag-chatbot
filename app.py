
# Import libraries
import streamlit as st
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

# API key input
api_key = st.text_input(
    "Enter Gemini API Key",
    type="password"
)

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# Upload PDFs
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

# Process PDFs
if uploaded_files and api_key:

    documents = []

    for uploaded_file in uploaded_files:

        # Save temporary PDF
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

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
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    # Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    # Vector DB
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    # Question box
    question = st.text_input(
        "Ask your question"
    )

    if question:

        retrieved_docs = retriever.invoke(question)

        context = "\n\n".join(
            [doc.page_content for doc in retrieved_docs]
        )

        prompt = f"""
Answer using only the context below.

Context:
{context}

Question:
{question}

Answer:
"""

        response = llm.invoke(prompt)

        st.subheader("Answer")
        st.write(response.content)

        st.subheader("Sources")

        shown = set()

        for doc in retrieved_docs:

            source = (
                doc.metadata["source_file"],
                doc.metadata["page"]
            )

            if source not in shown:

                st.write(
                    f"📄 {doc.metadata['source_file']} "
                    f"(Page {doc.metadata['page'] + 1})"
                )

                shown.add(source)
