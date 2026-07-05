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

# API key input
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# ---- Session state initialization ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (question, answer) tuples

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = None

# Upload PDFs
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

# Process PDFs
if uploaded_files:

    # Only reprocess if the set of uploaded files has actually changed.
    # This avoids re-embedding on every single question.
    file_signature = tuple(
        sorted(f.name + str(f.size) for f in uploaded_files)
    )

    if st.session_state.processed_files != file_signature:

        with st.spinner("Processing PDFs..."):

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
                chunk_size=2500,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(documents)

            # Embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001"
            )

            # Vector DB
            st.session_state.vectorstore = Chroma.from_documents(
                chunks,
                embeddings
            )

            st.session_state.processed_files = file_signature
            st.session_state.chat_history = []  # reset memory for new document set

    retriever = st.session_state.vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    # ---- Display past conversation ----
    for past_question, past_answer in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(past_question)
        with st.chat_message("assistant"):
            st.write(past_answer)

    # Question box
    question = st.chat_input("Ask your question")

    if question:

        with st.chat_message("user"):
            st.write(question)

        # Build a short history string (last 5 turns) for context
        history_text = "\n".join(
            f"User: {q}\nAssistant: {a}"
            for q, a in st.session_state.chat_history[-5:]
        )

        # ---- Step 1: rewrite follow-up questions into standalone questions ----
        # This is what fixes "explain it further" style follow-ups: without
        # this step, the retriever has no idea what "it" refers to.
        if st.session_state.chat_history:
            condense_prompt = f"""Given the conversation history and a follow-up question, rewrite the follow-up question as a standalone question that includes all necessary context from the history. If the question is already standalone, return it unchanged. Only output the rewritten question, nothing else.

Conversation history:
{history_text}

Follow-up question:
{question}

Standalone question:"""

            standalone_question = llm.invoke(condense_prompt).content.strip()
        else:
            standalone_question = question

        # ---- Step 2: retrieve using the standalone question ----
        retrieved_docs = retriever.invoke(standalone_question)

        context = "\n\n".join(
            [doc.page_content for doc in retrieved_docs]
        )

        # ---- Step 3: answer using context + recent conversation ----
        answer_prompt = f"""Answer the question using only the context below. Use the conversation history only to resolve references like "it" or "that" — do not treat the history itself as a source of facts.

Conversation history:
{history_text}

Context:
{context}

Question:
{question}

Answer:"""

        response = llm.invoke(answer_prompt)
        answer = response.content

        with st.chat_message("assistant"):
            st.write(answer)

            st.markdown("**Sources**")

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

        # ---- Save this turn to memory ----
        st.session_state.chat_history.append((question, answer))
