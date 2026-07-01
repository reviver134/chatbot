import streamlit as st
import shutil
import os
import sys
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- HOUSEKEEPING ---
shutil.rmtree("./chroma_db", ignore_errors=True)
if 'torch.classes' in sys.modules:
    del sys.modules['torch.classes']

# --- SETUP QA SYSTEM ---
@st.cache_resource
def setup_qa():
    loader = TextLoader("text.txt")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, # Increased size for better context
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents( 
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("API_KEY"),
        temperature=0
    )

    # MODERN PROMPT TEMPLATE
    prompt = ChatPromptTemplate.from_template("""
    Use the following context to answer the question.
    If you don't know the answer, just say you don't know.
    
    Context:
    {context}
    
    Question: {input}
    
    Answer:
    """)

    # MODERN CHAINS
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return rag_chain

rag_chain = setup_qa()

# --- STREAMLIT UI ---
st.title("📚 Modern RAG Chatbot (LangChain + Gemini)")
query = st.text_input("Your question:")

if query:
    with st.spinner("Thinking..."):
        # The invoke method is the standard LCEL way to run chains
        response = rag_chain.invoke({"input": query})

        st.subheader("✅ Answer")
        st.write(response["answer"])

        st.subheader("📄 Sources")
        for i, doc in enumerate(response["context"]):
            st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'text.txt')}")