import streamlit as st
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.retrieval_qa.base import RetrievalQA
import shutil
import sys

# Remove old DB
shutil.rmtree("./chroma_db", ignore_errors=True)

# Fix torch issue in Streamlit
if 'torch.classes' in sys.modules:
    del sys.modules['torch.classes']

# Setup QA system
@st.cache_resource
def setup_qa():
    loader = TextLoader("text.txt")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=5,
        separators=[r"\n\n", r"\n", r"?<=\.", "!", "?", ",", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents( 
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = OllamaLLM(model="llama2")

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Use the following context to answer the question.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        )
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=False)

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        verbose=False
    )

    qa_chain = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=combine_documents_chain,
        return_source_documents=True,
        verbose=False
    )

    return qa_chain, retriever, prompt_template


qa_chain, retriever, prompt_template = setup_qa()

# Streamlit UI
st.title("📚 RAG Chatbot with LangChain + Ollama")
st.markdown("Ask me anything based on the content of `text.txt`.")

query = st.text_input("Your question:", placeholder="e.g. What is meditation?")

if query:
    with st.spinner("Thinking..."):
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = prompt_template.format(context=context, question=query)

        st.subheader("🔍 Prompt sent to LLM")
        st.code(prompt)

        result = qa_chain({"query": query})

        st.subheader("✅ Answer")
        st.write(result["result"])

        st.subheader("📄 Sources")
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'text.txt')}")
