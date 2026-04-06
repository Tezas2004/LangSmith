# pip install -U langchain langchain-community langchain-huggingface faiss-cpu pypdf python-dotenv langsmith sentence-transformers

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RAG_Demo_Upgraded_V2"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

from langsmith import traceable

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    ChatHuggingFace,
    HuggingFaceEndpoint
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser

PDF_PATH = "islr.pdf"

# ----------------- helper functions -----------------

@traceable(name="load_pdf")
def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load()

@traceable(name="split_documents")
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

@traceable(name="build_vectorstore")
def build_vectorstore(splits):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(splits, embeddings)



@traceable(name="setup_pipeline", tags=["setup"])
def setup_pipeline(pdf_path: str, chunk_size=1000, chunk_overlap=150):
    docs = load_pdf(pdf_path)
    splits = split_documents(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    vectorstore = build_vectorstore(splits)
    return vectorstore



llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.3,
)

model = ChatHuggingFace(llm=llm)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer ONLY from the provided context. If not found, say 'I don't know.'"
    ),
    (
        "human",
        "Question: {question}\n\nContext:\n{context}"
    )
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@traceable(name="pdf_rag_full_run")
def setup_pipeline_and_query(pdf_path: str, question: str):
    vectorstore = setup_pipeline(
        pdf_path,
        chunk_size=1000,
        chunk_overlap=150
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    parallel = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })

    chain = parallel | prompt | model | StrOutputParser()

    lc_config = {
        "run_name": "pdf_rag_query",
        "tags": ["rag", "pdf", "huggingface"]
    }

    return chain.invoke(question, config=lc_config)



if __name__ == "__main__":
    print("PDF RAG ready. Ask a question (or type 'exit').")

    while True:
        q = input("\nQ: ").strip()

        if q.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        ans = setup_pipeline_and_query(PDF_PATH, q)
        print("\nA:", ans)