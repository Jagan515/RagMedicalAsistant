from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

#Load PDFs
def load_pdf_file(data):
    loader=DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

#Filter the documents
def fiter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs = []
    for doc in docs:
        minimal_doc = Document(page_content=doc.page_content, metadata={})
        minimal_docs.append(minimal_doc)
    return minimal_docs

#Split the dcocuments into chunks
def split_documents(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    text_chunks = text_splitter.split_documents(minimal_docs)
    return text_chunks

#Download the Embeding model
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
    return embeddings

#helper
def docs_to_prompt_input(docs, question):
    return {"context": "\n".join([d.page_content for d in docs]), "question": question}