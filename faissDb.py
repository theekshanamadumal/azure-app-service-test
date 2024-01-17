from langchain.embeddings import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
# from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores.faiss import FAISS

'''
# all-MiniLM-L6-v2
EMBEDDINGS_MODEL_NAME="all-MiniLM-L6-v2"
embeddings_model_name =EMBEDDINGS_MODEL_NAME
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
persist_directory = "data/cbsl"
index_path = persist_directory

'''
chunk_size=2000
chunk_overlap=100

embeddings_model_name = "BAAI/bge-large-en-v1.5"
# persist_directory = "faiss_index"
# persist_directory = "faiss_index_with_year_2000_chunk"
# persist_directory = "faiss_index_2000_chunk_BGE_large_embeddings"
persist_directory = "faiss_index_1000_chunk_BGE_large_embeddings"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)


def create_faiss():
    # documents = DirectoryLoader(persist_directory,  loader_cls=PyMuPDFLoader).load()
    documents = DirectoryLoader("CBSL",  loader_cls=PyPDFLoader).load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
   
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local("faiss_index")


def load_FAISS_store():
    print(f"> {persist_directory} loaded")
    return FAISS.load_local(persist_directory, embeddings)
