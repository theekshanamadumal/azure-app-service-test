from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
# from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores.faiss import FAISS

import os
from dotenv import load_dotenv
import logging
logger = logging.getLogger(__name__)
'''
# all-MiniLM-L6-v2
EMBEDDINGS_MODEL_NAME="all-MiniLM-L6-v2"
embeddings_model_name =EMBEDDINGS_MODEL_NAME
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
persist_directory = "data/cbsl"
index_path = persist_directory

'''
chunk_size=1000
chunk_overlap=100

embeddings_model_name = "BAAI/bge-large-en"
# embeddings_model_name = "BAAI/bge-large-en-v1.5"
# persist_directory = "faiss_index"
# persist_directory = "faiss_index_with_year_2000_chunk"
# persist_directory = "faiss_index_2000_chunk_BGE_large_embeddings"
# persist_directory = "faiss_index_1000_chunk_BGE_large_embeddings"
persist_directory = "faiss_index_1000_chunk_BGE_large_embeddings_1000"

load_dotenv()

from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
inference_api_key = os.environ.get('HUGGINGFACEHUB_API_TOKEN')
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=inference_api_key, model_name=embeddings_model_name
)

# from langchain.embeddings import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)


def create_faiss():
    # documents = DirectoryLoader(persist_directory,  loader_cls=PyMuPDFLoader).load()
    documents = DirectoryLoader("CBSL",  loader_cls=PyPDFLoader).load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local("faiss_index")


def load_FAISS_store():
    try:
        print(f"> {persist_directory} loading")
        return FAISS.load_local(persist_directory, embeddings)
    except Exception as e:
        logger.exception(e)
        raise e
        