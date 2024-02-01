"""
Python app to chat with research peper data  

01/09/2024
D.M. Theekshana Samaradiwakara

python -m streamlit run app.py
"""
from langchain.text_splitter import SentenceTransformersTokenTextSplitter

def get_token_counr(text):
    
    splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0)
    text_token_count = splitter.count_tokens(text=text)
    return text_token_count