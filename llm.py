"""
Python app to chat with research peper data  

01/09/2024
D.M. Theekshana Samaradiwakara

python -m streamlit run app.py
"""

import os
# import time
import logging
logger = logging.getLogger(__name__)
from dotenv import load_dotenv

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain_community.chat_models import ChatAnyscale

load_dotenv()

anyscale_api_key = os.environ.get('ANYSCALE_ENDPOINT_TOKEN')

verbose = os.environ.get('VERBOSE')

def get_model(model_type):
   
    match model_type:
        case "Llama-2-13b":
            llm = ChatAnyscale(anyscale_api_key=anyscale_api_key,temperature=0, model_name='meta-llama/Llama-2-13b-chat-hf', streaming=False)
        case "Llama-2-70b":
            llm = ChatAnyscale(anyscale_api_key=anyscale_api_key,temperature=0, model_name='meta-llama/Llama-2-70b-chat-hf', streaming=False)
        case "Mistral-7B":
            # add mistral model
            llm = ChatAnyscale(anyscale_api_key=anyscale_api_key,temperature=0, model_name='mistralai/Mistral-7B-Instruct-v0.1', streaming=False)
        case "Mixtral-8x7B":
            # add mistral model
            llm = ChatAnyscale(anyscale_api_key=anyscale_api_key,temperature=0, model_name='mistralai/Mixtral-8x7B-Instruct-v0.1', streaming=False)
            # llm = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",model_kwargs={"temperater":0,"max_length":100})
        case _default:
            # raise exception if model_type is not supported
            msg=f"Model type '{model_type}' is not supported. Please choose a valid one"
            logger.error(msg)
            return Exception(msg)

    logger.info(f"model_type: {model_type} loaded:")
    return llm