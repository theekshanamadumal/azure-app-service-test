"""
Python app to chat with research peper data  

01/09/2024
D.M. Theekshana Samaradiwakara

python -m streamlit run app.py
"""

import os
import time
import logging
logger = logging.getLogger(__name__)
from dotenv import load_dotenv

from faissDb import load_FAISS_store
from llmChain import get_qa_chain, get_general_qa_chain, get_router_chain
from output_parser import general_qa_chain_output_parser, qa_chain_output_parser, out_of_domain_chain_parser

from config import ANSWER_TYPES

load_dotenv()

verbose = os.environ.get('VERBOSE')


from conversationBufferWindowMemory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            input_key="question",
            output_key = "answer",
            return_messages=True,
            k=1
)

vectorstore=load_FAISS_store()
retriever = vectorstore.as_retriever(
    # search_type="mmr",
    search_kwargs={'k': 5, 'fetch_k': 10}
    )
logger.info("retriever loaded:")

# model_type="local/LLAMA2"
qa_model_type="Llama-2-13b"
general_qa_model_type="Llama-2-13b"
router_model_type="google/flan-t5-xxl"
# model_type="tiiuae/falcon-7b-instruct"
qa_chain= get_qa_chain(qa_model_type,retriever)
general_qa_chain= get_general_qa_chain(general_qa_model_type)
router_chain= get_router_chain(router_model_type)

def chain_selector(chain_type, query):
    chain_type = chain_type.lower().strip()
    logger.info(f"chain_selector : chain_type: {chain_type} Question: {query}")
    if "greeting" in chain_type:
        return run_general_qa_chain(query)
    elif "other" in chain_type:
        return run_out_of_domain_chain(query)
    elif ("relevant" in chain_type) or ("not sure" in chain_type) :
        return run_qa_chain(query)
    else:
        raise ValueError(
            f"Received invalid type '{chain_type}'"
        )
    
def run_agent(query):
    try:
        logger.info(f"run_agent : Question: {query}")
        # Get the answer from the chain
        start = time.time()
        chain_type = run_router_chain(query)
        res = chain_selector(chain_type,query)
        end = time.time()

        # log the result
        logger.info(f"Answer (took {round(end - start, 2)} s.) \n: {res}")

        return res

    except Exception as e:
        logger.exception(e)


def run_router_chain(query):
    try:
        logger.info(f"run_router_chain : Question: {query}")
        # Get the answer from the chain
        start = time.time()
        chain_type = router_chain(query)['text']
        end = time.time()

        # log the result
        logger.info(f"Answer (took {round(end - start, 2)} s.) chain_type: {chain_type}")

        return chain_type

    except Exception as e:
        logger.exception(e)


def run_qa_chain(query):
    try:
        logger.info(f"run_qa_chain : Question: {query}")
        # Get the answer from the chain
        start = time.time()
        # res = qa_chain(query)
        res = qa_chain({"question": query, "chat_history":""})
        # res = response
        # answer, docs = res['result'],res['source_documents']
        end = time.time()

        # log the result
        logger.info(f"Answer (took {round(end - start, 2)} s.) \n: {res}")

        return qa_chain_output_parser(res)

    except Exception as e:
        logger.exception(e)


def run_general_qa_chain(query):
    try:
        logger.info(f"run_general_qa_chain : Question: {query}")

        # Get the answer from the chain
        start = time.time()
        res = general_qa_chain(query)
        end = time.time()

        # log the result
       
        logger.info(f"Answer (took {round(end - start, 2)} s.) \n: {res}")

        return general_qa_chain_output_parser(res)

    except Exception as e:
        logger.exception(e)


def run_out_of_domain_chain(query):
    return out_of_domain_chain_parser(query)