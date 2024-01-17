"""
Python app to chat with research peper data  

01/09/2024
D.M. Theekshana Samaradiwakara

python -m streamlit run app.py
"""

import os
import logging
logger = logging.getLogger(__name__)
from dotenv import load_dotenv

load_dotenv()

verbose = os.environ.get('VERBOSE')

from llm import get_model
from langchain.chains import  ConversationalRetrievalChain
from conversationBufferWindowMemory import ConversationBufferWindowMemory

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from prompts import retrieval_qa_chain_prompt, document_combine_prompt, general_qa_chain_prompt, router_prompt

def get_qa_chain(model_type,retriever):
    logger.info("creating qa_chain")
    
    try:
        qa_llm = get_model(model_type)

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=qa_llm,
            chain_type="stuff",
            retriever = retriever, 
            # retriever = self.retriever(search_kwargs={"k": target_source_chunks}
            return_source_documents= True,
            get_chat_history=lambda h : h,
            combine_docs_chain_kwargs={
                "prompt": retrieval_qa_chain_prompt,
                "document_prompt": document_combine_prompt,
            },
            verbose=True,
            # memory=memory,
        )

        logger.info("qa_chain created")
        return qa_chain

    except Exception as e:
        msg=f"Error : {e}"
        logger.exception(msg)


def get_general_qa_chain(model_type):
    logger.info("creating general_qa_chain")
    
    try:
        general_qa_llm = get_model(model_type)
        general_qa_chain = LLMChain(llm=general_qa_llm, prompt=general_qa_chain_prompt)

        logger.info("general_qa_chain created")
        return general_qa_chain

    except Exception as e:
        msg=f"Error : {e}"
        logger.exception(msg)


def get_router_chain(model_type):
    logger.info("creating router_chain")
    
    try:
        router_llm = get_model(model_type)
        router_chain = LLMChain(llm=router_llm, prompt=router_prompt)

        logger.info("router_chain created")
        return router_chain

    except Exception as e:
        msg=f"Error : {e}"
        logger.exception(msg)


