"""
 /*************************************************************************
 * 
 * CONFIDENTIAL
 * __________________
 * 
 *  Copyright (2023-2024) AI Labs, IronOne Technologies, LLC
 *  All Rights Reserved
 *
 *  Author  : Theekshana Samaradiwakara
 *  Description :Python Backend API to chat with research paper data  
 *  CreatedDate : 10/01/2024
 *  LastModifiedDate : 10/01/2024
 *************************************************************************/
 """

import logging
logger = logging.getLogger(__name__)
# from config import MEMORY_WINDOW_K

# from qaPipeline import QAPipeline
# from qaPipeline import run_agent

def get_QA_Answers(userQuery):
    query=userQuery.question
    # chat_history = userQuery.chat_history[-MEMORY_WINDOW_K:]
    
    # logger.info(f"query : {query} \n chat_history : {chat_history}")
    logger.info(f"query : {query}")
    # answer= run_agent(query)
    answer = run_general_qa_chain(query)
    # logger.info(f"Response: {answer}")
    return answer



from llmChain import get_general_qa_chain

general_qa_model_type="Llama-2-13b"
general_qa_chain= get_general_qa_chain(general_qa_model_type)

    
def run_general_qa_chain(query):
    try:
        logger.info(f"run_general_qa_chain : Question: {query}")

        # Get the answer from the chain
        res = general_qa_chain(query)

        return res

    except Exception as e:
        logger.exception(e)