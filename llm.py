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

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.llms import HuggingFaceHub
from langchain.chat_models import ChatAnyscale

load_dotenv()

anyscale_api_key = os.environ.get('ANYSCALE_ENDPOINT_TOKEN')

verbose = os.environ.get('VERBOSE')

def get_local_LLAMA2():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-13b-chat-hf",
                                        # use_auth_token=True,
                                        )

    model = AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-13b-chat-hf",
                                            device_map='auto',
                                            torch_dtype=torch.float16,
                                            use_auth_token=True,
                                        #  load_in_8bit=True,
                                        #  load_in_4bit=True
                                        )
    from transformers import pipeline

    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer= tokenizer,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    max_new_tokens = 512,
                    do_sample=True,
                    top_k=30,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id
                    )
    
    from langchain import HuggingFacePipeline
    LLAMA2 = HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':0})
    print(f"\n\n> torch.cuda.is_available(): {torch.cuda.is_available()}")
    print("\n\n> local LLAMA2 loaded")
    return LLAMA2

def get_model(model_type):
   
    match model_type:
        case "google/flan-t5-xxl":
            # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.001, "max_length":1024})
            llm = HuggingFaceHub(repo_id="google/flan-t5-xxl",model_kwargs={"temperater":0,"max_length":100})
        case "tiiuae/falcon-7b-instruct":
            llm = HuggingFaceHub(repo_id=model_type, model_kwargs={"temperature":0.001, "max_length":1024})
        case "local/LLAMA2":
            llm = get_local_LLAMA2()
        case "Llama-2-13b":
            llm = ChatAnyscale(anyscale_api_key=anyscale_api_key,temperature=0, model_name='meta-llama/Llama-2-13b-chat-hf', streaming=False)
        case "Llama-2-70b":
            llm = ChatAnyscale(anyscale_api_key=anyscale_api_key,temperature=0, model_name='meta-llama/Llama-2-70b-chat-hf', streaming=False)
        case _default:
            # raise exception if model_type is not supported
            msg=f"Model type '{model_type}' is not supported. Please choose a valid one"
            logger.error(msg)

    logger.info(f"model_type: {model_type} loaded:")
    return llm  
