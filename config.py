AVALIABLE_MODELS=[
    {
    "id":"Llama-2-13b",
    "model_name":"anyscale/Llama-2-13b-chat-hf",
    "description":"llama2 13 billion model from anyscale"
    },
    {
    "id":"Llama-2-70b",
    "model_name":"anyscale/Llama-2-70b-chat-hf",
    "description":"llama2 70 billion model from anyscale"
    },
    {
    "id":"gpt-3.5",
    "model_name":"openai/gpt-3.5",
    "description":"gpt-3.5 model from openai"
    }
]

MODELS={
    "DEFAULT":"anyscale/Llama-2-13b-chat-hf",
    # "hf/falcon-7b-instruct":"tiiuae/falcon-7b-instruct",
    
    "Llama-2-13b":"anyscale/Llama-2-13b-chat-hf",
    "Llama-2-70b":"anyscale/Llama-2-70b-chat-hf",

    # "local/Llama-2-7b":"local/LLAMA2",

    "gpt-3.5":"openai",
    
}

MEMORY_WINDOW_K = 1

ANSWER_TYPES = [
    "relevant",
    "greeting",
    "other",
    "not sure",
]

QA_MODEL_TYPE = "Mistral-7B"
GENERAL_QA_MODEL_TYPE = "Mistral-7B"