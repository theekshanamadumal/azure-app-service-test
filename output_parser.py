"""
Python app to chat with research peper data  

01/09/2024
D.M. Theekshana Samaradiwakara

python -m streamlit run app.py
"""

def qa_chain_output_parser(result):
    return {
        "question": result["question"], 
        "answer": result["answer"],
        "source_documents": result["source_documents"]
    }

def general_qa_chain_output_parser(result):
    return {
        "question": result["question"], 
        "answer": result["text"],
        "source_documents": []
    }


def out_of_domain_chain_parser(query):
    return {
        "question": query, 
        "answer":"sorry this question is out of my domain.",
        "source_documents":[]
    }


