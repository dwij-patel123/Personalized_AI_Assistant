from langchain_huggingface import HuggingFaceEndpoint
from langchain import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()


def predict_input_text(input_text,temperature):
    HUGGINFACE_API_TOKEN = os.getenv('HUGGINFACE_API_TOKEN')
    # used mistral model
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    llm = HuggingFaceEndpoint(repo_id=repo_id, huggingfacehub_api_token=HUGGINFACE_API_TOKEN,temperature=temperature)
    # Prompt Template for restricting output to 2 Sentences -- ADDITIONAL REQUIREMENTS
    template = """You are a smart AI assistant can you please assist me with a concise answer to the following question in exactly 2 sentences:{question}"""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = prompt | llm
    pred_ans = llm_chain.invoke(input_text)
    return pred_ans
