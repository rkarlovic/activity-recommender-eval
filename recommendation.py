import cohere
import ollama
import os
import json
import pandas as pd
from litellm import completion
from dotenv import load_dotenv
import re

load_dotenv()
api_key = os.environ["COHERE_API_KEY"]

docs = None # Path to faiss_index

co = cohere.ClientV2(api_key)
response = co.chat(
    model="rerank-multilingual-v3.0",
    documents=docs,
    top_n=3,
)

print(response)