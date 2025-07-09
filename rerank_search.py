import cohere
import ollama
import os
import json
import pandas as pd
from litellm import completion
from dotenv import load_dotenv
import re
import faiss
import numpy as np
import random
from datetime import datetime

load_dotenv() 
api_key = os.environ["COHERE_API_KEY"]
INDEX_FILE = os.environ["FAISS_INDEX_FILE"]
CHUNKS_FILE = os.environ["CHUNKS_FILE"]


def get_embedding(text: str):
    if not text.strip():
        return None
    
    try:
        response = ollama.embed(
                model='nomic-embed-text:latest',
                input=text)
        emb = response["embeddings"][0]

        return np.array(emb, dtype=np.float32)
    except Exception as e:
        print(f"Error getting embedding: {e}. Skipping this chunk:")
        print(text)
        return None

def search_with_reranking(query: str, initial_k: int = 20, final_k: int = 5):
    """
    Two-stage search:
    1. FAISS retrieval to get initial candidates
    2. Cohere reranking to improve results
    """
    
    # Load FAISS index and chunks
    try:
        index = faiss.read_index(INDEX_FILE)
        chunks = np.load(CHUNKS_FILE, allow_pickle=True)
        print(f"Loaded FAISS index with {len(chunks)} chunks.")
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return []
    
    # Get query embedding
    query_embedding = get_embedding(query)
    if query_embedding is None:
        print("Failed to get embedding for query.")
        return []
    
    # FAISS search
    query_vector = query_embedding.reshape(1, -1)
    faiss_scores, faiss_indices = index.search(query_vector, initial_k)
    
    # Extract retrieved chunks
    retrieved_chunks = []
    for i, idx in enumerate(faiss_indices[0]):
        if idx != -1:  # Valid index
            chunk_text = chunks[idx]
            faiss_score = float(faiss_scores[0][i])
            retrieved_chunks.append({
                'text': chunk_text,
                'faiss_score': faiss_score,
                'index': idx
            })
    
    if not retrieved_chunks:
        print("FAISS search returned no results.")
        return []
    
    print(f"FAISS found {len(retrieved_chunks)} initial results.")
    
    # Cohere reranking
    try:
        co = cohere.ClientV2(api_key)
        documents = [chunk['text'] for chunk in retrieved_chunks]
        
        rerank_response = co.rerank(
            model="rerank-multilingual-v3.0",
            query=query,
            documents=documents,
            top_n=final_k
        )
        
        # Process reranking results
        reranked_results = []
        for result in rerank_response.results:
            original_chunk = retrieved_chunks[result.index]
            reranked_results.append({
                'text': original_chunk['text'],
                'rerank_score': result.relevance_score,
                'faiss_score': original_chunk['faiss_score'],
                'original_index': original_chunk['index']
            })
        
        print(f"Cohere reranking returned {len(reranked_results)} results.")
        return reranked_results
        
    except Exception as e:
        print(f"Error during Cohere reranking: {e}")
        # Fallback to FAISS results only
        return retrieved_chunks[:final_k]
# co = cohere.ClientV2(api_key)
# response = co.chat(
#     model="rerank-multilingual-v3.0",
#     documents=docs, 
#     top_n=10 #Should it be lowered to 5?
# )

# print(response)

#Generiranje sintetičkog vremena
def generate_synthetic_weather(location: str = "Lošinj") -> str:
   
    conditions = [
        "sunčano",
        "djelomično oblačno",
        "pretežno oblačno",
        "kišovito",
        "vedro",
        "vjetrovito",
        "sparno",
        "maglovito"
    ]
    condition = random.choice(conditions)
    temperature = random.randint(20, 35)  
    wind_speed = random.randint(5, 20)    

    current_time = datetime.now().strftime("%d.%m.%Y %H:%M")

    weather_description = (
        f"Vrijeme za lokaciju {location} ({current_time}): "
        f"{condition}, temperatura {temperature}°C, vjetar {wind_speed} km/h."
    )

    return weather_description


def get_response(message, model_name):
    system_message = {

        "role": "system",
        "content": """
        You are a helpful assistant that provides recommendations based on user input.
        """
    }

    if not message or message[0]["role"] != "system":
        message.insert(0, system_message)
    
    try:
        response = completion(
            model=model_name,
            messages= message,
            api_base="http://localhost:11434",
        )

        # Extract content from response
        if hasattr(response, 'choices') and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return None
                
    except Exception as e:
        print(f"Error making API call: {e}")
        return None

def main():
    print("Starting rerank search...")
    if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_FILE):
        print(f"FAISS files not found. Please run your FAISS building script first.")
        return
    
    # One input for all tests or multiple different inputs
    user_input = "Zelim razgledavati prirodu i dupine."

    reranked_results = search_with_reranking(user_input, initial_k=20, final_k=10)
    if reranked_results:
        for i, result in enumerate(reranked_results, 1):
            print(f"\n{i}. RERANKED RESULT:")
            print(f"   Rerank Score: {result['rerank_score']:.4f}")
            print(f"   FAISS Score: {result['faiss_score']:.4f}")
            # print(f"   Text: {result['text'][:300]}...")
            print(f"   Text: {result['text']}...")
            print(f"   Original Index: {result['original_index']}")

            weather_info = generate_synthetic_weather("Lošinj")
            print("\n Trenutno vrijeme na Lošinju je:")
            print(weather_info)


        # Output to a file if needed for further analysis
        # with open("reranked_output.txt", "w", encoding="utf-8") as f:
        #     for i, result in enumerate(reranked_results, 1):
        #         f.write(f"\n{i}. RERANKED RESULT:\n")
        #         f.write(f"   Rerank Score: {result['rerank_score']:.4f}\n")
        #         f.write(f"   FAISS Score: {result['faiss_score']:.4f}\n")
        #         f.write(f"   Text: {result['text']}\n")
        #         f.write(f"   Original Index: {result['original_index']}\n")
        #     else:
        #         print("No results found.")

    model_names = ["gemma3:latest", "qwen3:latest", "qwen2:latest", "granite3.3:latest", "granite3.2:latest", "llama3.2:latest", "llama3.1:latest", "deepseek-r1:8b", "phi4:14b", "mistral:7b"]
    for model_name in model_names:
        results = []
        #Add for loop here to iterate over all user inputs once they are defined
        # message = [
        #     {
        #         "role": "user",
        #         "content": (
        #             f"{user_input}\n\n"
        #             "\n".join([f"- {r['text']}" for r in reranked_results]) + f"\n\nAktualno vrijeme:\n{weather_info}""Please provide a recommendation based on the given input."
        #         ) 
        #     }
        # ]

        response = get_response(message, model_name = "llama3.1:latest")
        print("\n Response:")
        print(response)

        # Assuming the response is a JSON string, parse it
        # Add parser to handle the response once return format is defined
        try:
            response_data = json.loads(response)
            result = {
                "recommendation": response_data.get("recommendation", ""),
            }

            results.append(result)

        except json.JSONDecodeError as e:
            # print(f"Error decoding JSON for row {index}: {response}")
            print(f"JSON decode error: {e}")

            results.append(result)
            continue
        

if __name__ == "__main__":
    print("Script is being run")
    main()