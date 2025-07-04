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

def main():
    print("Starting rerank search...")
    if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_FILE):
        print(f"FAISS files not found. Please run your FAISS building script first.")
        return
    
    # One input for all tests or multiple different inputs
    user_input = "Zelim razgledavati prirodu i dupine."

    reranked_results = search_with_reranking(user_input, initial_k=20, final_k=5)
    if reranked_results:
        for i, result in enumerate(reranked_results, 1):
            print(f"\n{i}. RERANKED RESULT:")
            print(f"   Rerank Score: {result['rerank_score']:.4f}")
            print(f"   FAISS Score: {result['faiss_score']:.4f}")
            # print(f"   Text: {result['text'][:300]}...")
            print(f"   Text: {result['text']}...")
            print(f"   Original Index: {result['original_index']}")

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



if __name__ == "__main__":
    print("Script is being run")
    main()