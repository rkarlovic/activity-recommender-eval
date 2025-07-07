from flask import Flask, request, jsonify
import requests
import faiss
import numpy as np
import os
import json
import cohere

# ~~~~ Postavke modela i API-ja ~~~~
embedding_dimension = 768  # nomic-embed-text embedding dim
index_file = 'faiss_store.index'
corpus_file = "data/corpus.json"

ollama_base_url = "http://localhost:11434"
ollama_embedding_url = f"{ollama_base_url}/api/embeddings"
ollama_generating_url = f"{ollama_base_url}/api/generate"

# Cohere API
cohere_api_key = os.getenv("COHERE_API_KEY")
co = cohere.Client(cohere_api_key)

app = Flask(__name__)

# ~~~~ Učitavanje vector store-a ~~~~
print("Učitavanje FAISS indexa...")
index = faiss.read_index(index_file)

with open(corpus_file, 'r', encoding='utf-8') as file:
    corpus = json.load(file)

print(f"Učitani korpus s {len(corpus)} chunkova.")

# ~~~~ Embedding function ~~~~
def get_embedding(text):
    payload = {
        "model": "nomic-embed-text",
        "prompt": [text]
    }
    response = requests.post(ollama_embedding_url, json=payload, timeout=60)
    response.raise_for_status()
    embedding = response.json()["embedding"][0]
    embedding = np.array(embedding, dtype=np.float32)
    faiss.normalize_L2(embedding.reshape(1, -1))
    return embedding.reshape(1, -1)

# ~~~~ Cohere re-ranking ~~~~
def rerank_with_cohere(query, docs, top_n=5):
    response = co.rerank(
        model="rerank-english-v2.0",
        query=query,
        documents=docs,
        top_n=top_n
    )
    reranked = [docs[r.index] for r in response]
    return reranked

# ~~~~ Recommendation generator ~~~~
def generate_recommendation(prompt):
    payload = {
        "model": "llama3",
        "prompt": prompt
    }
    response = requests.post(ollama_generating_url, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["response"]

# ~~~~ API endpoint ~~~~
@app.route("/recommend", methods=["POST"])
def create_recommendation():
    data = request.json
    user_profile = data.get("profile", "")
    if "weather" in data:
        weather = data["weather"]
    else:
        from syn_weather import get_synthetic_weather
        weather = get_synthetic_weather()
    
    query_text = f"{user_profile} {weather}"

    # Embedding i FAISS search
    query_embedding = get_embedding(query_text)
    D, I = index.search(query_embedding, k=20)  # uzimamo top 20 za re-ranking
    faiss_candidates = [corpus[i] for i in I[0]]

    # Cohere re-ranking
    reranked_candidates = rerank_with_cohere(query_text, faiss_candidates, top_n=5)

    # Generiranje preporuke
    recommendation_prompt = f"Na osnovu sljedećih informacija: {query_text}, preporuči aktivnosti: {', '.join(reranked_candidates)}"

    
