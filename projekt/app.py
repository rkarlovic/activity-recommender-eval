from flask import Flask, request, jsonify
import requests
import faiss
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer

# ~~~~ Postavke modela ~~~~
embedding_dimension = 768 # nomic
index_file = 'faiss_store.index'
corpus = "data/corpus.json"
# TODO: ollama_url = ?

ollama_embedding_url = "TODO"
ollama_generating_url = "TODO"

app = Flask(__name__)


# ~~~~ Učitavanje vector store-a ~~~~
print("Učitavanje FAISS indexa...")
index = faiss.read_index(index_file)

with open(corpus, 'r', encoding='utf-8') as file:
    corpus = json.load(file)

print(f"Učitani korpusi su dužine {len(corpus)} chunkova.")

# ~~~~ Embedding function ~~~~
def get_embedding(text):
    payload = {
        "model": "nomic-embed-text",
        "prompt": [text],
        "timeout": 60
    }

    response = requests.post(ollama_embedding_url, json=payload)
    response.raise_for_status()
    # TODO, IF NEEDED: If the response is not successful, it will raise an HTTPError ?

    embedding = response.json(["embedding"])[0]  # emb = response["data"][0]["embedding"] -> "grab the first one only"
    embedding = np.array(embedding, dtype=np.float32) # standardization to float32
    
    faiss.normalize_L2(embedding) # L2 normalization
    
    return embedding.reshape(1, -1) # 2D shape (1, n) where n is the embedding size

# ~~~~ Recommendation generator ~~~~
def generate_recommendation(prompt):
    payload = {
        "model" : "llama3", # TODO: update with chosenmodel name
        "prompt": prompt
    }

    response = requests.post(ollama_generating_url, json = payload)
    response.raise_for_status()
    return response.json()["response"]

# ~~~~ API endpoint ~~~~
@app.route("/recommend", methods = ["POST"])
def create_recommendation():
    data = request.json
    tourist_profile = data.get("profile", "")
    weather = data.get("weather", "")

    query_text = f"{tourist_profile} {weather}" 
    query_embedding = get_embedding(query_text)

    D, I = index.search(query_embedding, k = 3) # get top 3 recommendations
    top_chunks = [corpus[i] for i in I[0]]

    prompt = f"Ti si turist koji je dosao na Lošinj. Profil turista: {tourist_profile}, vremenski uvjeti su: {weather}. Preporuči mi aktivnosti na temelju ovih informacija: {top_chunks}."

    rec = generate_recommendation(prompt)

    return jsonify({
        "profile": tourist_profile,
        "weather": weather,
        "faiss_recommendations": top_chunks,
        "recommendation": rec
    })


# ~~~~ MAIN APP ~~~~
if __name__ == "__main__":
    app.run(debug=True)