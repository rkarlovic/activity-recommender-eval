import time
import cohere
import ollama
import os
import json
import pandas as pd
from litellm import completion, rerank
from dotenv import load_dotenv
import re
import faiss
import numpy as np
from datetime import datetime
import random
import logging


load_dotenv() 
api_key = os.environ["COHERE_API_KEY"]
INDEX_FILE = os.environ["FAISS_INDEX_FILE"]
CHUNKS_FILE = os.environ["CHUNKS_FILE"]

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        logger.error(f"Error getting embedding: {e}")
        return None

def load_faiss_index_and_chunks():
    """
    Load FAISS index and chunks from files.
    Returns:
        index: FAISS index object
        chunks: List of text chunks
    """
    try:
        index = faiss.read_index(INDEX_FILE)
        chunks = np.load(CHUNKS_FILE, allow_pickle=True)
        # print(f"Loaded FAISS index with {len(chunks)} chunks.")
        return index, chunks
    except Exception as e:
        logger.error(f"Error loading FAISS index or chunks: {e}")
        return None, None

def search_with_reranking(query: str, index, chunks, initial_k: int = 20, final_k: int = 5):
    """
    Two-stage search:
    1. FAISS retrieval to get initial candidates
    2. Cohere reranking to improve results
    """
    
    # Load FAISS index and chunks
    # try:
    #     index = faiss.read_index(INDEX_FILE)
    #     chunks = np.load(CHUNKS_FILE, allow_pickle=True)
    #     print(f"Loaded FAISS index with {len(chunks)} chunks.")
    # except Exception as e:
    #     print(f"Error loading FAISS index: {e}")
    #     return []
    
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
    
    # print(f"FAISS found {len(retrieved_chunks)} initial results.")
    
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
        
        # print(f"Cohere reranking returned {len(reranked_results)} results.")
        return reranked_results
        
    except Exception as e:
        print(f"Error during Cohere reranking: {e}")
        # Fallback to FAISS results only
        return retrieved_chunks[:final_k]

def format_retrieved_chunks(results) -> str:
    """Format retrieved chunks for the prompt"""
    if not results:
        return "No relevant information found."
    
    formatted_chunks = []
    for i, result in enumerate(results, 1):
        formatted_chunks.append(f"{i}. {result["text"]}")
    
    return "\n".join(formatted_chunks)

def get_response(message, model_name):
    system_message = {

        "role": "system",
        "content": "You are an intelligent tourism assistant. You recommend personalized activities to tourists based on:\n\
                    1. The user's interests and preferences (as defined in their profile).\n\
                    2. Their current intent or question (described in the user input).\n\
                    3. Relevant local information retrieved via semantic search (provided as bullet points).\n\
                    4. The current weather conditions.\n\n\
                    Your response should be personalized, context-aware, and practical.\n\
                    If the weather is limiting, suggest creative alternatives or indoor activities.\n\
                    If there are strong user preferences, make sure to respect them.\n\
                    Respond in a friendly and informative tone. Your output should be a short and clear activity recommendation, followed by a brief explanation if needed."

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
    
def load_user_profiles(json_file_path: str):
    """Load user profiles from JSON file"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading user profiles: {e}")
        return {}
    
def save_results_to_csv(results, model_name, output_dir="csv_outputs"):
    """Save results to CSV file for each model"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with model name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name.replace(':', '_').replace('/', '_')}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Add model name column
    df['Model_Name'] = model_name
    
    # Reorder columns to have Model_Name first
    columns = ['Model_Name'] + [col for col in df.columns if col != 'Model_Name']
    df = df[columns]
    
    # Save to CSV
    df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"Results saved to: {filepath}")
    return filepath

def main():
    print("Starting rerank search...")
    if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_FILE):
        print(f"FAISS files not found. Please run your FAISS building script first.")
        return
    
    user_profiles = load_user_profiles("user_profiles_updated.json")
    if not user_profiles:
        print("No user profiles loaded. Exiting.")
        return
    
    

    conditions = [
        "Weather: sunny, temperature 25°C, Wind 10 km/h",
        "Weather: partly cloudy, temperature 22°C, Wind 12 km/h",
        "Weather: mostly cloudy, temperature 20°C, Wind 14 km/h",
        "Weather: rainy, temperature 18°C, Wind 20 km/h",
        "Weather: clear, temperature 24°C, Wind 8 km/h",
        "Weather: windy, temperature 21°C, Wind 25 km/h",
        "Weather: humid, temperature 27°C, Wind 15 km/h",
        "Weather: foggy, temperature 17°C, Wind 5 km/h"
    ]
    model_names = ["ollama/gemma3:latest", "ollama/qwen3:latest", "ollama/qwen2:latest", "ollama/granite3.3:latest", "ollama/granite3.2:latest", "ollama/llama3.2:latest", "ollama/llama3.1:latest", "ollama/deepseek-r1:8b", "ollama/phi4:14b", "ollama/mistral:7b"]
    total_prompts_per_model = len(user_profiles) * len(conditions)
    for model_name in model_names:
        print(f"\nProcessing with LLM model: {model_name}")
        results = []
        current_prompt_count = 0
        start_time = datetime.now()
        for user_id, user_profile in user_profiles.items():
            for weather in conditions:
                print(f"Processing user profile: {user_id}, with weather condition: {weather}")
                
                current_prompt_count += 1
                remaining = total_prompts_per_model - current_prompt_count
                print(f"📊 {model_name}: {current_prompt_count}/{total_prompts_per_model} Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}, (⏳ {remaining} left)")
                
                user_input = user_profile.get("input", "")
                user_age = user_profile.get("age", "Unknown")
                user_gender = user_profile.get("gender", "Unknown")
                user_diet = user_profile.get("diet", "Unknown")
                user_food_preferences = user_profile.get("food_preferences", [])
                user_likes = user_profile.get("likes", [])
                user_dislikes = user_profile.get("dislikes", [])
                user_weather_preference = user_profile.get("weather_preference", "Unknown")
                user_lifestyle = user_profile.get("lifestyle", [])

                # Search with reranking
                index, chunks = load_faiss_index_and_chunks()
                if index is None or chunks is None:
                    print("Failed to load FAISS index or chunks. Skipping this user.")
                    continue
                reranked_results = search_with_reranking(user_input, index, chunks, initial_k=30, final_k=10)
                # reranked_results = search_with_reranking(user_input, initial_k=30, final_k=10)
                if not reranked_results:
                    print("No results found after reranking.")
                    continue
                
                retrieved_chunks = format_retrieved_chunks(reranked_results)

                message = [
                    {
                        "role": "user",
                        "content": f"""
                                    User age: 
                                    {user_age}
                                    
                                    User gender: 
                                    {user_gender}
                                    
                                    User diet: 
                                    {user_diet}
                                    
                                    User food preferences: 
                                    {user_food_preferences}
                                    
                                    User likes: 
                                    {user_likes}
                                    
                                    User dislikes: 
                                    {user_dislikes}
                                    
                                    User weather preference: 
                                    {user_weather_preference}
                                    
                                    User lifestyle: 
                                    {user_lifestyle}

                                    User Input:
                                    {user_input}

                                    Relevant Information:
                                    {retrieved_chunks}

                                    Current Weather:
                                    {weather}

                                    Please provide a personalized activity recommendation that considers the user’s preferences, current input, background information, and weather conditions.
                                    """
                    }
                ]
            

                # response = get_response(message, model_name = "llama3.1:latest")
                response = get_response(message, model_name)
                time.sleep(6)
                
                if response is None:
                    response = "Error: No response from model"
                    print(f"Warning: No response from model for user {user_id}")
                
                # Create result record
                result = {
                    "User_ID": user_id,
                    "User_Age": user_age,
                    "User_Gender": user_gender,
                    "User_Diet": user_diet,
                    "User_Food_Preferences": str(user_food_preferences),  # Convert list to string for CSV
                    "User_Likes": str(user_likes),
                    "User_Dislikes": str(user_dislikes),
                    "User_Weather_Preference": user_weather_preference,
                    "User_Lifestyle": str(user_lifestyle),
                    "User_Input": user_input,
                    "Relevant_Information": retrieved_chunks,
                    "Current_Weather": weather,
                    "LLM_Response": response
                }
                
                results.append(result)

                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds() / 60

                print(f"✓ Processed successfully!!! Duration: {duration:.2f} minutes")
            
        if results:
            csv_path = save_results_to_csv(results, model_name)
            print(f"Completed processing {model_name}. Results saved to: {csv_path}")
            print(f"Total records generated: {len(results)}")
        else:
            print(f"No results generated for model: {model_name}")

if __name__ == "__main__":
    print("Script is being run")
    main()