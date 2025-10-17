from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import numpy as np
import redis
import os
import json
from dotenv import load_dotenv
from groq import Groq
import time

load_dotenv()

app = FastAPI(title = "Model Optimization Layer")

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True, 
  allow_methods=["*"],
  allow_headers=["*"],
)

print("Loading model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

redis_client = redis.Redis(
  host = os.getenv("REDIS_HOST", "localhost"),
  port = int(os.getenv("REDIS_PORT", 6379)),
  decode_responses = True
)

#Redis COnnection
try:
  redis_client.ping()
  print("Connected to Redis")
except redis.exceptions.ConnectionError as e:
  print("Redis connection error:", e)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

similarity_threshold = 0.85

metrics = {
  "total_requests": 0,
  "cache_hits": 0,
  "cache_misses": 0,
  "groq_requests": 0,
  "total_api_cost_saved" : 0.0
}

def cosine_similarity(vec1, vec2):
  return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def check_cache(query_embedding):
  cached_keys = redis_client.keys("embedding:*")

  if not cached_keys:
    return None
  
  max_similarity = 0
  best_match = None

  for key in cached_keys:
    cached_data = redis_client.get(key)
    if not cached_data:
      continue

    cached_obj = json.loads(cached_data)
    cached_embedding = np.array(cached_obj["embedding"])

    similarity = cosine_similarity(query_embedding, cached_embedding)

    if similarity > max_similarity:
      max_similarity = similarity
      best_match = cached_obj

  if max_similarity >= similarity_threshold:
    return {
      "response" : best_match["response"],
      "similarity" : max_similarity,
      "original_query" : best_match["query"]
    }
  
def call_llm(query):
  try:
    response = groq_client.chat.completions.create(
      messages = [
        {
          "role": "system",
          "content": "You are a helpful assistant. Provide concise, accurate answers."
        },
        {
          "role": "user",
          "content": query
        }
      ],
      model="llama3-8b-8192",
      temperature=0.7,
      max_tokens=500
    )

    return response
  except Exception as e:
    return f"Error calling Groq LLM: {str(e)}"
  
  def store_in_cache(query, query_embedding, response):
    cache_key = f"embedding:{abs(hash(query))}"
    
    cache_data = {
        "query": query,
        "embedding": embedding.tolist(),
        "response": response
    }
    
    redis_client.set(cache_key, json.dumps(cache_data))