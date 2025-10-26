from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import numpy as np
import redis
from urllib.parse import urlparse #To parse Redis URL into parts like hostname, port & password
import json
import os
from dotenv import load_dotenv
from groq import Groq
import time


load_dotenv()


app = FastAPI(title="Model Layer Optimization")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
#Optimising oading time & memory by lazy loading
embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        print("Loading embedding model")
        embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2") 
        print("Model loaded!")
    return embedding_model

redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
url = urlparse(redis_url)

redis_client = redis.Redis(
    host = url.hostname or 'localhost',
    port= url.port or 6379,
    password= url.password,
    decode_responses=True,
    socket_connect_timeout = 10,
    socket_timeout = 10
)

try:
    redis_client.ping()
    print("Connected to Redis")
except Exception as e:
    print(f"Redis connection failed: {e}")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SIMILARITY_THRESHOLD = 0.8

metrics = {
    "total_requests": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "total_api_cost_saved": 0.0
}

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def check_cache(query_embedding, query_text):
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
        
        print(f"Similarity between '{cached_obj['query']}' and '{query_text}': {similarity:.4f}")
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = cached_obj
    
    

    if max_similarity >= SIMILARITY_THRESHOLD:
        return {
            "response": best_match["response"],
            "similarity": float(max_similarity),
            "original_query": best_match["query"]
        }
    
    return None

def call_llm(query):
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Provide concise, accurate answers."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            model= "llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling LLM: {str(e)}"


def store_in_cache(query, embedding, response):
    cache_key = f"embedding:{abs(hash(query))}"
    
    cache_data = {
        "query": query,
        "embedding": embedding.tolist(),
        "response": response
    }
    
    redis_client.setex(cache_key, 3 * 24 * 3600, json.dumps(cache_data))
    # expiry in 3 days

# API Endpoints
@app.get("/")
async def root():
    
    return {
        "status": "API Running",
        "endpoints": ["/query", "/stats", "/cache/clear", "/cache/list"]
    }

@app.post("/query")
async def process_query(request: dict):
    """
    Main endpoint: Process a query with caching
    
    Request body: {"query": "Your question here"}
    """
    start_time = time.time()
    
    query = request.get("query", "").strip()
    if not query:
        return {"error": "Query cannot be empty"}
    
    metrics["total_requests"] += 1
    
    query_embedding = get_embedding_model().encode(query)

    
    cached_result = check_cache(query_embedding, query)
    
    if cached_result:

        metrics["cache_hits"] += 1
        
        response_time = (time.time() - start_time) * 1000
        
        return {
            "response": cached_result["response"],
            "from_cache": True,
            "similarity": round(cached_result["similarity"], 4),
            "original_query": cached_result["original_query"],
            "response_time_ms": round(response_time, 2)
        }
    
    
    metrics["cache_misses"] += 1
    llm_response = call_llm(query)
    
    
    store_in_cache(query, query_embedding, llm_response)
    
    response_time = (time.time() - start_time) * 1000
    
    return {
        "response": llm_response,
        "from_cache": False,
        "response_time_ms": round(response_time, 2)
    }

@app.get("/stats")
async def get_stats():
    """Get cache statistics"""
    total_cached = len(redis_client.keys("embedding:*"))
    cache_hit_rate = (metrics["cache_hits"] / metrics["total_requests"] * 100) if metrics["total_requests"] > 0 else 0
     
    return {
        "total_requests": metrics["total_requests"],
        "cache_hits": metrics["cache_hits"],
        "cache_misses": metrics["cache_misses"],
        "cache_hit_rate": round(cache_hit_rate, 2),
        "total_cached_queries": total_cached,
    }

@app.delete("/cache/clear")
async def clear_cache():
    redis_client.flushdb()
    return {"message": "Cache cleared successfully"}

@app.get("/cache/list")
async def list_cached():
    keys = redis_client.keys("embedding:*")
    queries = []
    
    for key in keys[:20]: 
        data = json.loads(redis_client.get(key))
        queries.append({
            "query": data["query"],
            "response_preview": data["response"][:100] + "..." if len(data["response"]) > 100 else data["response"]
        })
    
    return {
        "cached_queries": queries,
        "total_count": len(keys)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)