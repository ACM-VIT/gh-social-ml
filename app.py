import uvicorn
from fastapi import FastAPI, Body, status
from fastapi.middleware.cors import CORSMiddleware
from inference.feed_assembly import FeedAssemblySystem

# 1. Initialize the FastAPI Web Service Application Space
app = FastAPI(
    title="GH-Social ML Assembly Engine",
    description="Internal ML service serving freshness and exploration injections.",
    version="1.0.0"
)

# 2. Add safe local and production guardrails (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/internal/ml/assemble-feed", status_code=status.HTTP_200_OK)
async def assemble_feed_endpoint(candidates: list = Body(..., embed=True)):
    """
    Receives the 15 pre-ranked JSON objects from the main backend,
    applies time-decay freshness boosts and bottom-tier exploration shuffling,
    and returns the final ordered sequence of string IDs.
    """
    try:
        # Pass the candidates block down to your verified module
        ordered_ids = FeedAssemblySystem.process_feed_assembly(candidates, target_size=15)
        return {"rankedRepoIds": ordered_ids}
        
    except Exception as err:
        # Fail-soft fallback so an unexpected parsing error won't crash the web worker process
        return {
            "error": "Internal Processing Exception",
            "details": str(err),
            "rankedRepoIds": [item.get("repo_id") for item in candidates][:15] # Return base order as recovery
        }

if __name__ == "__main__":
    # Run the Uvicorn worker locally on Port 8000
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)