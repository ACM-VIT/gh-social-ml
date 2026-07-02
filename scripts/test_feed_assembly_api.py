import os
import sys
from datetime import datetime, timezone, timedelta

# Resolve project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def main():
    print("🚀 Starting Feed Assembly Endpoint Test...")
    
    # We will create 15 mock candidates.
    # Repo 1 is brand new (freshness boost should push it to rank 1).
    # Repos 2-15 are older.
    now = datetime.now(timezone.utc)
    candidates = []
    
    # 1. Fresh repo (base score 10.0)
    candidates.append({
        "repo_id": "repo-fresh-1",
        "final_score": 10.0,
        "created_at": (now - timedelta(hours=1)).isoformat()
    })
    
    # 2. Older repos (base scores starting at 10.1 and descending)
    for i in range(2, 16):
        candidates.append({
            "repo_id": f"repo-old-{i}",
            "final_score": 10.3 - (i * 0.1),  # repo-old-2 score is 10.1
            "created_at": (now - timedelta(days=10)).isoformat()
        })
        
    payload = {"candidates": candidates}
    
    print("\n--- Input Candidates Summary ---")
    for i, c in enumerate(candidates, 1):
        print(f"Candidate {i:<2}: ID={c['repo_id']:<12} Base Score={c['final_score']:<5} Created={c['created_at']}")
    print("--------------------------------")
    
    print("\nSending POST request to '/api/internal/ml/assemble-feed'...")
    response = client.post("/api/internal/ml/assemble-feed", json=payload)
    
    print(f"HTTP Response Status: {response.status_code}")
    
    if response.status_code == 200:
        res_json = response.json()
        print("✅ SUCCESS: Feed assembled successfully.")
        
        ranked_ids = res_json["rankedRepoIds"]
        print("\n--- Output Ranked Order (IDs) ---")
        for idx, rid in enumerate(ranked_ids, 1):
            status = ""
            if rid == "repo-fresh-1":
                status = " (🔥 Boosted to top due to freshness!)"
            elif idx > 10:
                status = " (🎲 Exploration / shuffle tail)"
            print(f"  {idx:<2}. {rid}{status}")
        print("--------------------------------")
    else:
        print(f"❌ FAILURE: Endpoint returned status {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    main()
