import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv()

from database import PostgreSQLConnector
from main import index_approved_repositories

def main():
    db = PostgreSQLConnector()
    if not db.enabled or not db.verify_connection():
        print("DB not enabled")
        return
        
    conn = db.connect()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT repo_id, full_name, description, primary_language, topics, 
               star_count, forks_count, pr_count, created_at, updated_at 
        FROM Repo 
        ORDER BY star_count DESC 
        LIMIT 150;
    """)
    rows = cursor.fetchall()
    
    approved = []
    for row in rows:
        topics = row[4]
        if isinstance(topics, str):
            import json
            try:
                topics = json.loads(topics)
            except:
                topics = []
                
        approved.append({
            # repo_id is the backend-issued canonical UUID. full_name is only
            # searchable/display metadata and must never become the point key.
            "id": str(row[0]),
            "repo_id": str(row[0]),
            "full_name": row[1],
            "description": row[2] or "",
            "primary_language": row[3] or "Unknown",
            "topics": topics or [],
            "star_count": row[5] or 0,
            "fork_count": row[6] or 0,
            "open_issues_count": row[7] or 0,
            "extracted_paragraphs": [row[2] or ""], # simple fallback for readme
        })
        
    print(f"Found {len(approved)} repos in Postgres. Indexing to Qdrant...")
    indexed = index_approved_repositories(approved)
    print(f"Successfully indexed {len(indexed)} repos to Qdrant!")

if __name__ == "__main__":
    main()
