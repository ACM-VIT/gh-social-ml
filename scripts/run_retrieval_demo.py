import logging
import random
from dotenv import load_dotenv

from database.connector import PostgreSQLConnector
from retrieval.candidate_retriever import CandidateRetriever

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
logger = logging.getLogger("demo.retrieval")

def main():
    load_dotenv()
    
    logger.info("Initializing database connector...")
    db = PostgreSQLConnector()
    if not db.enabled or not db.verify_connection():
        logger.error("PostgreSQL is not running or not configured properly. Check DATABASE_URL in .env.")
        return
        
    logger.info("Initializing CandidateRetriever...")
    retriever = CandidateRetriever(db_connector=db)
    
    # Generate a random 384-dimensional normalized vector for the mock user persona
    random.seed(42)
    raw_vector = [random.uniform(-1.0, 1.0) for _ in range(384)]
    # L2 normalize
    norm = sum(x*x for x in raw_vector) ** 0.5
    user_embedding = [x / norm for x in raw_vector]
    
    user_interests = ["AI/ML", "Backend", "Frontend"]
    
    logger.info("Starting candidate retrieval query...")
    logger.info(f"User Interests: {user_interests}")
    
    candidates = retriever.retrieve_candidates(
        user_embedding=user_embedding,
        user_interests=user_interests
    )
    
    logger.info("=" * 60)
    logger.info(f"Retrieved {len(candidates)} candidates in total.")
    logger.info("=" * 60)
    
    # Print semantic candidates
    semantic_cand = [c for c in candidates if c.get("retrieval_source") == "semantic"]
    logger.info(f"Semantic candidates count: {len(semantic_cand)}")
    for i, c in enumerate(semantic_cand[:5], 1):
        logger.info(f"  {i}. score={c.get('retrieval_score', 0):.4f} repo={c.get('full_name')} (ID: {c.get('repo_id')})")
        
    # Print trending candidates
    trending_cand = [c for c in candidates if c.get("retrieval_source") == "trending"]
    logger.info(f"Trending candidates count: {len(trending_cand)}")
    for i, c in enumerate(trending_cand[:5], 1):
        logger.info(f"  {i}. stars={c.get('star_count', 0)} repo={c.get('full_name')} (ID: {c.get('repo_id')})")
        
    # Print fallback candidates
    fallback_cand = [c for c in candidates if c.get("retrieval_source") == "fallback"]
    if fallback_cand:
        logger.info(f"Fallback candidates count: {len(fallback_cand)}")
        for i, c in enumerate(fallback_cand[:5], 1):
            logger.info(f"  {i}. repo={c.get('full_name')}")

if __name__ == "__main__":
    main()
