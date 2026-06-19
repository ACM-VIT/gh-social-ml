import logging
import json
from dotenv import load_dotenv

# Import connector
from database.connector import PostgreSQLConnector

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
logger = logging.getLogger("db.sync")

def sync():
    load_dotenv()
    
    # 1. Connect to Local PostgreSQL
    local_url = "postgresql://medhanshadhlakha@localhost:5432/gh_social"
    logger.info(f"Connecting to Local PostgreSQL: {local_url}")
    local_db = PostgreSQLConnector(database_url=local_url)
    if not local_db.verify_connection():
        logger.error("Failed to connect to local database.")
        return
        
    # 2. Connect to Supabase
    supabase_url = "postgresql://postgres.yauetbynmjslennjfczt:Github2405!@aws-1-ap-southeast-2.pooler.supabase.com:6543/postgres"
    logger.info("Connecting to Supabase PostgreSQL...")
    supabase_db = PostgreSQLConnector(database_url=supabase_url)
    if not supabase_db.verify_connection():
        logger.error("Failed to connect to Supabase database.")
        return
        
    # 3. Read rows from Local PostgreSQL
    logger.info("Reading repositories from local database...")
    local_conn = local_db.connect()
    local_cursor = local_conn.cursor()
    
    columns = [
        "repo_id", "github_repo_url", "owner_id", "repo_name", "full_name",
        "description", "primary_language", "language_used", "topics",
        "readme_summary", "star_count", "likes_count", "comments_count",
        "saves_count", "views_count", "forks_count", "pr_count",
        "created_at", "updated_at"
    ]
    
    query = f"SELECT {', '.join(columns)} FROM Repo;"
    local_cursor.execute(query)
    rows = local_cursor.fetchall()
    local_conn.close()
    
    logger.info(f"Retrieved {len(rows)} repositories from local database.")
    if not rows:
        logger.warning("No records to sync.")
        return
        
    # 4. Write/Upsert to Supabase
    logger.info("Initializing Supabase database schemas if not exist...")
    supabase_db.init_db()
    
    supabase_conn = supabase_db.connect()
    supabase_cursor = supabase_conn.cursor()
    
    logger.info("Syncing records to Supabase...")
    
    count = 0
    batch_size = 50
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        try:
            params = []
            values_clauses = []
            for row in batch:
                row_list = list(row)
                # Parse JSON columns if they are not already parsed dicts/lists
                # (column idx 7 is language_used, 8 is topics)
                for idx in [7, 8]:
                    if isinstance(row_list[idx], (dict, list)):
                        row_list[idx] = json.dumps(row_list[idx])
                params.extend(row_list)
                
                placeholders = [
                    "CAST(%s AS jsonb)" if col in ("language_used", "topics") else "%s"
                    for col in columns
                ]
                values_clauses.append(f"({', '.join(placeholders)})")
                
            upsert_query = f"""
            INSERT INTO Repo ({', '.join(columns)})
            VALUES {', '.join(values_clauses)}
            ON CONFLICT (github_repo_url) DO UPDATE SET
                repo_name = EXCLUDED.repo_name,
                full_name = EXCLUDED.full_name,
                description = EXCLUDED.description,
                primary_language = EXCLUDED.primary_language,
                language_used = EXCLUDED.language_used,
                topics = EXCLUDED.topics,
                readme_summary = EXCLUDED.readme_summary,
                star_count = EXCLUDED.star_count,
                likes_count = EXCLUDED.likes_count,
                comments_count = EXCLUDED.comments_count,
                saves_count = EXCLUDED.saves_count,
                views_count = EXCLUDED.views_count,
                forks_count = EXCLUDED.forks_count,
                pr_count = EXCLUDED.pr_count,
                updated_at = EXCLUDED.updated_at;
            """
            supabase_cursor.execute(upsert_query, params)
            supabase_conn.commit()
            count += len(batch)
            logger.info(f"  Synced {count}/{len(rows)} rows...")
        except Exception as exc:
            supabase_conn.rollback()
            logger.error(f"Failed to sync batch starting at index {i}: {exc}")
            
    supabase_conn.close()
    
    # 5. Verify Supabase Count
    logger.info("Verification: checking total count in Supabase...")
    final_count = supabase_db.get_repo_count()
    logger.info(f"Verification complete! Supabase database has {final_count} repositories.")

if __name__ == "__main__":
    sync()
