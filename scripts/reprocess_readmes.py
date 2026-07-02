import os
import sys

# Resolve project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv()

from database import PostgreSQLConnector
from utils.groq_client import generate_readme_markdown

def main():
    print("🚀 Starting Reprocess READMEs Test Script...")
    
    # 1. Initialize DB Connector
    db = PostgreSQLConnector()
    if not db.enabled or not db.verify_connection():
        print("❌ Error: Could not connect to the database. Verify DATABASE_URL in .env.")
        return
        
    db.init_db()
    
    # 2. Fetch 3 repositories that don't have readme_markdown populated yet
    print("Fetching up to 3 repositories from Supabase with empty 'readme_markdown'...")
    conn = db.connect()
    cursor = conn.cursor()
    
    try:
        # We look for repos that have descriptions/summaries to clean/re-structure
        cursor.execute(
            "SELECT repo_id, full_name, description, readme_summary "
            "FROM Repo "
            "WHERE readme_markdown IS NULL OR readme_markdown = '' "
            "LIMIT 3;"
        )
        repos = cursor.fetchall()
        
        if not repos:
            print("No repositories found with empty 'readme_markdown'. Fetching 3 random repositories instead...")
            cursor.execute(
                "SELECT repo_id, full_name, description, readme_summary "
                "FROM Repo "
                "LIMIT 3;"
            )
            repos = cursor.fetchall()
            
        if not repos:
            print("❌ No repositories found in the Repo table.")
            return
            
        print(f"Found {len(repos)} repositories to process.")
        
        # 3. Generate and save Markdown for each repo
        for repo_id, full_name, description, readme_summary in repos:
            print(f"\nProcessing: {full_name} (ID: {repo_id})")
            
            # Form clean source text from description or summary
            source_text = readme_summary or description
            if not source_text or not source_text.strip():
                source_text = f"Repository {full_name} is a GitHub project."
                
            print(f"Generating Markdown from text (length {len(source_text)} chars)...")
            markdown_out = generate_readme_markdown(source_text)
            
            if markdown_out:
                print(f"✅ Generated {len(markdown_out)} chars of structured Markdown.")
                
                # Update row in database
                cursor.execute(
                    "UPDATE Repo "
                    "SET readme_markdown = %s, updated_at = NOW() "
                    "WHERE repo_id = %s;",
                    (markdown_out, repo_id)
                )
                conn.commit()
                print(f"✅ Successfully updated {full_name} in Supabase!")
                
                # Print a small preview
                preview = "\n".join(markdown_out.splitlines()[:6])
                print(f"--- Preview ---\n{preview}\n...")
            else:
                print("❌ Failed to generate Markdown (check API key or network limits).")
                
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()
        print("\n🎉 Reprocess READMEs Test Script Completed.")

if __name__ == "__main__":
    main()
