import os
import sys

# Resolve project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv()

from utils.gemma_client import generate_readme_markdown

def main():
    print("🚀 Running Gemma README Markdown Generation Test...")
    
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMMA_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY is not set in your .env file!")
        print("Please add it to your .env: GEMINI_API_KEY=your_key_here")
        return
        
    print(f"API Key detected. Using model: {os.getenv('GEMMA_MODEL_ID', 'gemma-4-E4B-it')}")
    
    mock_clean_text = (
        "project name: osiris-lite\n\n"
        "description: this is a fast lightweight library for building vector embeddings "
        "and performing similarity search on python data dictionaries.\n\n"
        "features:\n"
        "- simple and fast\n"
        "- persistent storage support\n"
        "- zero runtime dependencies except requests\n\n"
        "how to run:\n"
        "run python3 main.py with appropriate options."
    )
    
    print("\n--- Input Clean Text ---")
    print(mock_clean_text)
    print("------------------------")
    
    print("\nCalling Gemma model via Gemini Cloud API (this may take a few seconds)...")
    markdown_out = generate_readme_markdown(mock_clean_text)
    
    if markdown_out:
        print("\n✅ SUCCESS: Gemma generated the following Markdown:")
        print("------------------------")
        print(markdown_out)
        print("------------------------")
    else:
        print("\n❌ FAILURE: Gemma API did not return any markdown output. Check logs above for details.")

if __name__ == "__main__":
    main()
