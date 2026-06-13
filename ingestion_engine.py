import os
import sys
import json
from ingestion import *

# For backwards compatibility with tests and direct module access:
import ingestion.config as config
from ingestion.novelty import NoveltyMatrix
from ingestion.features import DocResult

if __name__ == "__main__":
    target_data_json_path = "/mnt/user-data/uploads/staged_repositories.json"
    if not os.path.exists(target_data_json_path):
        target_data_json_path = os.path.join(os.path.dirname(__file__), "staged_repositories.json")
        
    if not os.path.exists(target_data_json_path):
        print("⚠️  Context Warning: JSON repository source data profile asset not found. Initializing pipeline standalone test modules...")
        smoke_payload_mock = [
            {"id": "facebook/react",       "star_count": 246000, "pushed_days_ago": 4, 
             "mentionable_users_count": 2, "primary_language": "JavaScript", 
             "readme_length": 5317, "readme_to_codebase_ratio": 0.000005, 
             "extracted_paragraphs": ["React is a JavaScript library for building component based rich interactive user interfaces for web platforms."], 
             "delta_3d": 370, "delta_7d": 1048, "delta_30d": 3499},
            {"id": "vuejs/vue",            "star_count": 208000, "pushed_days_ago": 7, 
             "mentionable_users_count": 3, "primary_language": "JavaScript", 
             "readme_length": 3100, "readme_to_codebase_ratio": 0.000008, 
             "extracted_paragraphs": ["Vue.js is a progressive modular component-based JavaScript framework for engineering clean dynamic user interface layouts."], 
             "delta_3d": 120, "delta_7d": 340, "delta_30d": 1100},
            {"id": "bitcoin/bitcoin",       "star_count": 77000,  "pushed_days_ago": 1, 
             "mentionable_users_count": 5, "primary_language": "C++", 
             "readme_length": 8000, "readme_to_codebase_ratio": 0.00002, 
             "extracted_paragraphs": ["Bitcoin Core is the foundational decentralized open-source p2p cryptocurrency blockchain node implementation engine."], 
             "delta_3d": 40,  "delta_7d": 110, "delta_30d": 380},
            {"id": "anomalous/react-shallow-wrapper", "star_count": 12, "pushed_days_ago": 2, 
             "mentionable_users_count": 1, "primary_language": "JavaScript", 
             "readme_length": 5100, "readme_to_codebase_ratio": 0.000005, 
             "extracted_paragraphs": ["React is a JavaScript library for building component based rich interactive user interfaces for web platforms."], 
             "delta_3d": 2, "delta_7d": 5, "delta_30d": 10}
        ]
        store = CorpusStore()
        batch_output_nodes = ingest_batch(smoke_payload_mock, corpus_store=store, verbose=True)
        print_batch_summary(batch_output_nodes, corpus_timeline=store.get_timeline())
        sys.exit(0)
        
    with open(target_data_json_path) as data_file_descriptor:
        active_staged_payload = json.load(data_file_descriptor)
        
    print(f"🚀 Osiris Core Framework Booted. Processing comprehensive stream matrix tracking across {len(active_staged_payload)} staged inputs.\\n")
    
    # Executing calculation streams directly across ALL 97 entries in data asset arrays
    store = CorpusStore()
    resolved_batch_matrix = ingest_batch(active_staged_payload, corpus_store=store, verbose=True)
    print_batch_summary(resolved_batch_matrix, corpus_timeline=store.get_timeline())
