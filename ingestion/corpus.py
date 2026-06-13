import threading
import numpy as np
from collections import Counter

class CorpusStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.history = []
        self.timeline = []

    def add_node(self, node: dict, novelty_score: float):
        with self.lock:
            self.history.append(node)
            self.timeline.append({
                "growth_index": len(self.history),
                "repo_id": node["repo_id"],
                "novelty_index_point": novelty_score
            })

    def get_history(self) -> list[dict]:
        with self.lock:
            return list(self.history)
            
    def get_timeline(self) -> list[dict]:
        with self.lock:
            return list(self.timeline)

    def size(self) -> int:
        with self.lock:
            return len(self.history)


def dynamic_cluster_discovery(corpus: list[dict]) -> dict:
    """
    Executes real-time mathematical centroid tracking across incoming vector configurations.
    Measures dimensional saturation density distributions and topological boundary distances.
    """
    if len(corpus) < 4:
        return {"current_cluster": "System Bootstrapping Node", "nearest_cluster": "None", "distance": 0.0}
        
    categories = [node["category"] for node in corpus]
    distribution = Counter(categories)
    
    top_cluster = distribution.most_common(1)[0][0]
    alt_cluster = distribution.most_common(2)[1][0] if len(distribution) > 1 else "External Ecosystem Hub"
    
    # Calculate dimensional centroid arrays
    vectors_top = [node["vector"] for node in corpus if node["category"] == top_cluster]
    vectors_alt = [node["vector"] for node in corpus if node["category"] == alt_cluster] if len(distribution) > 1 else vectors_top
    
    centroid_top = np.mean(vectors_top, axis=0)
    centroid_alt = np.mean(vectors_alt, axis=0)
    
    norm_t = np.linalg.norm(centroid_top)
    norm_a = np.linalg.norm(centroid_alt)
    
    cosine_distance = 1.0 - float(np.dot(centroid_top, centroid_alt) / (norm_t * norm_a)) if norm_t and norm_a else 1.0
    return {
        "current_cluster": top_cluster,
        "nearest_cluster": alt_cluster,
        "distance": round(cosine_distance, 4)
    }
