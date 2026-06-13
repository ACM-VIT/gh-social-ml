from dataclasses import dataclass

from .config import NOVELTY_WEIGHTS, NOVELTY_THRESHOLD, DUPLICATE_SIMILARITY_THRESHOLD, WRAPPER_SIMILARITY_THRESHOLD, TOP_K_COMPARISONS

@dataclass
class NoveltyMatrix:
    final: float
    semantic_penalty: float
    category_penalty: float 
    tech_stack_penalty: float
    activity_penalty: float
    top_k: list[dict]
    anomaly_tag: str
    explanation: str

def compute_multi_dimensional_novelty(
    repo_id: str,
    target_cat: str,
    activity_val: float,
    target_tags: list[str],
    qdrant_hits: list[dict],
    corpus_volume: int
) -> NoveltyMatrix:
    """
    Executes a high-dimensional functional convergence test against the active top-K neighborhood.
    Returns a unified tensor evaluation object representing absolute node novelty.
    """
    if corpus_volume == 0 or not qdrant_hits:
        return NoveltyMatrix(1.0, 0, 0, 0, 0, [], "NONE", "Genesis Node: Zero Collision Density in Target Corpus")

    top_neighbors = sorted(qdrant_hits, key=lambda x: x["sim"], reverse=True)[:TOP_K_COMPARISONS]
    
    anomalies = []
    agg_semantic, agg_cat, agg_tech, agg_act = 0.0, 0.0, 0.0, 0.0
    
    for neighbor in top_neighbors:
        sim_val = neighbor["sim"]
        
        if sim_val > DUPLICATE_SIMILARITY_THRESHOLD:
            anomalies.append(f"CRITICAL VECTOR COLLISION: {neighbor['repo_id']} ({sim_val:.2f})")
        elif sim_val > WRAPPER_SIMILARITY_THRESHOLD:
            anomalies.append(f"WRAPPER COPY WARNING: {neighbor['repo_id']} ({sim_val:.2f})")
            
        semantic_p = min(sim_val * 1.2, 1.0)
        cat_p = 1.0 if target_cat == neighbor.get("category") else 0.2
        
        neighbor_tags = neighbor.get("tags", [])
        shared_tags = set(target_tags).intersection(set(neighbor_tags))
        tech_p = len(shared_tags) / max(len(target_tags), 1) if target_tags else 0.5
        
        act_diff = abs(activity_val - neighbor.get("activity", 0.5))
        act_p = 1.0 - act_diff
        
        # Exponential decay weights nearest neighbors more heavily in the penalty calculation
        rank_weight = 1.0 / (top_neighbors.index(neighbor) + 1)
        
        agg_semantic += semantic_p * rank_weight
        agg_cat += cat_p * rank_weight
        agg_tech += tech_p * rank_weight
        agg_act += act_p * rank_weight
        
    normalization_factor = sum(1.0 / (i + 1) for i in range(len(top_neighbors)))
    
    penalty_semantic = (agg_semantic / normalization_factor) * NOVELTY_WEIGHTS["semantic"]
    penalty_cat = (agg_cat / normalization_factor) * NOVELTY_WEIGHTS["category"]
    penalty_tech = (agg_tech / normalization_factor) * NOVELTY_WEIGHTS["tech_stack"]
    penalty_act = (agg_act / normalization_factor) * NOVELTY_WEIGHTS["activity"]
    
    total_penalty = penalty_semantic + penalty_cat + penalty_tech + penalty_act
    final_novelty = max(0.0, 1.0 - total_penalty)
    
    # Floor Boost for small corpuses (prevents premature rejection)
    if corpus_volume < 100:
        boost = (100 - corpus_volume) * 0.002
        final_novelty = min(final_novelty + boost, 1.0)

    anomaly_str = " | ".join(anomalies) if anomalies else "NONE"
    
    expl = (f"  [Novelty Sub-Space Diagnostics]\n"
            f"    Semantic Gravity : -{penalty_semantic:.4f}\n"
            f"    Taxonomy Gravity : -{penalty_cat:.4f}\n"
            f"    Tech Stack Orbit : -{penalty_tech:.4f}\n"
            f"    Activity Orbit   : -{penalty_act:.4f}")
            
    return NoveltyMatrix(
        final=round(final_novelty, 4),
        semantic_penalty=round(penalty_semantic, 4),
        category_penalty=round(penalty_cat, 4),
        tech_stack_penalty=round(penalty_tech, 4),
        activity_penalty=round(penalty_act, 4),
        top_k=top_neighbors,
        anomaly_tag=anomaly_str,
        explanation=expl
    )


def calculate_quadrant(novelty_score: float, trend_velocity: float) -> tuple[str, str]:
    """Projects combined parameters into the 2x2 Action/Decision Matrix Space."""
    high_novelty = novelty_score >= NOVELTY_THRESHOLD
    high_traction = trend_velocity >= 0.40
    
    if high_novelty and high_traction:
        return "🔥 Viral Rockets", "APPROVED"
    elif high_novelty and not high_traction:
        return "💎 Hidden Gems", "APPROVED"
    elif not high_novelty and high_traction:
        return "⚠️ Copycats / Clones", "REJECTED"
    else:
        return "💤 Dormant Ecosystem Nodes", "REJECTED"
