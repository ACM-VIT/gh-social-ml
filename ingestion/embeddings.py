import re
import numpy as np

from .config import EMBEDDING_MODEL, HYBRID_WEIGHTS, EMBEDDING_DIM

_fallback_vocab: dict[str, int] = {}
_FALLBACK_DIM = EMBEDDING_DIM

try:
    from sentence_transformers import SentenceTransformer
    _ST_OK = True
except ImportError:
    _ST_OK = False

def _generate_fallback_vector(text: str) -> np.ndarray:
    """Mathematical fallback string hash distribution mapping if SentenceTransformer fails."""
    words = re.findall(r"[a-z]{2,}", text.lower())
    vector = np.zeros(_FALLBACK_DIM, dtype=np.float32)
    for word in words:
        if word not in _fallback_vocab:
            if len(_fallback_vocab) < _FALLBACK_DIM:
                _fallback_vocab[word] = len(_fallback_vocab)
            else:
                continue
        vector[_fallback_vocab[word]] += 1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

_loaded_transformer = None
def _generate_embedding(text: str) -> np.ndarray:
    """Generates normalized vector matrices using active neural pipeline topologies."""
    global _loaded_transformer
    if _ST_OK:
        try:
            if _loaded_transformer is None:
                _loaded_transformer = SentenceTransformer(EMBEDDING_MODEL)
            dense_vector = _loaded_transformer.encode(text, normalize_embeddings=True)
            return dense_vector.astype(np.float32)
        except Exception:
            pass
    return _generate_fallback_vector(text)

def generate_hybrid_embedding(repo: dict, tags: list[str], category: str, summary: str) -> np.ndarray:
    """
    Executes a Multi-Field Component Vector Fusion Model.
    Each feature field is embedded within its own localized latent context window 
    before linear combination via static projection coefficients to maintain vector space properties.
    """
    paragraphs = repo.get("extracted_paragraphs", [])
    raw_readme = " ".join(re.sub(r"<[^>]+>", " ", p) for p in paragraphs)[:1500]
    primary_lang = repo.get("primary_language", "unknown")
    repo_name = repo.get("id", "unknown/repo").split("/")[-1]

    # Distinct separate functional spaces
    fields = {
        "readme":      raw_readme,
        "description": summary,
        "topics":      " ".join(tags) + f" {repo_name}",
        "category":    category,
        "language":    primary_lang
    }

    # Generate isolated sub-vectors
    sub_vectors = {k: _generate_embedding(v if v.strip() else "missing") for k, v in fields.items()}
    
    # Weighted integration framework execution (Maintains geometric alignment bounds)
    fused_matrix = np.zeros(_FALLBACK_DIM, dtype=np.float32)
    for key, coefficient in HYBRID_WEIGHTS.items():
        fused_matrix += coefficient * sub_vectors[key]
        
    final_norm = np.linalg.norm(fused_matrix)
    return (fused_matrix / final_norm if final_norm > 0 else fused_matrix).astype(np.float32)
