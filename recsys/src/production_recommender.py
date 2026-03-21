"""
Production Recommender for Web Apps
Supports new users and new products using feature-based models.

Models used:
  - Content: Always works (text embeddings)
  - Two-Tower: Feature-based (works for new items if user exists in training)
  - ALS: Skipped for new items (ID-based limitation)
"""
import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

from recsys.src.embeddings import build_faiss_index, build_item_matrix

ART = Path("recsys/artifacts")
DATA = Path("recsys/data")

# =====================================================
# Model Definition (must match training)
# =====================================================

class ContentTwoTower(nn.Module):
    def __init__(self, feature_dim=384, embed_dim=128):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embed_dim),
            nn.ReLU(),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embed_dim),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    
    def forward(self, u_vec, i_vec):
        u = self.user_tower(u_vec)
        i = self.item_tower(i_vec)
        x = torch.abs(u - i)
        return torch.sigmoid(self.scorer(x).squeeze(1))

# =====================================================
# Lazy Loading
# =====================================================

_model = None
_device = None
DEFAULT_CANDIDATE_MULTIPLIER = 10
DEFAULT_MIN_CANDIDATES = 100

def get_model():
    global _model, _device
    if _model is None:
        _device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        _model = ContentTwoTower(feature_dim=384, embed_dim=128).to(_device)
        model_path = ART / "content_two_tower_hm.pt"
        _model.load_state_dict(torch.load(model_path, map_location=_device))
        _model.eval()
    return _model, _device

# =====================================================
# Core Functions
# =====================================================

def embed_products(products: List[dict]) -> np.ndarray:
    """
    Embed products from title + description.
    products: [{"title": str, "description": str}, ...]
    Returns: (N, 384) normalized float32 array
    """
    df = pd.DataFrame(products)
    if "title" not in df.columns and "name" in df.columns:
        df["title"] = df["name"]
    if "title" not in df.columns:
        df["title"] = ""
    if "description" not in df.columns:
        df["description"] = ""
    return build_item_matrix(df)


def build_user_vector(liked_products: List[dict]) -> np.ndarray:
    """
    Build user vector from products they liked.
    liked_products: [{"title": ..., "description": ...}, ...]
    Returns: (384,) normalized vector
    """
    if not liked_products:
        return None
    X = embed_products(liked_products)
    vec = np.mean(X, axis=0)
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec.astype("float32")


def score_content(user_vec: np.ndarray, item_vec: np.ndarray) -> float:
    """Content score: dot product (cosine similarity)."""
    return float(np.dot(user_vec, item_vec))


def score_two_tower(user_vec: np.ndarray, item_vec: np.ndarray) -> float:
    """Two-Tower neural score."""
    model, device = get_model()
    
    u = torch.tensor(user_vec, dtype=torch.float32, device=device).unsqueeze(0)
    i = torch.tensor(item_vec, dtype=torch.float32, device=device).unsqueeze(0)
    
    with torch.no_grad():
        return float(model(u, i).cpu().numpy()[0])


def score_hybrid(user_vec: np.ndarray, item_vec: np.ndarray,
                 w_content=0.5, w_tt=0.5) -> float:
    """Hybrid: content + local Two-Tower score."""
    s_content = score_content(user_vec, item_vec)
    s_tt = score_two_tower(user_vec, item_vec)
    s_tt_norm = 2 * (s_tt - 0.5)
    return w_content * s_content + w_tt * s_tt_norm


def recommend(
    liked_products: List[dict],
    catalog_products: List[dict],
    catalog_ids: List,
    top_k: int = 20,
    strategy: str = "hybrid",
    exclude_ids: Optional[List] = None,
) -> List[dict]:
    """
    End-to-end recommendation for new users and new products.
    
    Args:
        liked_products: What the user likes [{"title": ..., "description": ...}]
        catalog_products: Your product catalog
        catalog_ids: IDs for each catalog product
        top_k: How many to recommend
        strategy: "content", "two_tower", or "hybrid"
        exclude_ids: IDs to exclude (e.g. already purchased)
    
    Returns:
        [{"product_id": ..., "score": ...}, ...]
    """
    # Build user vector
    user_vec = build_user_vector(liked_products)
    if user_vec is None:
        return []
    
    # Embed catalog
    catalog_embs = embed_products(catalog_products)
    
    # Retrieve a reasonably small candidate set with FAISS, then rerank.
    candidate_k = min(
        len(catalog_ids),
        max(top_k * DEFAULT_CANDIDATE_MULTIPLIER, DEFAULT_MIN_CANDIDATES),
    )
    faiss_index = build_faiss_index(catalog_embs)
    scores, indices = faiss_index.search(user_vec.reshape(1, -1).astype("float32"), candidate_k)
    candidate_pairs = [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]

    # Score function
    if strategy == "content":
        score_fn = lambda u, i: score_content(u, i)
    elif strategy == "two_tower":
        score_fn = lambda u, i: score_two_tower(u, i)
    else:  # hybrid
        score_fn = lambda u, i: score_hybrid(u, i)
    
    # Rerank the FAISS candidates.
    exclude_set = set(exclude_ids) if exclude_ids else set()
    results = []
    for i, faiss_score in candidate_pairs:
        if len(results) >= top_k:
            break
        pid = catalog_ids[i]
        if pid not in exclude_set:
            final_score = faiss_score if strategy == "content" else score_fn(user_vec, catalog_embs[i])
            results.append({"product_id": pid, "score": float(final_score)})
    
    return results


# =====================================================
# Example Usage
# =====================================================

if __name__ == "__main__":
    # Example
    liked = [
        {"title": "Blue cotton t-shirt", "description": "Casual summer wear"},
        {"title": "White sneakers", "description": "Running shoes"},
    ]
    
    catalog = [
        {"title": "Red polo shirt", "description": "Smart casual"},
        {"title": "Black jeans", "description": "Slim fit denim"},
        {"title": "Gray hoodie", "description": "Cozy streetwear"},
    ]
    catalog_ids = ["prod_001", "prod_002", "prod_003"]
    
    print("[example] Building user vector from likes...")
    user_vec = build_user_vector(liked)
    print(f"User vector shape: {user_vec.shape}")
    
    print("\n[example] Embedding catalog...")
    catalog_embs = embed_products(catalog)
    print(f"Catalog embeddings shape: {catalog_embs.shape}")
    
    print("\n[example] Recommendations (content-only):")
    recs = recommend(liked, catalog, catalog_ids, top_k=3, strategy="content")
    for r in recs:
        print(f"  {r['product_id']}: {r['score']:.4f}")
    
    print("\n[example] Recommendations (hybrid):")
    recs = recommend(liked, catalog, catalog_ids, top_k=3, strategy="hybrid")
    for r in recs:
        print(f"  {r['product_id']}: {r['score']:.4f}")
