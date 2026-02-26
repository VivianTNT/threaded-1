"""
Unified recommendation engine for HM + RR datasets.

Strategy selection:
  - content: FAISS-based content filtering only
  - collab: ALS collaborative filtering (if artifact exists), else content
  - hybrid: content candidates reranked with hybrid_ranker (if available)
"""
import numpy as np
import joblib
from pathlib import Path

ART = Path("recsys/artifacts")
RR_USER_OFFSET = 1_000_000_000
HM_MAX_ID = 3_000_000
DATASETS = ("hm", "rr")


def _load_faiss(dataset: str):
    pack = joblib.load(ART / f"faiss_items_{dataset}.joblib")
    return pack["index"], pack["item_ids"], pack["X"], pack["row_map"]


def _load_user_vectors(dataset: str):
    return joblib.load(ART / f"user_vectors_{dataset}.joblib")


def _load_als(dataset: str):
    path = ART / f"als_{dataset}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def _load_hybrid_ranker():
    try:
        from recsys.src.models.hybrid_ranker import score_hybrid as hybrid_score_fn
        return hybrid_score_fn
    except Exception:
        return None


_faiss_packs = {}
_user_vectors = {}
_als_packs = {}
_hybrid_score_fn = None


def get_faiss(dataset: str):
    if dataset not in _faiss_packs:
        _faiss_packs[dataset] = _load_faiss(dataset)
    return _faiss_packs[dataset]


def get_user_vectors(dataset: str):
    if dataset not in _user_vectors:
        _user_vectors[dataset] = _load_user_vectors(dataset)
    return _user_vectors[dataset]


def get_als(dataset: str):
    if dataset not in _als_packs:
        _als_packs[dataset] = _load_als(dataset)
    return _als_packs[dataset]


def get_hybrid_score_fn():
    global _hybrid_score_fn
    if _hybrid_score_fn is None:
        _hybrid_score_fn = _load_hybrid_ranker()
    return _hybrid_score_fn


def _user_domain(user_id: int) -> str:
    return "rr" if user_id >= RR_USER_OFFSET else "hm"


def _item_domain(item_id: int) -> str:
    return "hm" if item_id >= HM_MAX_ID else "rr"


def faiss_search(vec: np.ndarray, dataset: str, k: int = 20) -> list[dict]:
    """Content-based search via FAISS."""
    index, item_ids, _, _ = get_faiss(dataset)
    vec = vec.reshape(1, -1).astype("float32")
    scores, indices = index.search(vec, k)
    return [
        {"item_id": int(item_ids[i]), "score": float(s)}
        for s, i in zip(scores[0], indices[0])
    ]


def recommend_for_user(
    user_id: int,
    top_k: int = 20,
    strategy: str = "content",
) -> list[dict]:
    """
    Recommend items for a user.

    strategy: "content", "collab", or "hybrid". Domain (HM/RR) is inferred
    from user_id offset and artifact availability.
    """
    strategy = strategy.lower()
    dataset = _user_domain(user_id)

    if strategy == "collab":
        als = get_als(dataset)
        if als is not None:
            recs = _recommend_collab(user_id, top_k, als)
            if recs:
                return recs
        return _recommend_content(user_id, top_k, dataset)

    if strategy == "hybrid":
        return _recommend_hybrid(user_id, top_k, dataset)

    return _recommend_content(user_id, top_k, dataset)


def _recommend_content(user_id: int, top_k: int, dataset: str) -> list[dict]:
    """Content-based: user vector -> domain FAISS."""
    user_vectors = get_user_vectors(dataset)
    if user_id not in user_vectors:
        return []
    vec = user_vectors[user_id].astype("float32")
    return faiss_search(vec, dataset=dataset, k=top_k)


def _recommend_hybrid(user_id: int, top_k: int, dataset: str) -> list[dict]:
    """
    Hybrid: content candidates from the domain FAISS, then rerank using
    hybrid_ranker.score_hybrid when available.
    """
    content_candidates = _recommend_content(user_id, max(top_k, 100), dataset)
    if not content_candidates:
        return []

    score_fn = get_hybrid_score_fn()
    if score_fn is None:
        return content_candidates[:top_k]

    reranked = []
    for rec in content_candidates:
        item_id = rec["item_id"]
        h_score = score_fn(user_id, item_id, w_content=0.5, w_tt=0.5)
        final_score = rec["score"] if h_score is None else float(h_score)
        reranked.append({"item_id": item_id, "score": final_score})
    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked[:top_k]


def _recommend_collab(user_id: int, top_k: int, als: dict) -> list[dict]:
    """ALS-based: user factors -> dot with all item factors -> top-K."""
    user_id_to_idx = als["user_id_to_idx"]
    item_factors = als["item_factors"]
    user_factors = als["user_factors"]
    idx_to_item_id = als["idx_to_item_id"]

    if user_id not in user_id_to_idx:
        return []

    u_idx = user_id_to_idx[user_id]
    u_vec = user_factors[u_idx].astype("float32")
    scores = np.dot(item_factors, u_vec)
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {"item_id": int(idx_to_item_id[i]), "score": float(scores[i])}
        for i in top_indices
        if i in idx_to_item_id
    ]


def recommend_for_item(item_id: int, top_k: int = 20) -> list[dict]:
    """Item-to-item via the FAISS index for the item's domain."""
    dataset = _item_domain(item_id)
    _, _, item_X, row_map = get_faiss(dataset)
    if item_id not in row_map:
        return []
    idx = row_map[item_id]
    vec = item_X[idx].astype("float32")
    return faiss_search(vec, dataset=dataset, k=top_k)
