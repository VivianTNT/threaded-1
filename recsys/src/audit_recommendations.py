"""
Audit what recommendation components are actually used in serving.

Checks:
1) Wiring in API/engine code (HM vs RR artifacts).
2) Runtime behavior for HM user vs RR user via recommend_engine.
3) RR FAISS + RR user-vector logic viability (offline simulation).
4) Two-tower model smoke test.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

from recsys.src.recommend_engine import recommend_for_user


ROOT = Path("recsys")
DATA = ROOT / "data"
ART = ROOT / "artifacts"
API_MAIN = ROOT / "api" / "main.py"
RECOMMEND_ENGINE = ROOT / "src" / "recommend_engine.py"


class ContentTwoTower(nn.Module):
    def __init__(self, feature_dim: int = 384, embed_dim: int = 128):
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


def _load_item_meta() -> dict[int, tuple[str, str]]:
    items = pd.read_parquet(DATA / "items.parquet", columns=["item_id", "title", "source"])
    return {int(r.item_id): (str(r.title), str(r.source)) for r in items.itertuples(index=False)}


def _code_wiring_checks() -> None:
    api_txt = API_MAIN.read_text()
    eng_txt = RECOMMEND_ENGINE.read_text()

    print("=== Static Wiring ===")
    print("api/main.py loads faiss_items_hm.joblib:", "faiss_items_hm.joblib" in api_txt)
    print("api/main.py loads user_vectors_hm.joblib:", "user_vectors_hm.joblib" in api_txt)
    print("api/main.py mentions user_vectors_rr.joblib:", "user_vectors_rr.joblib" in api_txt)
    print("recommend_engine uses faiss_items_hm.joblib:", "faiss_items_hm.joblib" in eng_txt)
    print("recommend_engine uses user_vectors_hm.joblib:", "user_vectors_hm.joblib" in eng_txt)
    print("recommend_engine mentions user_vectors_rr.joblib:", "user_vectors_rr.joblib" in eng_txt)


def _pick_users() -> tuple[int, int]:
    hm_uid = int(pd.read_parquet(DATA / "events_hm.parquet", columns=["user_id"]).iloc[0, 0])
    rr_uid = int(pd.read_parquet(DATA / "events_rr.parquet", columns=["user_id"]).iloc[0, 0])
    return hm_uid, rr_uid


def _print_recs(label: str, uid: int, recs: list[dict], item_meta: dict[int, tuple[str, str]]) -> None:
    print(f"\n[{label}] user_id={uid}, recommendations={len(recs)}")
    for r in recs:
        iid = int(r["item_id"])
        title, source = item_meta.get(iid, ("<missing>", "?"))
        print(f"  item={iid} score={r['score']:.4f} source={source} title={title[:100]}")


def _runtime_live_checks(top_k: int = 5) -> tuple[int, int]:
    print("\n=== Runtime: recommend_engine (live path) ===")
    item_meta = _load_item_meta()
    hm_uid, rr_uid = _pick_users()

    t0 = perf_counter()
    hm_recs = recommend_for_user(hm_uid, top_k=top_k, strategy="content")
    print(f"HM content call took {perf_counter() - t0:.2f}s")
    _print_recs("HM content", hm_uid, hm_recs, item_meta)

    t1 = perf_counter()
    rr_recs = recommend_for_user(rr_uid, top_k=top_k, strategy="content")
    print(f"RR content call took {perf_counter() - t1:.2f}s")
    _print_recs("RR content", rr_uid, rr_recs, item_meta)

    return hm_uid, rr_uid


def _rr_offline_simulation(top_k: int = 5) -> None:
    print("\n=== Offline RR Simulation (not used by live API) ===")
    events_rr = pd.read_parquet(DATA / "events_rr.parquet", columns=["user_id", "item_id", "timestamp"])
    rr_uid = int(events_rr["user_id"].iloc[0])

    faiss_pack = joblib.load(ART / "faiss_items_rr.joblib")
    row_map = faiss_pack["row_map"]
    x = np.asarray(faiss_pack["X"], dtype="float32")
    index = faiss_pack["index"]
    item_ids = np.asarray(faiss_pack["item_ids"])

    ev = events_rr[events_rr["user_id"] == rr_uid].copy()
    ev = ev[ev["item_id"].isin(row_map.keys())]
    if ev.empty:
        print("No RR events found for sampled RR user.")
        return

    tmin, tmax = events_rr["timestamp"].min(), events_rr["timestamp"].max()
    ev["weight"] = np.exp((ev["timestamp"] - tmin) / (tmax - tmin + 1e-8) * 5)
    rows = ev["item_id"].map(row_map).to_numpy()
    weights = ev["weight"].to_numpy(dtype="float32")
    user_vec = (x[rows] * weights[:, None]).sum(axis=0)
    user_vec = user_vec / (np.linalg.norm(user_vec) + 1e-8)
    user_vec = user_vec.astype("float32")

    scores, idx = index.search(user_vec.reshape(1, -1), top_k)
    item_meta = _load_item_meta()

    print(f"RR sampled user={rr_uid}, recs={top_k}")
    for sc, i in zip(scores[0], idx[0]):
        iid = int(item_ids[i])
        title, source = item_meta.get(iid, ("<missing>", "?"))
        print(f"  item={iid} score={float(sc):.4f} source={source} title={title[:100]}")


def _two_tower_smoke() -> None:
    print("\n=== Two-Tower Smoke Test ===")
    model_path = ART / "content_two_tower_hm.pt"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ContentTwoTower().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    rng = np.random.default_rng(7)
    u = rng.normal(size=384).astype("float32")
    i = rng.normal(size=384).astype("float32")
    u /= np.linalg.norm(u) + 1e-8
    i /= np.linalg.norm(i) + 1e-8

    with torch.no_grad():
        ut = torch.tensor(u, dtype=torch.float32, device=device).unsqueeze(0)
        it = torch.tensor(i, dtype=torch.float32, device=device).unsqueeze(0)
        s_tt = float(model(ut, it).cpu().numpy()[0])

    s_content = float(np.dot(u, i))
    s_h = 0.5 * s_content + 0.5 * (2 * (s_tt - 0.5))
    print(f"score_two_tower(random_vecs)={s_tt:.4f}")
    print(f"score_hybrid(random_vecs)={s_h:.4f}")


def main(top_k: int = 5) -> None:
    _code_wiring_checks()
    _runtime_live_checks(top_k=top_k)
    _rr_offline_simulation(top_k=top_k)
    _two_tower_smoke()


if __name__ == "__main__":
    main(top_k=5)
