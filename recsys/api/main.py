from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import joblib
from PIL import Image
import io
from pathlib import Path
from typing import Union

# Paths
BASE = Path("recsys")
DATA = BASE / "data"
ART = BASE / "artifacts"

app = FastAPI()

# Allow local dev + frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REQUIRED_ARTIFACTS = {
    "faiss_pack": ART / "faiss_items_hm.joblib",
    "two_tower_model": ART / "content_two_tower_hm.pt",
}

faiss_pack = None
faiss_index = None
item_ids = None
item_X = None
row_map = None

# Optional: Hybrid ranker (Mahout + Two-Tower). Falls back to content if missing.
score_hybrid = None


def get_hybrid_score():
    global score_hybrid
    if score_hybrid is None:
        try:
            from recsys.src.models.hybrid_ranker import score_hybrid as hybrid_score_fn

            score_hybrid = hybrid_score_fn
            print("[api] Hybrid ranker loaded.")
        except Exception as e:
            print(f"[api] Hybrid ranker not available ({e}). Using content/collab only.")
            score_hybrid = False
    return score_hybrid if score_hybrid is not False else None

############################################
# Load CLIP for image embedding
############################################

device = None
torch = None
CLIPModel = None
CLIPProcessor = None
clip_model = None
clip_processor = None


def get_torch_module():
    global torch
    if torch is None:
        import torch as torch_module

        torch = torch_module
    return torch


def get_clip_components():
    global clip_model, clip_processor, device, CLIPModel, CLIPProcessor
    if clip_model is None or clip_processor is None:
        torch_module = get_torch_module()
        if CLIPModel is None or CLIPProcessor is None:
            from transformers import CLIPProcessor as clip_processor_cls, CLIPModel as clip_model_cls

            CLIPModel = clip_model_cls
            CLIPProcessor = clip_processor_cls

        if device is None:
            device = "cuda" if torch_module.cuda.is_available() else "cpu"

        print(f"[api] Loading CLIP on {device} ...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()
    return clip_model, clip_processor


def artifact_status() -> dict[str, bool]:
    return {name: path.exists() for name, path in REQUIRED_ARTIFACTS.items()}


def ensure_faiss_assets_loaded():
    global faiss_pack, faiss_index, item_ids, item_X, row_map

    status = artifact_status()
    missing = [name for name in ("faiss_pack",) if not status[name]]
    if missing:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Recommender artifacts are not ready.",
                "missing_artifacts": missing,
                "artifact_status": status,
            },
        )

    if faiss_pack is None:
        print("[api] Loading H&M FAISS index...")
        faiss_pack = joblib.load(REQUIRED_ARTIFACTS["faiss_pack"])
        faiss_index = faiss_pack["index"]
        item_ids = faiss_pack["item_ids"]
        item_X = faiss_pack["X"]
        row_map = faiss_pack["row_map"]


############################################
# Utility: embed an uploaded image
############################################
def embed_image_file(file: UploadFile) -> np.ndarray:
    model, processor = get_clip_components()
    torch_module = get_torch_module()
    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch_module.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten().astype("float32")


############################################
# Utility: run FAISS search (Retrieval)
############################################
def faiss_search(vec: np.ndarray, k: int = 20):
    ensure_faiss_assets_loaded()
    vec = vec.reshape(1, -1).astype("float32")
    scores, indices = faiss_index.search(vec, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        item_id = int(item_ids[idx])
        results.append({"item_id": item_id, "score": float(score)})
    return results


############################################
# ROUTES
############################################

@app.get("/health")
def health():
    status = artifact_status()
    return {
        "status": "ok",
        "catalog_ready": status["faiss_pack"] and status["two_tower_model"],
        "recommender_ready": status["faiss_pack"] and status["two_tower_model"],
        "artifact_status": status,
    }


############################################
# 1. Embed image
############################################
@app.post("/embed/image")
async def embed_image(file: UploadFile = File(...)):
    vec = embed_image_file(file)
    return {"embedding": vec.tolist()}


############################################
# 2. Recommend from image(s) (Cold Start)
############################################
class RecommendFromImagesRequest(BaseModel):
    images: list[str] = []  # base64 if frontend wants this (optional)


@app.post("/recommend/from_image")
async def recommend_from_image(file: UploadFile = File(...), top_k: int = 20):
    vec = embed_image_file(file)
    results = faiss_search(vec, k=top_k)
    return {"recommendations": results}


############################################
# 3. Recommend from Supabase catalog (new users + new products)
############################################
class ProductInput(BaseModel):
    id: Union[int, str, None] = None
    title: str = ""
    name: str = ""
    description: str = ""


class HybridCatalogRequest(BaseModel):
    liked_products: list[ProductInput] = Field(default_factory=list)
    catalog_products: list[ProductInput] = Field(default_factory=list)
    top_k: int = 20
    strategy: str = "hybrid"  # content | two_tower | hybrid
    exclude_ids: list[Union[int, str]] = Field(default_factory=list)


@app.post("/recommend/hybrid/catalog")
def recommend_hybrid_catalog(req: HybridCatalogRequest):
    """
    Generalizable recommendation route for disjoint catalogs (e.g. Supabase).
    Uses content/two-tower/hybrid from vectors, no ALS IDs required.
    """
    from recsys.src.production_recommender import recommend as recommend_from_catalog

    liked_products = [
        {
            "title": p.title or p.name or "",
            "description": p.description or "",
        }
        for p in req.liked_products
    ]

    catalog_products = []
    catalog_ids = []
    for p in req.catalog_products:
        if p.id is None:
            continue
        catalog_products.append(
            {
                "title": p.title or p.name or "",
                "description": p.description or "",
            }
        )
        catalog_ids.append(p.id)

    recommendations = recommend_from_catalog(
        liked_products=liked_products,
        catalog_products=catalog_products,
        catalog_ids=catalog_ids,
        top_k=req.top_k,
        strategy=req.strategy,
        exclude_ids=req.exclude_ids,
    )
    return {"recommendations": recommendations}


############################################
# 4. Recommend for user_id
############################################
@app.get("/recommend/user/{user_id}")
def recommend_user(user_id: int, top_k: int = 20, strategy: str = "content"):
    raise HTTPException(
        status_code=503,
        detail={
            "message": "The user_id recommendation endpoint is disabled in this deployment.",
            "reason": "This Render deployment only ships the catalog recommender artifacts.",
        },
    )


############################################
# 5. Item-to-item similarity
############################################
@app.get("/recommend/item/{item_id}")
def recommend_item(item_id: int, top_k: int = 20):
    ensure_faiss_assets_loaded()
    if item_id not in row_map:
        return {"recommendations": []}

    idx = row_map[item_id]
    vec = item_X[idx].astype("float32")
    results = faiss_search(vec, k=top_k)
    return {"recommendations": results}
