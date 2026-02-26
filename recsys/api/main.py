from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import joblib
from PIL import Image
import io
from pathlib import Path
import torch
from transformers import CLIPProcessor, CLIPModel
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

############################################
# Load embeddings + FAISS (for retrieval)
############################################

print("[api] Loading H&M FAISS index...")
faiss_pack = joblib.load(ART / "faiss_items_hm.joblib")
faiss_index = faiss_pack["index"]
item_ids = faiss_pack["item_ids"]

############################################
# Load CLIP for image embedding
############################################

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[api] Loading CLIP on {device} ...")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()


############################################
# Utility: embed an uploaded image
############################################
def embed_image_file(file: UploadFile) -> np.ndarray:
    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten().astype("float32")


############################################
# Utility: run FAISS search (Retrieval)
############################################
def faiss_search(vec: np.ndarray, k: int = 20):
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
    return {"status": "ok"}


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
    """
    Recommend items for a user.
    strategy: "content" (FAISS), "collab" (ALS), or "hybrid" (content+TwoTower).
    Default: content. If collab requested but ALS not available, falls back to content.
    """
    from recsys.src.recommend_engine import recommend_for_user
    results = recommend_for_user(user_id, top_k=top_k, strategy=strategy)
    return {"recommendations": results}


############################################
# 5. Item-to-item similarity
############################################
@app.get("/recommend/item/{item_id}")
def recommend_item(item_id: int, top_k: int = 20):
    from recsys.src.recommend_engine import recommend_for_item
    results = recommend_for_item(item_id, top_k=top_k)
    return {"recommendations": results}
