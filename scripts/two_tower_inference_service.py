#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "PyTorch is required for two_tower_inference_service.py. Install torch in your Python env."
    ) from exc


# Load local environment variables when running as a standalone service.
load_dotenv(".env.local")
load_dotenv()

MODEL_PATH = os.getenv("TWO_TOWER_MODEL_PATH", "models/content_two_tower_hm.pt")
MODEL_DEVICE = os.getenv("TWO_TOWER_DEVICE", "cpu")


class ReconstructedTwoTower(nn.Module):
    """Best-effort reconstruction for saved state_dict checkpoints."""

    def __init__(
        self,
        input_dim: int,
        tower_hidden_dim: int,
        embedding_dim: int,
        scorer_hidden_dim: int,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(input_dim, tower_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(tower_hidden_dim, embedding_dim),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(input_dim, tower_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(tower_hidden_dim, embedding_dim),
        )
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim, scorer_hidden_dim),
            nn.ReLU(),
            nn.Linear(scorer_hidden_dim, 1),
        )

    def encode_user(self, user_features: torch.Tensor) -> torch.Tensor:
        return self.user_tower(user_features)

    def encode_item(self, item_features: torch.Tensor) -> torch.Tensor:
        return self.item_tower(item_features)

    def score(self, user_features: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        user_emb = self.encode_user(user_features)
        item_emb = self.encode_item(item_features)
        interaction = user_emb * item_emb
        return self.scorer(interaction).squeeze(-1)

    def forward(self, user_features: torch.Tensor, item_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        if item_features is None:
            return self.encode_user(user_features)
        return self.score(user_features, item_features)


def _to_float_list(values: List[float]) -> List[float]:
    return [float(v) for v in values]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dim = min(len(a), len(b))
    if dim == 0:
        return 0.0

    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(dim):
        av = float(a[i])
        bv = float(b[i])
        dot += av * bv
        na += av * av
        nb += bv * bv

    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(dot / ((na ** 0.5) * (nb ** 0.5)))


class CandidateEmbedding(BaseModel):
    id: str
    embedding: List[float]


class RankRequest(BaseModel):
    user_embedding: List[float]
    candidates: List[CandidateEmbedding]
    top_k: Optional[int] = Field(default=None, ge=1)


class RankItem(BaseModel):
    id: str
    score: float


class RankResponse(BaseModel):
    results: List[RankItem]
    used_model: bool
    model_loaded: bool
    fallback_reason: Optional[str] = None


class EmbedUserRequest(BaseModel):
    liked_embeddings: List[List[float]]


class EmbedUserResponse(BaseModel):
    embedding: List[float]
    used_model: bool
    model_loaded: bool
    fallback_reason: Optional[str] = None


class EmbedItemsRequest(BaseModel):
    items: List[CandidateEmbedding]


class EmbeddedItem(BaseModel):
    id: str
    embedding: List[float]
    used_model: bool


class EmbedItemsResponse(BaseModel):
    items: List[EmbeddedItem]
    model_loaded: bool
    used_model: bool
    fallback_reason: Optional[str] = None


class TwoTowerRuntime:
    def __init__(self, model_path: str, device_name: str) -> None:
        self.model_path = model_path
        self.device_name = device_name
        self.device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
        self.model: Optional[Any] = None
        self.load_error: Optional[str] = None
        self.model_kind: str = "unloaded"
        self._load_model()

    @property
    def available(self) -> bool:
        return self.model is not None

    def _load_model(self) -> None:
        if not os.path.exists(self.model_path):
            self.load_error = f"Model file not found at {self.model_path}"
            return

        try:
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model_kind = "torchscript"
        except Exception:
            try:
                loaded = torch.load(self.model_path, map_location=self.device)
            except Exception as exc:
                self.load_error = str(exc)
                return

            if isinstance(loaded, dict) and all(isinstance(k, str) for k in loaded.keys()):
                reconstructed = self._try_reconstruct_from_state_dict(loaded)
                if reconstructed is not None:
                    self.model = reconstructed
                    self.model_kind = "reconstructed_from_state_dict"
                else:
                    self.model = None
                    self.model_kind = "state_dict_only"
                    self.load_error = (
                        "Checkpoint appears to be state_dict-only and could not be reconstructed. "
                        "Provide model architecture code or export a TorchScript model."
                    )
                    return
            else:
                self.model = loaded
                self.model_kind = "torch_load"

        try:
            if hasattr(self.model, "to"):
                self.model = self.model.to(self.device)
            if hasattr(self.model, "eval"):
                self.model.eval()
        except Exception as exc:
            self.model = None
            self.load_error = f"Model initialized but failed to prepare for inference: {exc}"

    def _try_reconstruct_from_state_dict(self, state_dict: Dict[str, Any]) -> Optional[nn.Module]:
        required = [
            "user_tower.0.weight",
            "user_tower.3.weight",
            "item_tower.0.weight",
            "item_tower.3.weight",
            "scorer.0.weight",
            "scorer.2.weight",
        ]
        if not all(k in state_dict for k in required):
            return None

        try:
            input_dim = int(state_dict["user_tower.0.weight"].shape[1])
            tower_hidden_dim = int(state_dict["user_tower.0.weight"].shape[0])
            embedding_dim = int(state_dict["user_tower.3.weight"].shape[0])
            scorer_hidden_dim = int(state_dict["scorer.0.weight"].shape[0])
        except Exception:
            return None

        model = ReconstructedTwoTower(
            input_dim=input_dim,
            tower_hidden_dim=tower_hidden_dim,
            embedding_dim=embedding_dim,
            scorer_hidden_dim=scorer_hidden_dim,
            dropout_p=0.1,
        )

        try:
            model.load_state_dict(state_dict, strict=True)
            return model
        except Exception:
            return None

    def _tensor(self, values: List[float]) -> Any:
        return torch.tensor(values, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _extract_vector(self, output: Any) -> List[float]:
        if isinstance(output, dict):
            for key in ("embedding", "output", "score", "logit", "prediction"):
                if key in output:
                    output = output[key]
                    break
            else:
                output = next(iter(output.values()))
        if isinstance(output, (list, tuple)):
            output = output[0]
        if not torch.is_tensor(output):
            output = torch.tensor(output, dtype=torch.float32, device=self.device)
        output = output.detach().float().view(-1)
        return output.cpu().tolist()

    def _call_one_arg(self, method_names: List[str], arg: Any) -> Optional[List[float]]:
        if not self.model:
            return None
        for method_name in method_names:
            method = getattr(self.model, method_name, None)
            if not callable(method):
                continue
            try:
                with torch.no_grad():
                    return self._extract_vector(method(arg))
            except Exception:
                continue
        return None

    def encode_user(self, liked_embeddings: List[List[float]]) -> Optional[List[float]]:
        if not self.model or not liked_embeddings:
            return None

        matrix = torch.tensor(liked_embeddings, dtype=torch.float32, device=self.device)
        pooled = matrix.mean(dim=0, keepdim=True)

        encoded = self._call_one_arg(
            ["encode_user", "user_tower", "user_encoder", "forward_user"],
            pooled,
        )
        if encoded:
            return encoded

        # Last-resort attempt: call model(pooled) if the model only has a single tower exported.
        try:
            with torch.no_grad():
                return self._extract_vector(self.model(pooled))
        except Exception:
            return None

    def encode_item(self, item_embedding: List[float]) -> Optional[List[float]]:
        if not self.model or not item_embedding:
            return None

        item_tensor = self._tensor(item_embedding)
        encoded = self._call_one_arg(
            ["encode_item", "item_tower", "item_encoder", "forward_item"],
            item_tensor,
        )
        if encoded:
            return encoded

        # Last-resort attempt: single-tower export.
        try:
            with torch.no_grad():
                return self._extract_vector(self.model(item_tensor))
        except Exception:
            return None

    def score(self, user_embedding: List[float], item_embedding: List[float]) -> Optional[float]:
        if not self.model:
            return None

        user_tensor = self._tensor(user_embedding)
        item_tensor = self._tensor(item_embedding)

        for method_name in ["score", "predict_score", "predict"]:
            method = getattr(self.model, method_name, None)
            if not callable(method):
                continue
            try:
                with torch.no_grad():
                    out = self._extract_vector(method(user_tensor, item_tensor))
                if out:
                    return float(out[0])
            except Exception:
                continue

        # Common two-tower shape: forward(user, item)
        try:
            with torch.no_grad():
                out = self._extract_vector(self.model(user_tensor, item_tensor))
            if out:
                return float(out[0])
        except Exception:
            pass

        # Fallback attempt for models expecting concatenated features.
        try:
            concat = torch.cat([user_tensor, item_tensor], dim=1)
            with torch.no_grad():
                out = self._extract_vector(self.model(concat))
            if out:
                return float(out[0])
        except Exception:
            return None

        return None


runtime = TwoTowerRuntime(MODEL_PATH, MODEL_DEVICE)
app = FastAPI(title="Two Tower Inference Service", version="1.0.0")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "model_path": runtime.model_path,
        "model_loaded": runtime.available,
        "model_kind": runtime.model_kind,
        "device": str(runtime.device),
        "load_error": runtime.load_error,
    }


@app.post("/rank", response_model=RankResponse)
def rank(request: RankRequest) -> RankResponse:
    candidates = request.candidates or []
    if not candidates:
        return RankResponse(results=[], used_model=False, model_loaded=runtime.available)

    user_embedding = _to_float_list(request.user_embedding)
    fallback_reason: Optional[str] = None
    scored: List[RankItem] = []
    used_model = False

    # Attempt full model scoring first; if it fails, fall back to cosine for all candidates.
    if runtime.available:
        try:
            temp_scores: List[RankItem] = []
            for candidate in candidates:
                item_embedding = _to_float_list(candidate.embedding)
                score = runtime.score(user_embedding, item_embedding)
                if score is None:
                    raise RuntimeError("Model could not produce score for one or more candidates.")
                temp_scores.append(RankItem(id=candidate.id, score=float(score)))
            scored = temp_scores
            used_model = True
        except Exception as exc:
            fallback_reason = str(exc)

    if not scored:
        if not fallback_reason and not runtime.available:
            fallback_reason = runtime.load_error or "Model unavailable"
        for candidate in candidates:
            item_embedding = _to_float_list(candidate.embedding)
            score = _cosine_similarity(user_embedding, item_embedding)
            scored.append(RankItem(id=candidate.id, score=float(score)))

    scored.sort(key=lambda row: row.score, reverse=True)
    if request.top_k:
        scored = scored[: request.top_k]

    return RankResponse(
        results=scored,
        used_model=used_model,
        model_loaded=runtime.available,
        fallback_reason=fallback_reason,
    )


@app.post("/embed-user", response_model=EmbedUserResponse)
def embed_user(request: EmbedUserRequest) -> EmbedUserResponse:
    liked = [list(map(float, vec)) for vec in request.liked_embeddings if vec]
    if not liked:
        return EmbedUserResponse(
            embedding=[],
            used_model=False,
            model_loaded=runtime.available,
            fallback_reason="No liked embeddings supplied",
        )

    if runtime.available:
        encoded = runtime.encode_user(liked)
        if encoded and len(encoded) > 0:
            return EmbedUserResponse(
                embedding=[float(v) for v in encoded],
                used_model=True,
                model_loaded=True,
            )

    # Fallback: average liked item vectors.
    dim = len(liked[0])
    sums = [0.0] * dim
    for vec in liked:
        for i in range(min(dim, len(vec))):
            sums[i] += vec[i]
    averaged = [s / len(liked) for s in sums]

    return EmbedUserResponse(
        embedding=averaged,
        used_model=False,
        model_loaded=runtime.available,
        fallback_reason=runtime.load_error or "User tower unavailable, used mean embedding fallback",
    )


@app.post("/embed-items", response_model=EmbedItemsResponse)
def embed_items(request: EmbedItemsRequest) -> EmbedItemsResponse:
    out: List[EmbeddedItem] = []
    used_model = False

    for item in request.items:
        vec = _to_float_list(item.embedding)
        encoded = runtime.encode_item(vec) if runtime.available else None
        if encoded and len(encoded) > 0:
            out.append(EmbeddedItem(id=item.id, embedding=[float(v) for v in encoded], used_model=True))
            used_model = True
        else:
            out.append(EmbeddedItem(id=item.id, embedding=vec, used_model=False))

    return EmbedItemsResponse(
        items=out,
        model_loaded=runtime.available,
        used_model=used_model,
        fallback_reason=None if used_model else (runtime.load_error or "Item tower unavailable"),
    )
