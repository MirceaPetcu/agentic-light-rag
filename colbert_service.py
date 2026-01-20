"""
FastAPI service for ColBERT embeddings.

This service provides an OpenAI-compatible embeddings endpoint for ColBERT model.
It can be used by both the retriever (for reranking) and judge (for similarity computation).
"""

import time
from typing import List, Union

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer

app = FastAPI(
    title="ColBERT Embeddings Service",
    description="OpenAI-compatible embeddings API using ColBERT model",
    version="1.0.0",
)

# Global model and tokenizer
_model = None
_tokenizer = None
_device = None


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""
    model: str = Field(..., description="Model name (ignored, uses configured model)")
    input: Union[str, List[str]] = Field(..., description="Text or list of texts to embed")
    encoding_format: str = Field(default="float", description="Encoding format (only 'float' supported)")


class EmbeddingData(BaseModel):
    """Single embedding response data."""
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingUsage(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response."""
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


class SimilarityRequest(BaseModel):
    """Request for computing ColBERT MaxSim similarity between two texts."""
    query: str = Field(..., description="The query text")
    document: str = Field(..., description="The document text to compare against")


class SimilarityResponse(BaseModel):
    """Response containing similarity score."""
    similarity: float = Field(..., description="ColBERT MaxSim similarity score between 0 and 1")
    query: str
    document: str


def load_model(model_name: str = "colbert-ir/colbertv2.0", device: str = None):
    """Load ColBERT model and tokenizer."""
    global _model, _tokenizer, _device

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    _device = device
    print(f"Loading ColBERT model '{model_name}' on {device}...")

    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModel.from_pretrained(model_name)
    _model.to(device)
    _model.eval()

    print(f"ColBERT model loaded successfully on {device}")


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Get pooled embeddings for a list of texts.

    For compatibility with OpenAI's embeddings API, we return a single
    pooled embedding per text (mean of token embeddings).

    Args:
        texts: List of texts to embed

    Returns:
        List of embedding vectors
    """
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    embeddings = []

    for text in texts:
        tokens = _tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(_device)

        with torch.no_grad():
            output = _model(**tokens)
            # Get token embeddings
            token_embeddings = output.last_hidden_state  # [1, seq_len, hidden_dim]

            # Apply attention mask for proper mean pooling
            attention_mask = tokens["attention_mask"].unsqueeze(-1).float()
            masked_embeddings = token_embeddings * attention_mask

            # Mean pooling over valid tokens
            pooled = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1)

            # Normalize
            pooled = F.normalize(pooled, p=2, dim=-1)

            embeddings.append(pooled.squeeze(0).cpu().tolist())

    return embeddings


def compute_colbert_similarity(query: str, document: str) -> float:
    """
    Compute ColBERT MaxSim similarity between query and document.

    ColBERT uses late interaction where each token embedding from query
    is matched against all token embeddings from document, taking the maximum
    similarity for each query token, then averaging.

    Args:
        query: The query text
        document: The document text

    Returns:
        Similarity score between 0 and 1
    """
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    # Tokenize both texts
    query_tokens = _tokenizer(
        query,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(_device)

    doc_tokens = _tokenizer(
        document,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(_device)

    with torch.no_grad():
        # Get token embeddings
        query_embeddings = _model(**query_tokens).last_hidden_state  # [1, seq_len_q, hidden_dim]
        doc_embeddings = _model(**doc_tokens).last_hidden_state  # [1, seq_len_d, hidden_dim]

        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        doc_embeddings = F.normalize(doc_embeddings, p=2, dim=-1)

        # Compute similarity matrix: [1, seq_len_q, seq_len_d]
        similarity_matrix = torch.matmul(query_embeddings, doc_embeddings.transpose(-1, -2))

        # Get attention masks
        query_mask = query_tokens["attention_mask"].unsqueeze(-1).float()  # [1, seq_len_q, 1]
        doc_mask = doc_tokens["attention_mask"].unsqueeze(1).float()  # [1, 1, seq_len_d]

        # Apply document mask: set padding positions to very negative value
        masked_similarity = similarity_matrix * doc_mask + (1 - doc_mask) * (-1e9)

        # MaxSim: max over document tokens for each query token
        max_sim_per_token = masked_similarity.max(dim=-1).values  # [1, seq_len_q]

        # Apply query mask and compute mean over valid tokens
        max_sim_per_token = max_sim_per_token * query_mask.squeeze(-1)
        num_valid_tokens = query_mask.squeeze(-1).sum()

        similarity_score = max_sim_per_token.sum() / num_valid_tokens

    return float(similarity_score.cpu())


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    import os
    model_name = os.environ.get("COLBERT_MODEL", "colbert-ir/colbertv2.0")
    device = os.environ.get("COLBERT_DEVICE", None)
    load_model(model_name, device)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": _model is not None}


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    Create embeddings for the input text(s).

    This endpoint is compatible with OpenAI's embeddings API.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Normalize input to list
    if isinstance(request.input, str):
        texts = [request.input]
    else:
        texts = request.input

    if not texts:
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    try:
        embeddings = get_embeddings(texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

    # Count tokens (approximate)
    total_tokens = sum(len(_tokenizer.encode(t)) for t in texts)

    return EmbeddingResponse(
        object="list",
        data=[
            EmbeddingData(
                object="embedding",
                embedding=emb,
                index=i
            )
            for i, emb in enumerate(embeddings)
        ],
        model=request.model,
        usage=EmbeddingUsage(
            prompt_tokens=total_tokens,
            total_tokens=total_tokens
        )
    )


@app.post("/v1/similarity", response_model=SimilarityResponse)
async def compute_similarity(request: SimilarityRequest):
    """
    Compute ColBERT MaxSim similarity between query and document.

    This is the proper ColBERT late-interaction similarity, which provides
    better semantic matching than simple cosine similarity of pooled embeddings.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        similarity = compute_colbert_similarity(request.query, request.document)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity computation failed: {str(e)}")

    return SimilarityResponse(
        similarity=similarity,
        query=request.query,
        document=request.document
    )


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="ColBERT Embeddings Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind to")
    parser.add_argument("--model", default="colbert-ir/colbertv2.0", help="ColBERT model name")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu, auto-detected if not specified)")

    args = parser.parse_args()

    # Set environment variables for startup
    import os
    os.environ["COLBERT_MODEL"] = args.model
    if args.device:
        os.environ["COLBERT_DEVICE"] = args.device

    uvicorn.run(app, host=args.host, port=args.port)
