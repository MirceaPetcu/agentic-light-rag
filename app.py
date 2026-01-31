"""
FastAPI Backend for Multi-Agent LightRAG Framework.

This module provides endpoints for:
1. /ingest - Ingest documents using LightRAG's ingestion pipeline
2. /query - Query using the multi-agent architecture (proxies to pipeline service with SSE)
3. /query/simple - Simple query without multi-agent pipeline
"""

import os
import sys
from contextlib import asynccontextmanager
from functools import partial
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Add light_rag to the path for imports
sys.path.insert(0, './light_rag')

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.ollama import ollama_embed
from lightrag.utils import EmbeddingFunc

# Configuration from environment
WORKING_DIR = os.getenv("LIGHTRAG_WORKING_DIR", "./rag_storage")

# Pipeline Service Configuration
PIPELINE_SERVICE_URL = os.getenv("PIPELINE_SERVICE_URL", "http://localhost:10001")

# LightRAG Storage Configuration
LIGHTRAG_GRAPH_STORAGE = os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage")

# LLM Configuration
LLM_BINDING_HOST = os.getenv("LLM_BINDING_HOST", "https://api.deepseek.com")
LLM_BINDING_API_KEY = os.getenv("LLM_BINDING_API_KEY", os.getenv("OPENAI_API_KEY", ""))
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# Embedding Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")
EMBEDDING_HOST = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
MAX_EMBED_TOKENS = int(os.getenv("MAX_EMBED_TOKENS", "32768"))

# Multi-agent configuration (defaults for query requests)
MAX_STEPS = int(os.getenv("MAX_STEPS", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
TOP_K = int(os.getenv("TOP_K", "10"))

# Global instances
rag: LightRAG | None = None
http_client: httpx.AsyncClient | None = None


# ============================================================================
# LLM and Embedding Functions
# ============================================================================

async def llm_model_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list = [],
    keyword_extraction: bool = False,
    **kwargs
) -> str:
    """LLM function for LightRAG using vLLM (OpenAI-compatible API)."""
    return await openai_complete_if_cache(
        model=LLM_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=LLM_BINDING_HOST,
        api_key=LLM_BINDING_API_KEY,
        keyword_extraction=keyword_extraction,
        timeout=600,
        **kwargs,
    )


# ============================================================================
# Pydantic Models
# ============================================================================

class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    content: str = Field(..., description="The document content to ingest")
    doc_id: str | None = Field(None, description="Optional document ID")
    file_path: str | None = Field(None, description="Optional file path for citation")


class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    status: str
    message: str
    track_id: str | None = None
    doc_id: str | None = None


class QueryRequest(BaseModel):
    """Request model for querying."""
    query: str = Field(..., description="The user query")
    max_steps: int = Field(default=MAX_STEPS, description="Maximum number of iteration steps")
    similarity_threshold: float = Field(default=SIMILARITY_THRESHOLD, description="Convergence threshold")
    top_k: int = Field(default=TOP_K, description="Number of top results to retrieve")
    stream: bool = Field(default=True, description="Whether to stream SSE events")


class QueryResponse(BaseModel):
    """Response model for query results."""
    status: str
    response: str
    citations: list[dict[str, Any]] = []
    confidence: float = 0.0
    steps_taken: int = 1
    converged: bool = False
    metadata: dict[str, Any] = {}


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    rag_initialized: bool
    pipeline_service_healthy: bool = False


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - initialize and cleanup resources."""
    global rag, http_client

    # Startup
    print("Initializing LightRAG...")

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Initialize LightRAG
    # Neo4j configuration is automatically read from environment variables:
    # NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=MAX_EMBED_TOKENS,
            func=partial(
                ollama_embed,
                embed_model=EMBEDDING_MODEL,
                host=EMBEDDING_HOST,
            ),
        ),
        graph_storage=LIGHTRAG_GRAPH_STORAGE,
        use_guided_json_extraction=True
    )

    await rag.initialize_storages()
    print(f"LightRAG initialized with working directory: {WORKING_DIR}")
    print(f"Graph storage backend: {LIGHTRAG_GRAPH_STORAGE}")

    # Initialize HTTP client for pipeline service
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
    print(f"Pipeline service URL: {PIPELINE_SERVICE_URL}")

    yield

    # Shutdown
    print("Shutting down...")
    if rag:
        await rag.finalize_storages()
    if http_client:
        await http_client.aclose()
    print("Cleanup complete.")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Multi-Agent LightRAG API",
    description="FastAPI backend for document ingestion and multi-agent querying with LightRAG",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# Helper Functions
# ============================================================================

async def check_pipeline_service_health() -> bool:
    """Check if the pipeline service is healthy."""
    if not http_client:
        return False
    try:
        response = await http_client.get(f"{PIPELINE_SERVICE_URL}/health")
        return response.status_code == 200
    except Exception:
        return False


async def stream_pipeline_sse(query_request: QueryRequest):
    """
    Stream SSE events from the pipeline service.

    This function connects to the pipeline service and forwards
    all SSE events to the client.
    """
    if not http_client:
        yield "event: error\ndata: {\"error\": \"HTTP client not initialized\"}\n\n"
        return

    try:
        async with http_client.stream(
            "POST",
            f"{PIPELINE_SERVICE_URL}/query",
            json={
                "query": query_request.query,
                "max_steps": query_request.max_steps,
                "similarity_threshold": query_request.similarity_threshold,
                "top_k": query_request.top_k,
            },
            headers={"Accept": "text/event-stream"},
        ) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                yield f"event: error\ndata: {{\"error\": \"Pipeline service error: {response.status_code}\", \"detail\": \"{error_text.decode()}\"}}\n\n"
                return

            # Stream the SSE events
            async for line in response.aiter_lines():
                if line:
                    yield line + "\n"
                else:
                    yield "\n"

    except httpx.ConnectError:
        yield f"event: error\ndata: {{\"error\": \"Cannot connect to pipeline service at {PIPELINE_SERVICE_URL}\"}}\n\n"
    except Exception as e:
        yield f"event: error\ndata: {{\"error\": \"Stream error: {str(e)}\"}}\n\n"


async def call_pipeline_service(query_request: QueryRequest) -> dict[str, Any]:
    """
    Call the pipeline service and wait for the final result.

    This function consumes the SSE stream and returns only the final result.
    """
    if not http_client:
        raise HTTPException(status_code=503, detail="HTTP client not initialized")

    import json

    final_result = None

    try:
        async with http_client.stream(
            "POST",
            f"{PIPELINE_SERVICE_URL}/query",
            json={
                "query": query_request.query,
                "max_steps": query_request.max_steps,
                "similarity_threshold": query_request.similarity_threshold,
                "top_k": query_request.top_k,
            },
            headers={"Accept": "text/event-stream"},
        ) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Pipeline service error: {error_text.decode()}"
                )

            # Parse SSE events and extract the final result
            current_event = None
            current_data = []

            async for line in response.aiter_lines():
                if line.startswith("event:"):
                    current_event = line[6:].strip()
                elif line.startswith("data:"):
                    current_data.append(line[5:].strip())
                elif line == "" and current_event:
                    # End of event
                    if current_data:
                        data_str = "".join(current_data)
                        try:
                            data = json.loads(data_str)
                            if current_event == "completed":
                                final_result = data.get("result", {})
                            elif current_event == "error":
                                raise HTTPException(
                                    status_code=500,
                                    detail=data.get("error", "Unknown pipeline error")
                                )
                        except json.JSONDecodeError:
                            pass
                    current_event = None
                    current_data = []

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to pipeline service at {PIPELINE_SERVICE_URL}"
        )

    if final_result is None:
        raise HTTPException(status_code=500, detail="No result received from pipeline service")

    return final_result


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    pipeline_healthy = await check_pipeline_service_health()

    return HealthResponse(
        status="healthy",
        rag_initialized=rag is not None,
        pipeline_service_healthy=pipeline_healthy,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    """
    Ingest a document into the LightRAG knowledge graph.

    This endpoint uses LightRAG's ingestion pipeline to:
    1. Chunk the document
    2. Extract entities and relationships
    3. Build the knowledge graph
    4. Store embeddings for retrieval
    """
    if not rag:
        raise HTTPException(status_code=503, detail="LightRAG not initialized")

    try:
        # Prepare input parameters
        ids = [request.doc_id] if request.doc_id else None
        file_paths = [request.file_path] if request.file_path else None

        # Use LightRAG's async insert
        track_id = await rag.ainsert(
            input=request.content,
            ids=ids,
            file_paths=file_paths,
        )

        return IngestResponse(
            status="success",
            message="Document ingested successfully",
            track_id=track_id,
            doc_id=request.doc_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    doc_id: str | None = Form(None),
):
    """
    Ingest a file into the LightRAG knowledge graph.

    Supports text files (.txt, .md, .json, etc.)
    """
    if not rag:
        raise HTTPException(status_code=503, detail="LightRAG not initialized")

    try:

        # Prepare input parameters
        ids = [doc_id] if doc_id else None
        file_paths = [file.filename] if file.filename else None

        import tempfile
        import textract
        import os as os_module

        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os_module.path.splitext(file.filename)[1]) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        try:
            # Extract text content from the temporary file
            # Use errors='replace' to handle encoding issues gracefully
            text_bytes = textract.process(temp_file_path)
            text_content = text_bytes.decode("utf-8", errors="replace")

            # Clean up any replacement characters if needed
            if not text_content.strip():
                raise ValueError("No text content could be extracted from the file")

            # Use LightRAG's async insert
            track_id = await rag.ainsert(
                input=text_content,
                ids=ids,
                file_paths=file_paths,
            )

            return IngestResponse(
                status="success",
                message=f"File '{file.filename}' ingested successfully",
                track_id=track_id,
                doc_id=doc_id,
            )
        finally:
            # Clean up temporary file
            try:
                os_module.unlink(temp_file_path)
            except OSError:
                pass

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File ingestion failed: {str(e)}")


@app.post("/query")
async def query(request: QueryRequest):
    """
    Query the knowledge graph using the multi-agent architecture.

    This endpoint proxies to the pipeline service and can either:
    - Stream SSE events (stream=True, default) for real-time progress updates
    - Return the final result directly (stream=False)

    SSE Events (when streaming):
    - started: Pipeline started
    - progress: Progress update
    - decomposition: Query decomposed into subqueries
    - retrieval: Context retrieved from knowledge graph
    - deduction: Answer and query inferred
    - judgement: Convergence check result
    - iteration: New iteration started
    - completed: Final result with response and citations
    - error: Error occurred
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Check if pipeline service is available
    if not await check_pipeline_service_health():
        raise HTTPException(
            status_code=503,
            detail=f"Pipeline service not available at {PIPELINE_SERVICE_URL}"
        )

    if request.stream:
        # Return SSE stream
        return StreamingResponse(
            stream_pipeline_sse(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    else:
        # Wait for final result and return it
        result = await call_pipeline_service(request)
        return QueryResponse(**result)


@app.post("/query/simple")
async def query_simple(request: QueryRequest):
    """
    Simple query endpoint that uses LightRAG directly without the multi-agent pipeline.

    Useful for quick queries or debugging.
    """
    if not rag:
        raise HTTPException(status_code=503, detail="LightRAG not initialized")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        param = QueryParam(
            mode="mix",
            top_k=request.top_k,
        )

        response = await rag.aquery(request.query, param=param)

        return {
            "status": "success",
            "response": response,
            "query": request.query,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/status/{track_id}")
async def get_ingestion_status(track_id: str):
    """
    Get the status of a document ingestion job.
    """
    if not rag:
        raise HTTPException(status_code=503, detail="LightRAG not initialized")

    try:
        # Check pipeline status
        status = await rag.get_pipeline_status(track_id)
        return {
            "track_id": track_id,
            "status": status,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "10000"))

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=True,
    )
