"""
FastAPI Backend for Multi-Agent LightRAG Framework.

This module provides endpoints for:
1. /ingest - Ingest documents using LightRAG's ingestion pipeline
2. /query - Query using the multi-agent architecture (proxies to pipeline service with SSE)
3. /query/simple - Simple query without multi-agent pipeline
"""

import os
import sys
import traceback
import logging
import uuid
from urllib.parse import urlparse
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from psycopg2.pool import SimpleConnectionPool

# Add light_rag to the path for imports
sys.path.insert(0, './light_rag')

from lightrag import QueryParam

# Configuration from environment
# Pipeline Service Configuration
PIPELINE_HOST = os.getenv("PIPELINE_HOST", "localhost")
PIPELINE_PORT = os.getenv("PIPELINE_PORT", "8003")

# LightRAG FastAPI Service Configuration
LIGHTRAG_HOST = os.getenv("LIGHTRAG_HOST", "localhost")
LIGHTRAG_PORT = os.getenv("LIGHTRAG_PORT", "8005")

# Multi-agent configuration (defaults for query requests)
MAX_STEPS = int(os.getenv("MAX_STEPS", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
TOP_K = int(os.getenv("TOP_K", "10"))


def build_service_url(host: str, port: str | int) -> str:
    """Build a service URL without double-schemes or duplicate ports."""
    host = host.strip().rstrip("/")
    port_str = str(port)
    if host.startswith("http://") or host.startswith("https://"):
        parsed = urlparse(host)
        if parsed.netloc and ":" in parsed.netloc:
            return host
        return f"{host}:{port_str}"
    return f"http://{host}:{port_str}"


PIPELINE_SERVICE_URL = build_service_url(PIPELINE_HOST, PIPELINE_PORT)
LIGHTRAG_SERVICE_URL = build_service_url(LIGHTRAG_HOST, LIGHTRAG_PORT)

# Postgres configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "postgres")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_SSLMODE = os.getenv("POSTGRES_SSLMODE", "prefer")
POSTGRES_POOL_MIN = int(os.getenv("POSTGRES_POOL_MIN", "1"))
POSTGRES_POOL_MAX = int(os.getenv("POSTGRES_POOL_MAX", "5"))

# Global instances
http_client: httpx.AsyncClient | None = None
db_pool: SimpleConnectionPool | None = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)




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


class DocumentStatusRecord(BaseModel):
    """Document status as stored in Postgres."""

    document_id: str
    document_status: str
    file_path: str | None = None
    updated_at: datetime


class DocumentStatusRequest(BaseModel):
    """Request model for fetching document statuses."""

    document_ids: list[str]


class DocumentStatusResponse(BaseModel):
    """Response model containing document statuses."""

    documents: list[DocumentStatusRecord]


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
    global http_client, db_pool

    # Startup
    print("Initializing app...")

    # Initialize HTTP clients for communicating with LightRAG and Pipeline services
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
    print(f"LightRAG service URL: {LIGHTRAG_SERVICE_URL}")
    print(f"Pipeline service URL: {PIPELINE_SERVICE_URL}")

    # Initialize Postgres connection pool for document status tracking
    db_pool = init_db_pool()

    yield

    # Shutdown
    print("Shutting down...")
    if http_client:
        await http_client.aclose()
    if db_pool:
        db_pool.closeall()
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
# Exception Handlers
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler that logs full tracebacks."""
    logger.error(
        f"Unhandled exception: {type(exc).__name__}: {str(exc)}\n"
        f"Full traceback:\n{traceback.format_exc()}",
        exc_info=True
    )
    
    # Print to console as well for immediate visibility
    print(f"\n{'='*80}")
    print(f"ERROR: {type(exc).__name__}: {str(exc)}")
    print(f"{'='*80}")
    traceback.print_exc()
    print(f"{'='*80}\n")
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "type": type(exc).__name__,
            "traceback": traceback.format_exc() if os.getenv("DEBUG", "false").lower() == "true" else None
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler for HTTPExceptions that logs tracebacks."""
    logger.error(
        f"HTTPException: {exc.status_code} - {exc.detail}\n"
        f"Request: {request.method} {request.url}\n"
        f"Full traceback:\n{traceback.format_exc()}",
        exc_info=True
    )
    
    # Print to console as well
    print(f"\n{'='*80}")
    print(f"HTTPException: {exc.status_code} - {exc.detail}")
    print(f"Request: {request.method} {request.url}")
    print(f"{'='*80}")
    traceback.print_exc()
    print(f"{'='*80}\n")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handler for validation errors."""
    logger.error(
        f"Validation error: {exc.errors()}\n"
        f"Request: {request.method} {request.url}",
        exc_info=True
    )
    
    print(f"\n{'='*80}")
    print(f"Validation Error: {exc.errors()}")
    print(f"Request: {request.method} {request.url}")
    print(f"{'='*80}\n")
    
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body}
    )


# ============================================================================
# Helper Functions
# ============================================================================


def init_db_pool() -> SimpleConnectionPool | None:
    """Initialize Postgres connection pool and ensure documents table exists."""
    try:
        pool = SimpleConnectionPool(
            POSTGRES_POOL_MIN,
            POSTGRES_POOL_MAX,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            sslmode=POSTGRES_SSLMODE,
        )
        ensure_documents_table(pool)
        ensure_queries_table(pool)
        logger.info("Initialized Postgres pool for document tracking")
        return pool
    except Exception:
        logger.exception("Failed to initialize Postgres pool; document tracking disabled")
        return None


def ensure_documents_table(pool: SimpleConnectionPool) -> None:
    """Create Documents table if it does not already exist."""
    if not pool:
        return

    conn = pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        document_id UUID PRIMARY KEY,
                        document_status TEXT NOT NULL,
                        file_path TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
    except Exception:
        logger.exception("Failed to ensure documents table")
    finally:
        pool.putconn(conn)

def ensure_queries_table(pool: SimpleConnectionPool) -> None:
    """Create Queries table if it does not already exist."""
    if not pool:
        return

    conn = pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS queries (
                        query_id UUID PRIMARY KEY,
                        query_text TEXT NOT NULL,
                        query_max_steps INT,
                        query_similarity_threshold FLOAT,
                        query_top_k INT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        reasoning_trace JSONB
                    );
                    """
                )
    except Exception:
        logger.exception("Failed to ensure queries table")
    finally:
        pool.putconn(conn)


def record_query(
    query_id: str,
    query_text: str,
    query_max_steps: int = MAX_STEPS,
    query_similarity_threshold: float = SIMILARITY_THRESHOLD,
    query_top_k: int = TOP_K,
) -> None:
    """Insert or update a query record in the Queries table."""
    if not db_pool:
        logger.warning("Postgres pool not available; skipping query record")
        return

    conn = db_pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO queries (query_id, query_text, query_max_steps, query_similarity_threshold, query_top_k)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (query_id) DO UPDATE SET
                        query_text = EXCLUDED.query_text,
                        query_max_steps = EXCLUDED.query_max_steps,
                        query_similarity_threshold = EXCLUDED.query_similarity_threshold,
                        query_top_k = EXCLUDED.query_top_k,
                        updated_at = NOW();
                    """,
                    (query_id, query_text, query_max_steps, query_similarity_threshold, query_top_k),
                )
                logger.info(
                    "Query record upsert succeeded: id=%s text=%s (%s)",
                    query_id,
                    query_text,
                    cur.statusmessage,
                )
    except Exception:
        logger.exception("Failed to upsert query record for %s", query_id)
    finally:
        db_pool.putconn(conn)


def record_document_status(
    document_id: str,
    status: str,
    file_path: str | None = None,
) -> None:
    """Insert or update document status in the Documents table."""
    if not db_pool:
        logger.warning("Postgres pool not available; skipping document status record")
        return

    conn = db_pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents (document_id, document_status, file_path)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (document_id) DO UPDATE SET
                        document_status = EXCLUDED.document_status,
                        file_path = COALESCE(EXCLUDED.file_path, documents.file_path),
                        updated_at = NOW();
                    """,
                    (document_id, status, file_path),
                )
                logger.info(
                    "Document status upsert succeeded: id=%s status=%s file_path=%s (%s)",
                    document_id,
                    status,
                    file_path,
                    cur.statusmessage,
                )
    except Exception:
        logger.exception("Failed to upsert document status for %s", document_id)
    finally:
        db_pool.putconn(conn)


def fetch_document_statuses(document_ids: list[str]) -> list[DocumentStatusRecord]:
    """Fetch statuses for the given document IDs from Postgres."""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database connection not initialized")

    if not document_ids:
        return []

    try:
        # Use strings to avoid psycopg2 UUID adaptation issues when passing arrays
        uuid_ids = [str(uuid.UUID(doc_id)) for doc_id in document_ids]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid document_id format") from exc

    conn = db_pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT document_id::text, document_status, file_path, updated_at
                    FROM documents
                    WHERE document_id = ANY(%s::uuid[]);
                    """,
                    (uuid_ids,),
                )
                rows = cur.fetchall()

        return [
            DocumentStatusRecord(
                document_id=row[0],
                document_status=row[1],
                file_path=row[2],
                updated_at=row[3],
            )
            for row in rows
        ]
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to fetch document statuses")
        raise HTTPException(status_code=500, detail="Failed to fetch document statuses")
    finally:
        db_pool.putconn(conn)

async def check_lightrag_service_health() -> bool:
    """Check if the LightRAG service is healthy."""
    if not http_client:
        return False
    try:
        response = await http_client.get(f"{LIGHTRAG_SERVICE_URL}/health")
        return response.status_code == 200
    except Exception:
        return False


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

    except httpx.ConnectError as e:
        logger.error(f"Cannot connect to pipeline service: {str(e)}")
        yield f"event: error\ndata: {{\"error\": \"Cannot connect to pipeline service at {PIPELINE_SERVICE_URL}\"}}\n\n"
    except Exception as e:
        logger.error(f"Stream error: {str(e)}\n{traceback.format_exc()}", exc_info=True)
        print(f"\n{'='*80}")
        print(f"Stream Error: {str(e)}")
        traceback.print_exc()
        print(f"{'='*80}\n")
        yield f"event: error\ndata: {{\"error\": \"Stream error: {str(e)}\"}}\n\n"


async def call_pipeline_service(query_request: QueryRequest, query_id: str) -> dict[str, Any]:
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
                "query_id": query_id,
                "query": query_request.query,
                "max_steps": query_request.max_steps,
                "similarity_threshold": query_request.similarity_threshold,
                "top_k": query_request.top_k,
            },
            headers={"Accept": "text/event-stream"},
        ) as response:
            if response.status_code != 200:
                error_text = (await response.aread()).decode().strip()
                detail = error_text or "Pipeline service returned an empty error response"
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Pipeline service error: {detail}"
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
                                error_detail = data.get("error") or data.get("message") or "Unknown pipeline error"
                                raise HTTPException(
                                    status_code=500,
                                    detail=error_detail
                                )
                        except json.JSONDecodeError:
                            pass
                    logger.info(f"Received event: {current_event} with data: {data_str}")
                    current_event = None
                    current_data = []

    except httpx.ConnectError as e:
        logger.error(f"Cannot connect to pipeline service: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to pipeline service at {PIPELINE_SERVICE_URL}"
        )
    except Exception as e:
        logger.error(f"Pipeline service call failed: {str(e)}\n{traceback.format_exc()}", exc_info=True)
        print(f"\n{'='*80}")
        print(f"Pipeline Service Error: {str(e)}")
        traceback.print_exc()
        print(f"{'='*80}\n")
        raise

    if final_result is None:
        logger.error("No result received from pipeline service")
        raise HTTPException(status_code=500, detail="No result received from pipeline service")

    return final_result


async def call_lightrag_service(query_request: QueryRequest) -> dict[str, Any]:
    """Call the LightRAG service directly for a non-streaming response."""
    if not http_client:
        raise HTTPException(status_code=503, detail="HTTP client not initialized")

    if not await check_lightrag_service_health():
        raise HTTPException(
            status_code=503,
            detail=f"LightRAG service not available at {LIGHTRAG_SERVICE_URL}"
        )

    response = await http_client.post(
        f"{LIGHTRAG_SERVICE_URL}/query",
        json={
            "query": query_request.query,
            "mode": "mix",
            "top_k": query_request.top_k,
        },
    )
    response.raise_for_status()
    result = response.json()

    return {
        "status": "success",
        "response": result.get("response", ""),
        "citations": result.get("references", []),
        "confidence": 0.0,
        "steps_taken": 1,
        "converged": False,
        "metadata": {
            "fallback": "lightrag",
        },
    }


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    lightrag_healthy = await check_lightrag_service_health()
    pipeline_healthy = await check_pipeline_service_health()

    return HealthResponse(
        status="healthy",
        rag_initialized=lightrag_healthy,
        pipeline_service_healthy=pipeline_healthy,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    """
    Ingest a document into the LightRAG knowledge graph via the LightRAG service.

    This endpoint calls the LightRAG service's /ingest endpoint to:
    1. Chunk the document
    2. Extract entities and relationships
    3. Build the knowledge graph
    4. Store embeddings for retrieval
    """
    if not http_client:
        raise HTTPException(status_code=503, detail="HTTP client not initialized")

    document_id = request.doc_id or str(uuid.uuid4())
    record_document_status(document_id, "uploaded", request.file_path)

    try:
        # Prepare request body for LightRAG service
        ingest_request = {
            "text": request.content,
        }
        if request.file_path:
            ingest_request["file_path"] = request.file_path

        # Call LightRAG service /ingest endpoint using async client
        response = await http_client.post(
            f"{LIGHTRAG_SERVICE_URL}/ingest",
            json=ingest_request,
        )
        response.raise_for_status()

        result = response.json()

        return IngestResponse(
            status=result.get("status", "success"),
            message=result.get("message", "Document ingested successfully"),
            track_id=result.get("track_id"),
            doc_id=document_id,
        )

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to LightRAG service at {LIGHTRAG_SERVICE_URL}"
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"LightRAG service HTTP error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"LightRAG service error: {e.response.text}"
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}\n{traceback.format_exc()}", exc_info=True)
        print(f"\n{'='*80}")
        print(f"Ingestion Error: {str(e)}")
        traceback.print_exc()
        print(f"{'='*80}\n")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    doc_id: str | None = Form(None),
):
    """
    Ingest a file into the LightRAG knowledge graph via the LightRAG service.

    Supports text files (.txt, .md, .json, etc.)
    """
    if not http_client:
        raise HTTPException(status_code=503, detail="HTTP client not initialized")

    document_id = doc_id or str(uuid.uuid4())
    record_document_status(document_id, "uploaded", file.filename)

    try:
        import tempfile
        import fitz  # PyMuPDF
        import os as os_module

        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os_module.path.splitext(file.filename)[1]) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        try:
            # Extract text content from PDF using PyMuPDF
            doc = fitz.open(temp_file_path)
            text_content = ""
            for page in doc:
                text_content += page.get_text()
            doc.close()
            
            # Handle encoding issues gracefully
            if isinstance(text_content, bytes):
                text_content = text_content.decode("utf-8", errors="replace")

            # Clean up any replacement characters if needed
            if not text_content.strip():
                raise ValueError("No text content could be extracted from the file")

            # Prepare request body for LightRAG service
            ingest_request = {
                "text": text_content,
                "document_id": document_id,
            }
            if file.filename:
                ingest_request["file_path"] = file.filename

            # Call LightRAG service /ingest endpoint using async client
            response = await http_client.post(
                f"{LIGHTRAG_SERVICE_URL}/ingest",
                json=ingest_request,
            )
            response.raise_for_status()

            result = response.json()

            return IngestResponse(
                status=result.get("status", "success"),
                message=f"File '{file.filename}' ingested successfully",
                track_id=result.get("track_id"),
                doc_id=document_id,
            )
        finally:
            # Clean up temporary file
            try:
                os_module.unlink(temp_file_path)
            except OSError:
                pass

    except ValueError as ve:
        logger.error(f"File ingestion validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to LightRAG service at {LIGHTRAG_SERVICE_URL}"
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"LightRAG service HTTP error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"LightRAG service error: {e.response.text}"
        )
    except Exception as e:
        logger.error(f"File ingestion failed: {str(e)}\n{traceback.format_exc()}", exc_info=True)
        print(f"\n{'='*80}")
        print(f"File Ingestion Error: {str(e)}")
        traceback.print_exc()
        print(f"{'='*80}\n")
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
    
    # Record the query in the database for tracking
    query_id = str(uuid.uuid4())
    record_query(query_id, request.query, request.max_steps, request.similarity_threshold, request.top_k)

    # Check if pipeline service is available
    if not await check_pipeline_service_health():
        raise HTTPException(
            status_code=503,
            detail=f"Pipeline service not available at {PIPELINE_SERVICE_URL}"
        )
    # Wait for final result and return it
    result = await call_pipeline_service(request, query_id=query_id)
    result.setdefault("metadata", {})["query_id"] = query_id
    return QueryResponse(**result)


@app.post("/query/simple")
async def query_simple(request: QueryRequest):
    """
    Simple query endpoint that uses the LightRAG service directly without the multi-agent pipeline.

    Useful for quick queries or debugging.
    """
    if not await check_lightrag_service_health():
        raise HTTPException(
            status_code=503,
            detail=f"LightRAG service not available at {LIGHTRAG_SERVICE_URL}"
        )

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # Call LightRAG service /query endpoint
        response = await http_client.post(
            f"{LIGHTRAG_SERVICE_URL}/query",
            json={
                "query": request.query,
                "mode": "mix",
                "top_k": request.top_k,
            },
        )
        response.raise_for_status()

        result = response.json()

        return {
            "status": "success",
            "response": result.get("response", ""),
            "query": request.query,
        }

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to LightRAG service at {LIGHTRAG_SERVICE_URL}"
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"LightRAG service HTTP error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"LightRAG service error: {e.response.text}"
        )
    except Exception as e:
        logger.error(f"Query failed: {str(e)}\n{traceback.format_exc()}", exc_info=True)
        print(f"\n{'='*80}")
        print(f"Query Error: {str(e)}")
        traceback.print_exc()
        print(f"{'='*80}\n")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/status/{track_id}")
async def get_ingestion_status(track_id: str):
    """
    Get the status of a document ingestion job.
    """
    if not http_client:
        raise HTTPException(status_code=503, detail="HTTP client not initialized")

    try:
        response = await http_client.get(
            f"{LIGHTRAG_SERVICE_URL}/document/status/{track_id}"
        )
        response.raise_for_status()
        status = response.text.strip('"')
        return {
            "track_id": track_id,
            "status": status,
        }
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to LightRAG service at {LIGHTRAG_SERVICE_URL}"
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get status: {str(e)}\n{traceback.format_exc()}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@app.post("/documents/status", response_model=DocumentStatusResponse)
async def get_document_statuses(request: DocumentStatusRequest):
    """Return the current statuses for the requested document IDs."""
    documents = fetch_document_statuses(request.document_ids)
    return DocumentStatusResponse(documents=documents)


@app.get("/reasoning_trace/{query_id}")
async def get_reasoning_trace(query_id: str):
    """Fetch the reasoning trace for a given query ID."""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database connection not initialized")

    conn = db_pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT reasoning_trace
                    FROM queries
                    WHERE query_id = %s;
                    """,
                    (query_id,),
                )
                row = cur.fetchone()

        if row and row[0]:
            return {"query_id": query_id, "reasoning_trace": row[0]}
        else:
            raise HTTPException(status_code=404, detail="Reasoning trace not found for query_id: " + query_id)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to fetch reasoning trace for query_id: %s", query_id)
        raise HTTPException(status_code=500, detail="Failed to fetch reasoning trace")
    finally:
        db_pool.putconn(conn)
# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=False,
    )
