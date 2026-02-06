"""
Multi-Agent Pipeline Service with Server-Sent Events (SSE).

This service handles the multi-agent retrieval pipeline and streams
progress updates via SSE. It provides three endpoints:
1. /health - Health check
2. /query - Multi-agent pipeline with SSE streaming
3. /query/simple - Simple prompt-response query
"""

import asyncio
import json
import os
import sys
import uuid
from urllib.parse import urlparse
from contextlib import asynccontextmanager
from typing import Any

import httpx

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from torchgen import local

# Import agents
from query_rewriter import QueryRewriterAgent
from retriever import RetrieverAgent, RetrieverResult
from deducer_model import DeducerAgent
from judge import JudgeAgent
from respone_agent import ResponseAgent
from observation import Observation
import logging
from psycopg2.pool import SimpleConnectionPool
from psycopg2 import sql


logger = logging.getLogger(__name__)

POSTGRES_POOL_MIN = int(os.getenv("POSTGRES_POOL_MIN", "1"))
POSTGRES_POOL_MAX = int(os.getenv("POSTGRES_POOL_MAX", "5"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "lightrag")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_SSLMODE = os.getenv("POSTGRES_SSLMODE", "prefer")

# Configuration from environment
WORKING_DIR = os.getenv("LIGHTRAG_WORKING_DIR", "./rag_storage")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
VLLM_MODEL = os.getenv("VLLM_MODEL", "LiquidAI/LFM2-2.6B")

ZAI_BASE_URL = os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4/")
ZAI_API_KEY = os.getenv("ZAI_API_KEY", "")
ZAI_MODEL = os.getenv("ZAI_MODEL", "glm-4.7-flash")

# LightRAG Service Configuration (container)
LIGHTRAG_HOST = os.getenv("LIGHTRAG_HOST", "localhost")
LIGHTRAG_PORT = os.getenv("LIGHTRAG_PORT", "8005")
LIGHTRAG_API_KEY = os.getenv("LIGHTRAG_API_KEY", "")

# LLM Configuration
LLM_BINDING_HOST = os.getenv("LLM_BINDING_HOST", "https://api.deepseek.com")
LLM_BINDING_API_KEY = os.getenv("LLM_BINDING_API_KEY", os.getenv("OPENAI_API_KEY", ""))
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# Embedding Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")
EMBEDDING_HOST = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
MAX_EMBED_TOKENS = int(os.getenv("MAX_EMBED_TOKENS", "32768"))

# Multi-agent configuration
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


LIGHTRAG_SERVICE_URL = build_service_url(LIGHTRAG_HOST, LIGHTRAG_PORT)

# Store for active pipeline jobs
pipeline_jobs: dict[str, dict[str, Any]] = {}


# ============================================================================
# HTTP Client for LightRAG Service
# ============================================================================

_http_client: httpx.AsyncClient | None = None

async def get_lightrag_client() -> httpx.AsyncClient:
    """Get or create HTTP client for LightRAG service."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            base_url=LIGHTRAG_SERVICE_URL,
            timeout=httpx.Timeout(600.0, connect=10.0),
            headers={"X-API-Key": LIGHTRAG_API_KEY} if LIGHTRAG_API_KEY else {}
        )
    return _http_client


# ============================================================================
# Pydantic Models
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for querying."""
    query_id: str = Field(..., description="Unique ID for the query")
    query: str = Field(..., description="The user query")
    max_steps: int = Field(default=MAX_STEPS, description="Maximum number of iteration steps")
    similarity_threshold: float = Field(default=SIMILARITY_THRESHOLD, description="Convergence threshold")
    top_k: int = Field(default=TOP_K, description="Number of top results to retrieve")


class SimpleQueryRequest(BaseModel):
    """Request model for simple querying."""
    query: str = Field(..., description="The user query")
    top_k: int = Field(default=TOP_K, description="Number of top results to retrieve")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    lightrag_service_available: bool
    service: str = "pipeline"


class SSEEvent(BaseModel):
    """Model for SSE event data."""
    event: str
    data: dict[str, Any]


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - initialize and cleanup resources."""
    global _http_client, db_pool

    # Startup
    print("Initializing Pipeline Service...")
    print(f"LightRAG Service URL: {LIGHTRAG_SERVICE_URL}")

    # Initialize HTTP client for LightRAG service
    await get_lightrag_client()
    # Initialize Postgres connection pool for document status tracking
    db_pool = init_db_pool()
    print("Pipeline Service initialized")
    

    yield

    # Shutdown
    print("Shutting down Pipeline Service...")
    if _http_client:
        await _http_client.aclose()
        _http_client = None
    if db_pool:
        db_pool.closeall()
    print("Pipeline Service cleanup complete.")


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
        ensure_queries_table(pool)
        logger.info("Initialized Postgres pool for document tracking")
        return pool
    except Exception:
        logger.exception("Failed to initialize Postgres pool; document tracking disabled")
        return None
    
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


def update_query_reasoning_trace(
    query_id: str,
    query_reasoning_trace: dict[str, Any],
) -> None:
    """Update the reasoning_trace field for a query record."""
    if not db_pool:
        logger.warning("Postgres pool not available; skipping query record")
        return

    conn = db_pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE queries
                    SET reasoning_trace = %s,
                        updated_at = NOW()
                    WHERE query_id = %s;
                    """,
                    (json.dumps(query_reasoning_trace), query_id),
                )
                if cur.rowcount == 0:
                    logger.warning("No query record found for query_id=%s", query_id)
                else:
                    logger.info(
                        "Query reasoning_trace updated: id=%s (%s)",
                        query_id,
                        cur.statusmessage,
                    )
    except Exception:
        logger.exception("Failed to update query record for %s", query_id)
    finally:
        db_pool.putconn(conn)
# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Multi-Agent Pipeline Service",
    description="Pipeline service for multi-agent querying with SSE streaming",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# Helper Functions
# ============================================================================

def create_agents(
    top_k: int = TOP_K,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> dict[str, Any]:
    """Create all agents for the multi-agent pipeline."""
    return {
        "query_rewriter": QueryRewriterAgent(
            name="query_rewriter",
            base_url=VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
            model=VLLM_MODEL,
        ),
        "retriever": RetrieverAgent(
            name="retriever",
            lightrag_base_url=LIGHTRAG_SERVICE_URL,
            lightrag_api_key=LIGHTRAG_API_KEY,
            top_k=top_k,
            query_mode="mix",
        ),
        "deducer": DeducerAgent(
            name="deducer",
            base_url=VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
            model=VLLM_MODEL,
        ),
        "judge": JudgeAgent(
            name="judge",
            base_url=VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
            model=VLLM_MODEL,
            similarity_threshold=similarity_threshold,
        ),
        "response": ResponseAgent(
            name="response",
            base_url=VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
            model=VLLM_MODEL,
        ),
    }


def format_sse(event: str, data: dict[str, Any]) -> str:
    """Format data as SSE message."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def run_pipeline_with_sse(
    job_id: str,
    query: str,
    query_id: str,
    max_steps: int,
    similarity_threshold: float,
    top_k: int,
):
    """
    Run the multi-agent pipeline and yield SSE events.

    Yields SSE events for each stage of the pipeline:
    - started: Pipeline started
    - decomposition: Query decomposed into subqueries
    - retrieval: Context retrieved
    - deduction: Answer and query inferred
    - judgement: Convergence checked
    - iteration: New iteration started (if not converged)
    - response: Final response generated
    - completed: Pipeline completed
    - error: Error occurred
    """
    assert query_id is not None, "query_id must be provided for tracking"
    try:
        query_logs = {}
        update_query_reasoning_trace(query_id=query_id, query_reasoning_trace={"status": "started"})
        # Update job status
        pipeline_jobs[job_id] = {"status": "running", "progress": 0}

        logger.info(f"Job {job_id} started with query: {query_id}")
        # Emit start event
        yield format_sse("started", {
            "job_id": job_id,
            "query": query,
            "max_steps": max_steps,
            "message": "Pipeline started"
        })

        # Create agents
        agents = create_agents(top_k=top_k, similarity_threshold=similarity_threshold)
        query_rewriter = agents["query_rewriter"]
        retriever = agents["retriever"]
        deducer = agents["deducer"]
        judge = agents["judge"]
        response_agent = agents["response"]


        # We no longer perform query decomposition in the initial step.
        # # Step 1 & 2: Decompose query
        # yield format_sse("progress", {
        #     "job_id": job_id,
        #     "stage": "decomposition",
        #     "message": "Decomposing query into subqueries..."
        # })

        # # decomposition = await query_rewriter._decompose_query(query)
        # # subqueries = [sq.canonical_form for sq in decomposition.subqueries]
        subqueries = None
        if not subqueries:
            subqueries = [query]

        # yield format_sse("decomposition", {
        #     "job_id": job_id,
        #     "subqueries": subqueries,
        #     "count": len(subqueries),
        #     "message": f"Query decomposed into {len(subqueries)} subqueries"
        # })

        # Initialize observation
        observation = Observation(
            original_query=query,
            subqueries=subqueries,
            step=1,
            max_steps=max_steps,
            similarity_threshold=similarity_threshold,
        )

        # Step 3: Initial retrieval
        yield format_sse("progress", {
            "job_id": job_id,
            "stage": "retrieval",
            "step": 1,
            "message": "Retrieving context from knowledge graph..."
        })

        retriever_result: RetrieverResult = await retriever.act(observation)
        context_string = retriever.get_context_as_string(retriever_result)
        local_step = 1
        query_logs[f'{local_step}'] = {}

        query_logs[f'{local_step}']['first_retrieval_context'] = [ctx.to_dict() for ctx in retriever_result.contexts]
        local_step += 1
        observation.retrieved_context = [ctx.to_dict() for ctx in retriever_result.contexts]
        logger.info(f"Job {job_id} retrieved {len(retriever_result.contexts)} context items in initial retrieval")

        yield format_sse("retrieval", {
            "job_id": job_id,
            "step": 1,
            "contexts_count": len(retriever_result.contexts),
            "message": f"Retrieved {len(retriever_result.contexts)} context items"
        })

        # Main loop
        current_step = 1
        converged = False
        inferred_answer = {}
        inferred_query = {}

        while current_step < max_steps and not converged:
            pipeline_jobs[job_id]["progress"] = int((current_step / max_steps) * 100)

            # Step 4 & 5: Deduce
            yield format_sse("progress", {
                "job_id": job_id,
                "stage": "deduction",
                "step": current_step,
                "message": f"Step {current_step}: Inferring answer and query from context..."
            })

            deduction_result = await deducer.deduce(context_string)
            inferred_answer = deduction_result["inferred_answer"]
            inferred_query = deduction_result["inferred_query"]
            query_logs[f'{local_step}'] = {}

            query_logs[f'{local_step}'][f"deduction_step_{current_step}"] = {
                "inferred_answer": inferred_answer,
                "inferred_query": inferred_query,
            }
            local_step += 1
            logger.info(f"Job {job_id} deduction result: {query_logs[f'{local_step - 1}'][f'deduction_step_{current_step}']}")
            observation.inferred_answer = inferred_answer.get("answer", "")
            observation.inferred_query = inferred_query.get("query", "")

            yield format_sse("deduction", {
                "job_id": job_id,
                "step": current_step,
                "inferred_answer_preview": observation.inferred_answer[:200] + "..." if len(observation.inferred_answer) > 200 else observation.inferred_answer,
                "inferred_query": observation.inferred_query,
                "answer_confidence": inferred_answer.get("confidence", 0),
                "query_confidence": inferred_query.get("confidence", 0),
            })

            # Step 6, 7, 8: Judge
            yield format_sse("progress", {
                "job_id": job_id,
                "stage": "judgement",
                "step": current_step,
                "message": f"Step {current_step}: Evaluating convergence..."
            })

            judgement = await judge.judge(
                original_query=query,
                inferred_query=observation.inferred_query,
                inferred_answer=observation.inferred_answer,
                context=context_string,
                threshold=similarity_threshold,
            )

            observation.similarity_score = judgement.similarity_score
            observation.converged = judgement.converged

            query_logs[f'{local_step}'] = {}
            query_logs[f'{local_step}'][f"judgement_step_{current_step}"] = {
                "similarity_score": judgement.similarity_score,
                "converged": judgement.converged,
                "new_subqueries": [sq.canonical_form for sq in judgement.new_subqueries.subqueries] if judgement.new_subqueries else [],
                "missing_context": judgement.missing_context,
            }
            local_step += 1

            logger.info(f"Job {job_id} judgement result: {query_logs[f'{local_step - 1}'][f'judgement_step_{current_step}']}")


            yield format_sse("judgement", {
                "job_id": job_id,
                "step": current_step,
                "similarity_score": judgement.similarity_score,
                "threshold": similarity_threshold,
                "converged": judgement.converged,
                "message": f"Similarity: {judgement.similarity_score:.3f} (threshold: {similarity_threshold})"
            })

            if judgement.converged:
                converged = True
                break

            # Step 8: Generate new subqueries
            if judgement.new_subqueries:
                new_subqueries = [sq.canonical_form for sq in judgement.new_subqueries.subqueries]
                observation.subqueries = new_subqueries
                observation.step = current_step + 1

                yield format_sse("iteration", {
                    "job_id": job_id,
                    "step": current_step + 1,
                    "new_subqueries": new_subqueries,
                    "message": f"Generating {len(new_subqueries)} new subqueries for step {current_step + 1}"
                })

                # Step 9: Retrieve again
                yield format_sse("progress", {
                    "job_id": job_id,
                    "stage": "retrieval",
                    "step": current_step + 1,
                    "message": f"Step {current_step + 1}: Retrieving additional context..."
                })

                try:
                    retriever_result = await retriever.act(observation)
                    context_string = retriever.get_context_as_string(retriever_result)
                    observation.retrieved_context = [ctx.to_dict() for ctx in retriever_result.contexts]
                    query_logs[f'{local_step}'] = {}
                    query_logs[f'{local_step}'][f"retrieval_step_{current_step + 1}"] = {
                        "fused_context": [ctx.to_dict() for ctx in retriever_result.contexts]
                    }
                    local_step += 1

                    logger.info(f"Job {job_id} retrieved {len(retriever_result.contexts)} context items in retrieval step {current_step + 1}")

                    yield format_sse("retrieval", {
                        "job_id": job_id,
                        "step": current_step + 1,
                        "contexts_count": len(retriever_result.contexts),
                        "message": f"Retrieved {len(retriever_result.contexts)} context items"
                    })
                except NotImplementedError:
                    pass

            current_step += 1

        # Final step: Generate response
        yield format_sse("progress", {
            "job_id": job_id,
            "stage": "response_generation",
            "message": "Generating final response with citations..."
        })

        response_observation = {
            "original_query": query,
            "contexts": observation.retrieved_context,
            "inferred_answer": {
                "answer": observation.inferred_answer,
                "supporting_evidence": inferred_answer.get("supporting_evidence", []),
            } if observation.inferred_answer else {},
            "inferred_query": {
                "query": observation.inferred_query,
                "key_topics": inferred_query.get("key_topics", []),
            } if observation.inferred_query else {},
            "rrf_scores": retriever_result.rrf_scores,
        }

        logger.info(f"Job {job_id} response observation: {response_observation}")
        query_logs[f'{local_step}'] = {}
        query_logs[f'{local_step}']["final_response_observation"] = response_observation
        local_step += 1

        response_result = await response_agent.act(response_observation)

        # Build final result
        result = {
            "status": "success",
            "response": response_result.get("response", ""),
            "citations": response_result.get("citations", []),
            "confidence": response_result.get("confidence", 0.0),
            "steps_taken": current_step,
            "converged": converged,
            "metadata": {
                "original_query": query,
                "final_similarity_score": observation.similarity_score,
                "subqueries_used": observation.subqueries,
                "key_points": response_result.get("key_points", []),
                "limitations": response_result.get("limitations", []),
            }
        }
        logger.info(f"Job {job_id} final result: {result}")
        query_logs[f'{local_step}'] = {}
        query_logs[f'{local_step}']["final_result"] = result
        local_step += 1


        # Update job status
        pipeline_jobs[job_id] = {"status": "completed", "progress": 100, "result": result}

        # Emit completion event
        yield format_sse("completed", {
            "job_id": job_id,
            "result": result,
            "message": f"Pipeline completed in {current_step} steps (converged: {converged})"
        })

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        pipeline_jobs[job_id] = {"status": "error", "error": str(e), "traceback": error_trace}

        query_logs[f'{local_step}'] = {}
        query_logs[f'{local_step}']["error"] = {
            "error_message": str(e),
            "traceback": error_trace,
        }
        local_step += 1
        logger.error(f"Job {job_id} failed with error: {str(e)}\n{error_trace}")
        yield format_sse("error", {
            "job_id": job_id,
            "error": str(e),
            "traceback": error_trace,
            "message": f"Pipeline failed: {str(e)}"
        })
    finally:
        # store the query log in postgres
        update_query_reasoning_trace(query_id=query_id, query_reasoning_trace=query_logs)
        logger.info(f"Job {job_id} query log stored in database")



# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    lightrag_available = False
    try:
        client = await get_lightrag_client()
        response = await client.get("/health", timeout=5.0)
        lightrag_available = response.status_code == 200
    except Exception:
        pass
    
    return HealthResponse(
        status="healthy",
        lightrag_service_available=lightrag_available,
        service="pipeline",
    )


@app.post("/query")
async def query_with_sse(request: QueryRequest):
    """
    Query the knowledge graph using the multi-agent pipeline with SSE streaming.

    Returns a streaming response with SSE events for each stage of the pipeline.

    Events:
    - started: Pipeline started
    - progress: Progress update
    - decomposition: Query decomposed
    - retrieval: Context retrieved
    - deduction: Answer/query inferred
    - judgement: Convergence checked
    - iteration: New iteration
    - completed: Final result
    - error: Error occurred
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Return SSE stream
    return StreamingResponse(
        run_pipeline_with_sse(
            job_id=job_id,
            query_id=request.query_id,
            query=request.query,
            max_steps=request.max_steps,
            similarity_threshold=request.similarity_threshold,
            top_k=request.top_k,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Job-ID": job_id,
        }
    )


@app.get("/query/status/{job_id}")
async def get_query_status(job_id: str):
    """Get the status of a pipeline job."""
    if job_id not in pipeline_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return pipeline_jobs[job_id]


@app.post("/query/simple")
async def simple_query(request: SimpleQueryRequest):
    """
    Simple query endpoint that uses LightRAG container directly without the multi-agent pipeline.

    Useful for quick queries or debugging.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        client = await get_lightrag_client()
        response = await client.post(
            "/query",
            json={
                "query": request.query,
                "mode": "mix",
                "top_k": request.top_k,
            }
        )
        response.raise_for_status()
        result = response.json()

        return {
            "status": "success",
            "response": result.get("response", ""),
            "query": request.query,
            "references": result.get("references", []),
        }

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"LightRAG service error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("PIPELINE_HOST", "0.0.0.0")
    port = int(os.getenv("PIPELINE_PORT", "8003"))

    uvicorn.run(
        "pipeline:app",
        host=host,
        port=port,
        reload=False,
    )
