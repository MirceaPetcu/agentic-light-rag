"""
FastAPI Backend for Multi-Agent LightRAG Framework.

This module provides two main endpoints:
1. /ingest - Ingest documents using LightRAG's ingestion pipeline
2. /query - Query using the multi-agent architecture
"""

import os
import sys
from contextlib import asynccontextmanager
from functools import partial
from typing import Any

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

# Add light_rag to the path for imports
sys.path.insert(0, './light_rag')

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.ollama import ollama_embed
from lightrag.utils import EmbeddingFunc

# Import agents
from query_rewriter import QueryRewriterAgent
from retriever import RetrieverAgent, RetrieverResult
from deducer_model import DeducerAgent
from judge import JudgeAgent
from respone_agent import ResponseAgent
from observation import Observation

# Configuration from environment
WORKING_DIR = os.getenv("LIGHTRAG_WORKING_DIR", "./rag_storage")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
VLLM_MODEL = os.getenv("VLLM_MODEL", "LiquidAI/LFM2-2.6B")

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

# Multi-agent configuration
MAX_STEPS = int(os.getenv("MAX_STEPS", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
TOP_K = int(os.getenv("TOP_K", "10"))

# Global instances
rag: LightRAG | None = None
redis_client: redis.Redis | None = None


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
    redis_connected: bool


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - initialize and cleanup resources."""
    global rag, redis_client
    
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
                ollama_embed.func,
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
    
    # Initialize Redis (optional - for caching between steps)
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        await redis_client.ping()
        print(f"Redis connected: {REDIS_URL}")
    except Exception as e:
        print(f"Redis connection failed (optional): {e}")
        redis_client = None
    
    yield
    
    # Shutdown
    print("Shutting down...")
    if rag:
        await rag.finalize_storages()
    if redis_client:
        await redis_client.close()
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

def create_agents(
    base_url: str = VLLM_BASE_URL,
    api_key: str = VLLM_API_KEY,
    model: str = VLLM_MODEL,
    top_k: int = TOP_K,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> dict[str, Any]:
    """Create all agents for the multi-agent pipeline."""
    return {
        "query_rewriter": QueryRewriterAgent(
            name="query_rewriter",
            base_url=base_url,
            api_key=api_key,
            model=model,
        ),
        "retriever": RetrieverAgent(
            name="retriever",
            lightrag=rag,
            top_k=top_k,
            query_mode="mix",
        ),
        "deducer": DeducerAgent(
            name="deducer",
            base_url=base_url,
            api_key=api_key,
            model=model,
        ),
        "judge": JudgeAgent(
            name="judge",
            base_url=base_url,
            api_key=api_key,
            model=model,
            similarity_threshold=similarity_threshold,
        ),
        "response": ResponseAgent(
            name="response",
            base_url=base_url,
            api_key=api_key,
            model=model,
        ),
    }


async def run_multi_agent_pipeline(
    query: str,
    max_steps: int = MAX_STEPS,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """
    Run the multi-agent retrieval pipeline.
    
    Pipeline:
    Step 0: Save the original user query
    Step 1: Decompose the user query
    Step 2: Contextualize the decomposed subqueries using LLM
    Step 3: Retrieve initial knowledge graph context using the enriched queries
    
    Loop until convergence or max_steps:
        Step 4: LLM infers what answer could be found from the retrieved context
        Step 5: LLM infers what was the original query addressed from the retrieved context
        Step 6: Compute similarity between original and inferred query
        Step 7: If similarity <= threshold, identify missing context using LLM
        Step 8: Generate subsequent subqueries from missing context using LLM
        Step 9: Retrieve knowledge graph context again
    
    Final: response_generation_agent(full_retrieved_context, original_query)
    """
    # Create agents
    agents = create_agents(
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )
    
    query_rewriter = agents["query_rewriter"]
    retriever = agents["retriever"]
    deducer = agents["deducer"]
    judge = agents["judge"]
    response_agent = agents["response"]
    
    # Step 0: Save original query
    original_query = query
    # gather some global context of the knowledge base related to the user query
    # for more informed decomposition and retrieval
    global_context_result = await retriever.act(Observation(
        original_query=original_query,
        subqueries=[original_query],
        step=0,
        max_steps=max_steps,
        similarity_threshold=similarity_threshold,
    ), mode='global')
    global_context = retriever.get_context_as_string(global_context_result)

    # Step 1 & 2: Decompose and contextualize the query
    decomposition = await query_rewriter._decompose_query(original_query, context=global_context)
    subqueries = [sq.canonical_form for sq in decomposition.subqueries]
    
    if not subqueries:
        # If no subqueries, use the original query
        subqueries = [original_query]
    
    # Initialize observation
    observation = Observation(
        original_query=original_query,
        subqueries=subqueries,
        step=1,
        max_steps=max_steps,
        similarity_threshold=similarity_threshold,
    )
    
    # Step 3: Initial retrieval
    retriever_result: RetrieverResult = await retriever.act(observation)
    
    # Build context string from retrieved contexts
    context_string = retriever.get_context_as_string(retriever_result)
    
    # Store contexts in observation
    observation.retrieved_context = [ctx.to_dict() for ctx in retriever_result.contexts]
    
    # Main loop
    current_step = 1
    converged = False
    inferred_answer = {}
    inferred_query = {}

    while current_step < max_steps and not converged:
        # Step 4 & 5: Deduce answer and query from context
        deduction_result = await deducer.deduce(context_string)
        
        inferred_answer = deduction_result["inferred_answer"]
        inferred_query = deduction_result["inferred_query"]
        
        observation.inferred_answer = inferred_answer.get("answer", "")
        observation.inferred_query = inferred_query.get("query", "")
        
        # Step 6, 7, 8: Judge convergence and generate new subqueries if needed
        judgement = await judge.judge(
            original_query=original_query,
            inferred_query=observation.inferred_query,
            inferred_answer=observation.inferred_answer,
            context=context_string,
            threshold=similarity_threshold,
        )
        
        observation.similarity_score = judgement.similarity_score
        observation.converged = judgement.converged
        
        if judgement.converged:
            converged = True
            break
        
        # Step 8: Generate new subqueries from missing context
        if judgement.new_subqueries:
            new_subqueries = [sq.canonical_form for sq in judgement.new_subqueries.subqueries]
            observation.subqueries = new_subqueries
            observation.step = current_step + 1
            
            # Step 9: Retrieve again with new subqueries
            # Note: For now, we do a fresh retrieval. 
            # TODO: Implement incremental retrieval that merges with old context
            try:
                retriever_result = await retriever.act(observation)
                context_string = retriever.get_context_as_string(retriever_result)
                observation.retrieved_context = [ctx.to_dict() for ctx in retriever_result.contexts]
            except NotImplementedError:
                # Subsequent step retrieval not implemented yet
                # Fall back to using the existing context
                pass
        
        current_step += 1
    
    # Final step: Generate response with citations
    response_observation = {
        "original_query": original_query,
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
    
    response_result = await response_agent.act(response_observation)
    
    return {
        "status": "success",
        "response": response_result.get("response", ""),
        "citations": response_result.get("citations", []),
        "confidence": response_result.get("confidence", 0.0),
        "steps_taken": current_step,
        "converged": converged,
        "metadata": {
            "original_query": original_query,
            "final_similarity_score": observation.similarity_score,
            "subqueries_used": observation.subqueries,
            "key_points": response_result.get("key_points", []),
            "limitations": response_result.get("limitations", []),
        }
    }


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    redis_connected = False
    if redis_client:
        try:
            await redis_client.ping()
            redis_connected = True
        except Exception:
            pass
    
    return HealthResponse(
        status="healthy",
        rag_initialized=rag is not None,
        redis_connected=redis_connected,
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
            except:
                pass
    
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File ingestion failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the knowledge graph using the multi-agent architecture.
    
    This endpoint triggers the following pipeline:
    1. Decompose the user query into subqueries
    2. Retrieve context from the knowledge graph
    3. Iteratively refine until convergence or max steps
    4. Generate a comprehensive response with citations
    """
    if not rag:
        raise HTTPException(status_code=503, detail="LightRAG not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = await run_multi_agent_pipeline(
            query=request.query,
            max_steps=request.max_steps,
            similarity_threshold=request.similarity_threshold,
            top_k=request.top_k,
        )
        
        return QueryResponse(**result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


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
