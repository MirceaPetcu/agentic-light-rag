"""
LightRAG FastAPI Server

A FastAPI server that exposes LightRAG query endpoints with support for all query modes
and parameters. This server provides comprehensive RAG querying capabilities including
streaming, non-streaming, and data-only retrieval endpoints.
"""

import os
import sys
import json
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Add light_rag to the path for imports
sys.path.insert(0, './light_rag')

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.ollama import ollama_embed
from lightrag.utils import (
    EmbeddingFunc,
    logger,
    generate_track_id,
    sanitize_text_for_encoding,
    compute_mdhash_id,
)

# ============================================================================
# Configuration from environment
# ============================================================================

WORKING_DIR = os.getenv("LIGHTRAG_WORKING_DIR", "./rag_storage")
WORKSPACE = os.getenv("WORKSPACE", "")

# LLM Configuration
LLM_BINDING_HOST = os.getenv("LLM_BINDING_HOST", "https://api.deepseek.com")
LLM_BINDING_API_KEY = os.getenv("LLM_BINDING_API_KEY", os.getenv("OPENAI_API_KEY", ""))
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# Embedding Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")
EMBEDDING_HOST = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
MAX_EMBED_TOKENS = int(os.getenv("MAX_EMBED_TOKENS", "32768"))

# Storage Configuration
KV_STORAGE = os.getenv("KV_STORAGE", "JsonKVStorage")
VECTOR_STORAGE = os.getenv("VECTOR_STORAGE", "NanoVectorDBStorage")
GRAPH_STORAGE = os.getenv("GRAPH_STORAGE", "NetworkXStorage")
DOC_STATUS_STORAGE = os.getenv("DOC_STATUS_STORAGE", "JsonDocStatusStorage")

# Global LightRAG instance
rag: LightRAG | None = None


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
    """LLM function for LightRAG using OpenAI-compatible API."""
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


def create_embedding_func() -> EmbeddingFunc:
    """Create embedding function using Ollama."""
    return EmbeddingFunc(
        func=ollama_embed,
        model_name=EMBEDDING_MODEL,
        embedding_dim=EMBEDDING_DIM,
        max_token_size=MAX_EMBED_TOKENS,
        base_url=EMBEDDING_HOST,
    )


# ============================================================================
# Pydantic Models
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for RAG queries with all LightRAG parameters."""
    
    query: str = Field(
        min_length=3,
        description="The query text to process (minimum 3 characters)",
    )
    
    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = Field(
        default="mix",
        description="Query mode: 'local' (entity-focused), 'global' (pattern analysis), "
                   "'hybrid' (combined), 'naive' (vector search only), 'mix' (recommended), "
                   "'bypass' (direct LLM without retrieval)",
    )
    
    only_need_context: Optional[bool] = Field(
        default=None,
        description="If True, only returns the retrieved context without generating a response.",
    )
    
    only_need_prompt: Optional[bool] = Field(
        default=None,
        description="If True, only returns the generated prompt without producing a response.",
    )
    
    response_type: Optional[str] = Field(
        min_length=1,
        default=None,
        description="Defines the response format. Examples: 'Multiple Paragraphs', "
                   "'Single Paragraph', 'Bullet Points'.",
    )
    
    top_k: Optional[int] = Field(
        ge=1,
        default=None,
        description="Number of top items to retrieve. Represents entities in 'local' mode "
                   "and relationships in 'global' mode.",
    )
    
    chunk_top_k: Optional[int] = Field(
        ge=1,
        default=None,
        description="Number of text chunks to retrieve initially from vector search "
                   "and keep after reranking.",
    )
    
    max_entity_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of tokens allocated for entity context.",
    )
    
    max_relation_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of tokens allocated for relationship context.",
    )
    
    max_total_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum total tokens budget for the entire query context "
                   "(entities + relations + chunks + system prompt).",
    )
    
    hl_keywords: List[str] = Field(
        default_factory=list,
        description="List of high-level keywords to prioritize in retrieval. "
                   "Leave empty to use the LLM to generate the keywords.",
    )
    
    ll_keywords: List[str] = Field(
        default_factory=list,
        description="List of low-level keywords to refine retrieval focus. "
                   "Leave empty to use the LLM to generate the keywords.",
    )
    
    conversation_history: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="History messages are only sent to LLM for context, not used for retrieval. "
                   "Format: [{'role': 'user/assistant', 'content': 'message'}].",
    )
    
    user_prompt: Optional[str] = Field(
        default=None,
        description="User-provided prompt for the query. If provided, this will be used "
                   "instead of the default value from prompt template.",
    )
    
    enable_rerank: Optional[bool] = Field(
        default=None,
        description="Enable reranking for retrieved text chunks. If True but no rerank model "
                   "is configured, a warning will be issued.",
    )
    
    include_references: Optional[bool] = Field(
        default=True,
        description="If True, includes reference list in responses. Affects /query and "
                   "/query/stream endpoints. /query/data always includes references.",
    )
    
    include_chunk_content: Optional[bool] = Field(
        default=False,
        description="If True, includes actual chunk text content in references. Only applies "
                   "when include_references=True. Useful for evaluation and debugging.",
    )
    
    stream: Optional[bool] = Field(
        default=True,
        description="If True, enables streaming output for real-time responses. "
                   "Only affects /query/stream endpoint.",
    )
    
    @field_validator("query", mode="after")
    @classmethod
    def query_strip_after(cls, query: str) -> str:
        return query.strip()
    
    @field_validator("conversation_history", mode="after")
    @classmethod
    def conversation_history_role_check(
        cls, conversation_history: List[Dict[str, Any]] | None
    ) -> List[Dict[str, Any]] | None:
        if conversation_history is None:
            return None
        for msg in conversation_history:
            if "role" not in msg:
                raise ValueError("Each message must have a 'role' key.")
            if not isinstance(msg["role"], str) or not msg["role"].strip():
                raise ValueError("Each message 'role' must be a non-empty string.")
        return conversation_history
    
    def to_query_params(self, is_stream: bool) -> QueryParam:
        """Converts a QueryRequest instance into a QueryParam instance."""
        request_data = self.model_dump(
            exclude_none=True, exclude={"query", "include_chunk_content"}
        )
        param = QueryParam(**request_data)
        param.stream = is_stream
        return param


class ReferenceItem(BaseModel):
    """A single reference item in query responses."""
    
    reference_id: str = Field(description="Unique reference identifier")
    file_path: str = Field(description="Path to the source file")
    content: Optional[List[str]] = Field(
        default=None,
        description="List of chunk contents from this file (only present when "
                   "include_chunk_content=True)",
    )


class QueryResponse(BaseModel):
    """Response model for non-streaming queries."""
    
    response: str = Field(description="The generated response")
    references: Optional[List[ReferenceItem]] = Field(
        default=None,
        description="Reference list (Disabled when include_references=False)",
    )


class QueryDataResponse(BaseModel):
    """Response model for data-only queries."""
    
    status: str = Field(description="Query execution status")
    message: str = Field(description="Status message")
    data: Dict[str, Any] = Field(
        description="Query result data containing entities, relationships, chunks, and references"
    )
    metadata: Dict[str, Any] = Field(
        description="Query metadata including mode, keywords, and processing information"
    )


class IngestDocumentRequest(BaseModel):
    """Request model for ingesting a document."""
    
    text: str = Field(
        min_length=1,
        description="The text content to ingest into the RAG system",
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Optional file path for citation purposes",
    )
    
    @field_validator("text", mode="after")
    @classmethod
    def strip_text_after(cls, text: str) -> str:
        return text.strip()
    
    @field_validator("file_path", mode="after")
    @classmethod
    def strip_file_path_after(cls, file_path: Optional[str]) -> Optional[str]:
        if file_path is None:
            return None
        return file_path.strip() if file_path.strip() else None


class IngestDocumentsRequest(BaseModel):
    """Request model for ingesting multiple documents."""
    
    texts: List[str] = Field(
        min_length=1,
        description="List of text contents to ingest into the RAG system",
    )
    file_paths: Optional[List[str]] = Field(
        default=None,
        description="Optional list of file paths corresponding to each text for citation purposes",
    )
    
    @field_validator("texts", mode="after")
    @classmethod
    def strip_texts_after(cls, texts: List[str]) -> List[str]:
        return [text.strip() for text in texts]
    
    @field_validator("file_paths", mode="after")
    @classmethod
    def strip_file_paths_after(cls, file_paths: Optional[List[str]]) -> Optional[List[str]]:
        if file_paths is None:
            return None
        return [fp.strip() if fp.strip() else None for fp in file_paths]


class IngestResponse(BaseModel):
    """Response model for document ingestion operations."""
    
    status: Literal["success", "duplicated", "failure"] = Field(
        description="Status of the ingestion operation"
    )
    message: str = Field(description="Message describing the operation result")
    track_id: str = Field(description="Tracking ID for monitoring processing status")


# ============================================================================
# FastAPI App Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage LightRAG instance lifecycle."""
    global rag
    
    # Initialize LightRAG
    logger.info("Initializing LightRAG instance...")
    try:
        rag = LightRAG(
            working_dir=WORKING_DIR,
            workspace=WORKSPACE,
            llm_model_func=llm_model_func,
            llm_model_name=LLM_MODEL,
            embedding_func=create_embedding_func(),
            kv_storage=KV_STORAGE,
            vector_storage=VECTOR_STORAGE,
            graph_storage=GRAPH_STORAGE,
            doc_status_storage=DOC_STATUS_STORAGE,
        )
        
        # Initialize storages
        await rag.initialize_storages()
        logger.info("LightRAG instance initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize LightRAG: {e}", exc_info=True)
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down LightRAG instance...")
    if rag:
        try:
            await rag.finalize_storages()
            logger.info("LightRAG instance finalized successfully")
        except Exception as e:
            logger.error(f"Error during LightRAG finalization: {e}", exc_info=True)


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="LightRAG Query API",
    description="FastAPI server for LightRAG query endpoints with support for all query modes",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Helper Functions for Document Ingestion
# ============================================================================

async def pipeline_index_texts(
    rag: LightRAG,
    texts: List[str],
    file_paths: Optional[List[str]] = None,
    track_id: Optional[str] = None,
):
    """Index a list of texts with track_id
    
    Args:
        rag: LightRAG instance
        texts: The texts to index
        file_paths: Sources of the texts (optional)
        track_id: Optional tracking ID
    """
    if not texts:
        return
    if file_paths is not None:
        if len(file_paths) != 0 and len(file_paths) != len(texts):
            # Pad file_paths to match texts length
            file_paths = list(file_paths)
            file_paths.extend(["unknown_source"] * (len(texts) - len(file_paths)))
    
    await rag.apipeline_enqueue_documents(
        input=texts, file_paths=file_paths, track_id=track_id
    )
    await rag.apipeline_process_enqueue_documents()


# ============================================================================
# Query Endpoints
# ============================================================================

@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Non-streaming RAG Query",
    description="""
    Comprehensive RAG query endpoint with non-streaming response. Parameter "stream" is ignored.
    
    This endpoint performs Retrieval-Augmented Generation (RAG) queries using various modes
    to provide intelligent responses based on your knowledge base.
    
    **Query Modes:**
    - **local**: Focuses on specific entities and their direct relationships
    - **global**: Analyzes broader patterns and relationships across the knowledge graph
    - **hybrid**: Combines local and global approaches for comprehensive results
    - **naive**: Simple vector similarity search without knowledge graph
    - **mix**: Integrates knowledge graph retrieval with vector search (recommended)
    - **bypass**: Direct LLM query without knowledge retrieval
    
    conversation_history parameter is sent to LLM only, does not affect retrieval results.
    """,
    responses={
        200: {
            "description": "Successful RAG query response",
            "content": {
                "application/json": {
                    "example": {
                        "response": "Artificial Intelligence (AI) is a branch of computer science...",
                        "references": [
                            {
                                "reference_id": "1",
                                "file_path": "/documents/ai_overview.pdf"
                            }
                        ]
                    }
                }
            }
        },
        400: {"description": "Bad Request - Invalid input parameters"},
        500: {"description": "Internal Server Error - Query processing failed"},
    },
)
async def query_text(request: QueryRequest):
    """
    Non-streaming RAG query endpoint.
    
    Returns a complete response with optional references. The stream parameter is ignored
    for this endpoint - it always returns a complete response.
    """
    if not rag:
        raise HTTPException(status_code=503, detail="LightRAG instance not initialized")
    
    try:
        param = request.to_query_params(False)  # Ensure stream=False
        param.stream = False
        
        # Use aquery_llm for unified approach
        result = await rag.aquery_llm(request.query, param=param)
        
        # Extract LLM response and references
        llm_response = result.get("llm_response", {})
        data = result.get("data", {})
        references = data.get("references", [])
        
        # Get the non-streaming response content
        response_content = llm_response.get("content", "")
        if not response_content:
            response_content = "No relevant context found for the query."
        
        # Enrich references with chunk content if requested
        if request.include_references and request.include_chunk_content:
            chunks = data.get("chunks", [])
            ref_id_to_content = {}
            for chunk in chunks:
                ref_id = chunk.get("reference_id", "")
                content = chunk.get("content", "")
                if ref_id and content:
                    ref_id_to_content.setdefault(ref_id, []).append(content)
            
            enriched_references = []
            for ref in references:
                ref_copy = ref.copy()
                ref_id = ref.get("reference_id", "")
                if ref_id in ref_id_to_content:
                    ref_copy["content"] = ref_id_to_content[ref_id]
                enriched_references.append(ref_copy)
            references = enriched_references
        
        # Return response with or without references
        if request.include_references:
            return QueryResponse(response=response_content, references=references)
        else:
            return QueryResponse(response=response_content, references=None)
            
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/query/stream",
    summary="Streaming RAG Query",
    description="""
    Advanced RAG query endpoint with flexible streaming response.
    
    This endpoint provides the most flexible querying experience, supporting both real-time streaming
    and complete response delivery based on your integration needs.
    
    **Response Format:**
    - Content-Type: `application/x-ndjson` (Newline-Delimited JSON)
    - Structure: Each line is an independent, valid JSON object
    - First line: `{"references": [...]}` (if include_references=True)
    - Subsequent lines: `{"response": "content chunk"}` or `{"error": "error message"}`
    
    **Query Modes:** Same as /query endpoint (local, global, hybrid, naive, mix, bypass)
    """,
    responses={
        200: {
            "description": "Flexible RAG query response - format depends on stream parameter",
            "content": {
                "application/x-ndjson": {
                    "example": '{"references": [{"reference_id": "1", "file_path": "/documents/ai.pdf"}]}\n{"response": "Artificial Intelligence is"}\n{"response": " a field of computer science"}\n{"response": " that focuses on creating intelligent machines."}'
                }
            }
        },
        400: {"description": "Bad Request - Invalid input parameters"},
        500: {"description": "Internal Server Error - Query processing failed"},
    },
)
async def query_text_stream(request: QueryRequest):
    """
    Streaming RAG query endpoint.
    
    Returns NDJSON streaming response. If stream=False, returns complete response in a single
    NDJSON line. If stream=True, returns multiple NDJSON lines with response chunks.
    """
    if not rag:
        raise HTTPException(status_code=503, detail="LightRAG instance not initialized")
    
    try:
        stream_mode = request.stream if request.stream is not None else True
        param = request.to_query_params(stream_mode)
        
        # Use aquery_llm for unified approach
        result = await rag.aquery_llm(request.query, param=param)
        
        async def stream_generator():
            references = result.get("data", {}).get("references", [])
            llm_response = result.get("llm_response", {})
            
            # Enrich references with chunk content if requested
            if request.include_references and request.include_chunk_content:
                data = result.get("data", {})
                chunks = data.get("chunks", [])
                ref_id_to_content = {}
                for chunk in chunks:
                    ref_id = chunk.get("reference_id", "")
                    content = chunk.get("content", "")
                    if ref_id and content:
                        ref_id_to_content.setdefault(ref_id, []).append(content)
                
                enriched_references = []
                for ref in references:
                    ref_copy = ref.copy()
                    ref_id = ref.get("reference_id", "")
                    if ref_id in ref_id_to_content:
                        ref_copy["content"] = ref_id_to_content[ref_id]
                    enriched_references.append(ref_copy)
                references = enriched_references
            
            if llm_response.get("is_streaming"):
                # Streaming mode: send references first, then stream response chunks
                if request.include_references:
                    yield f"{json.dumps({'references': references})}\n"
                
                response_stream = llm_response.get("response_iterator")
                if response_stream:
                    try:
                        async for chunk in response_stream:
                            if chunk:
                                yield f"{json.dumps({'response': chunk})}\n"
                    except Exception as e:
                        logger.error(f"Streaming error: {str(e)}")
                        yield f"{json.dumps({'error': str(e)})}\n"
            else:
                # Non-streaming mode: send complete response in one message
                response_content = llm_response.get("content", "")
                if not response_content:
                    response_content = "No relevant context found for the query."
                
                complete_response = {"response": response_content}
                if request.include_references:
                    complete_response["references"] = references
                
                yield f"{json.dumps(complete_response)}\n"
        
        return StreamingResponse(
            stream_generator(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "application/x-ndjson",
                "X-Accel-Buffering": "no",
            },
        )
        
    except Exception as e:
        logger.error(f"Error processing streaming query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/query/data",
    response_model=QueryDataResponse,
    summary="Data-Only RAG Query",
    description="""
    Advanced data retrieval endpoint for structured RAG analysis.
    
    This endpoint provides raw retrieval results without LLM generation, perfect for:
    - **Data Analysis**: Examine what information would be used for RAG
    - **System Integration**: Get structured data for custom processing
    - **Debugging**: Understand retrieval behavior and quality
    - **Research**: Analyze knowledge graph structure and relationships
    
    **Key Features:**
    - No LLM generation - pure data retrieval
    - Complete structured output with entities, relationships, and chunks
    - Always includes references for citation
    - Detailed metadata about processing and keywords
    - Compatible with all query modes and parameters
    
    **Query Mode Behaviors:**
    - **local**: Returns entities and their direct relationships + related chunks
    - **global**: Returns relationship patterns across the knowledge graph
    - **hybrid**: Combines local and global retrieval strategies
    - **naive**: Returns only vector-retrieved text chunks (no knowledge graph)
    - **mix**: Integrates knowledge graph data with vector-retrieved chunks
    - **bypass**: Returns empty data arrays (used for direct LLM queries)
    """,
    responses={
        200: {
            "description": "Successful data retrieval response with structured RAG data",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "message": "Query executed successfully",
                        "data": {
                            "entities": [
                                {
                                    "entity_name": "Neural Networks",
                                    "entity_type": "CONCEPT",
                                    "description": "Computational models...",
                                    "source_id": "chunk-123",
                                    "file_path": "/documents/ai_basics.pdf",
                                    "reference_id": "1"
                                }
                            ],
                            "relationships": [],
                            "chunks": [],
                            "references": [
                                {
                                    "reference_id": "1",
                                    "file_path": "/documents/ai_basics.pdf"
                                }
                            ]
                        },
                        "metadata": {
                            "query_mode": "local",
                            "keywords": {
                                "high_level": ["neural", "networks"],
                                "low_level": ["computation", "model"]
                            }
                        }
                    }
                }
            }
        },
        400: {"description": "Bad Request - Invalid input parameters"},
        500: {"description": "Internal Server Error - Data retrieval failed"},
    },
)
async def query_data(request: QueryRequest):
    """
    Data-only RAG query endpoint.
    
    Returns structured retrieval results without LLM generation. Always includes references
    regardless of the include_references parameter.
    """
    if not rag:
        raise HTTPException(status_code=503, detail="LightRAG instance not initialized")
    
    try:
        param = request.to_query_params(False)  # No streaming for data endpoint
        response = await rag.aquery_data(request.query, param=param)
        
        # aquery_data returns the new format with status, message, data, and metadata
        if isinstance(response, dict):
            return QueryDataResponse(**response)
        else:
            return QueryDataResponse(
                status="failure",
                message="Invalid response type",
                data={},
                metadata={},
            )
            
    except Exception as e:
        logger.error(f"Error processing data query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "rag_initialized": rag is not None,
    }


@app.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest Document",
    description="""
    Ingest a single document into the RAG system.
    
    This endpoint allows you to add text content to the knowledge base for later retrieval
    and use in generating responses. The document will be processed asynchronously in the
    background, and you can track its processing status using the returned track_id.
    
    **Processing Flow:**
    1. Document is enqueued for processing
    2. Content is chunked and embedded
    3. Entities and relationships are extracted
    4. Knowledge graph is updated
    5. Document status is updated to "processed"
    
    Use the track_id to monitor processing status via the document status endpoints.
    """,
    responses={
        200: {
            "description": "Document ingestion initiated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "message": "Document successfully received. Processing will continue in background.",
                        "track_id": "insert_20250131_123456_abc123",
                    }
                }
            }
        },
        400: {"description": "Bad Request - Invalid input parameters"},
        500: {"description": "Internal Server Error - Ingestion failed"},
    },
)
async def ingest_document(
    request: IngestDocumentRequest,
    background_tasks: BackgroundTasks,
):
    """
    Ingest a single document into the RAG system.
    
    Returns immediately with a track_id for monitoring processing status.
    The actual processing happens in the background.
    """
    if not rag:
        raise HTTPException(status_code=503, detail="LightRAG instance not initialized")
    
    try:
        # Check if file_path already exists in doc_status storage
        if (
            request.file_path
            and request.file_path.strip()
            and request.file_path != "unknown_source"
        ):
            existing_doc_data = await rag.doc_status.get_doc_by_file_path(
                request.file_path
            )
            if existing_doc_data:
                status = existing_doc_data.get("status", "unknown")
                existing_track_id = existing_doc_data.get("track_id") or ""
                return IngestResponse(
                    status="duplicated",
                    message=f"File path '{request.file_path}' already exists in document storage (Status: {status}).",
                    track_id=existing_track_id,
                )
        
        # Check if content already exists by computing content hash (doc_id)
        sanitized_text = sanitize_text_for_encoding(request.text)
        content_doc_id = compute_mdhash_id(sanitized_text, prefix="doc-")
        existing_doc = await rag.doc_status.get_by_id(content_doc_id)
        if existing_doc:
            # Content already exists, return duplicated with existing track_id
            status = existing_doc.get("status", "unknown")
            existing_track_id = existing_doc.get("track_id") or ""
            return IngestResponse(
                status="duplicated",
                message=f"Identical content already exists in document storage (doc_id: {content_doc_id}, Status: {status}).",
                track_id=existing_track_id,
            )
        
        # Generate track_id for document ingestion
        track_id = generate_track_id("insert")
        
        # Add background task for processing
        background_tasks.add_task(
            pipeline_index_texts,
            rag,
            [request.text],
            file_paths=[request.file_path] if request.file_path else None,
            track_id=track_id,
        )
        
        return IngestResponse(
            status="success",
            message="Document successfully received. Processing will continue in background.",
            track_id=track_id,
        )
        
    except Exception as e:
        logger.error(f"Error processing document ingestion: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/ingest/batch",
    response_model=IngestResponse,
    summary="Ingest Multiple Documents",
    description="""
    Ingest multiple documents into the RAG system in a single request.
    
    This endpoint allows you to add multiple text contents to the knowledge base at once.
    All documents will be processed asynchronously in the background, and you can track
    their processing status using the returned track_id.
    
    **Note:** If any document is duplicated (by file_path or content), the entire batch
    will return a "duplicated" status with the existing track_id.
    
    **Processing Flow:** Same as single document ingestion, but processes multiple documents
    in parallel based on max_parallel_insert configuration.
    """,
    responses={
        200: {
            "description": "Documents ingestion initiated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "message": "Documents successfully received. Processing will continue in background.",
                        "track_id": "insert_20250131_123456_abc123",
                    }
                }
            }
        },
        400: {"description": "Bad Request - Invalid input parameters"},
        500: {"description": "Internal Server Error - Ingestion failed"},
    },
)
async def ingest_documents(
    request: IngestDocumentsRequest,
    background_tasks: BackgroundTasks,
):
    """
    Ingest multiple documents into the RAG system.
    
    Returns immediately with a track_id for monitoring processing status.
    The actual processing happens in the background.
    """
    if not rag:
        raise HTTPException(status_code=503, detail="LightRAG instance not initialized")
    
    try:
        # Validate file_paths length if provided
        if request.file_paths is not None:
            if len(request.file_paths) != len(request.texts):
                raise HTTPException(
                    status_code=400,
                    detail="Number of file_paths must match the number of texts",
                )
        
        # Check if any file_paths already exist in doc_status storage
        if request.file_paths:
            for file_path in request.file_paths:
                if (
                    file_path
                    and file_path.strip()
                    and file_path != "unknown_source"
                ):
                    existing_doc_data = await rag.doc_status.get_doc_by_file_path(
                        file_path
                    )
                    if existing_doc_data:
                        status = existing_doc_data.get("status", "unknown")
                        existing_track_id = existing_doc_data.get("track_id") or ""
                        return IngestResponse(
                            status="duplicated",
                            message=f"File path '{file_path}' already exists in document storage (Status: {status}).",
                            track_id=existing_track_id,
                        )
        
        # Check if any content already exists by computing content hash (doc_id)
        for text in request.texts:
            sanitized_text = sanitize_text_for_encoding(text)
            content_doc_id = compute_mdhash_id(sanitized_text, prefix="doc-")
            existing_doc = await rag.doc_status.get_by_id(content_doc_id)
            if existing_doc:
                # Content already exists, return duplicated with existing track_id
                status = existing_doc.get("status", "unknown")
                existing_track_id = existing_doc.get("track_id") or ""
                return IngestResponse(
                    status="duplicated",
                    message=f"Identical content already exists in document storage (doc_id: {content_doc_id}, Status: {status}).",
                    track_id=existing_track_id,
                )
        
        # Generate track_id for documents ingestion
        track_id = generate_track_id("insert")
        
        # Add background task for processing
        background_tasks.add_task(
            pipeline_index_texts,
            rag,
            request.texts,
            file_paths=request.file_paths,
            track_id=track_id,
        )
        
        return IngestResponse(
            status="success",
            message="Documents successfully received. Processing will continue in background.",
            track_id=track_id,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing batch document ingestion: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "LightRAG Query API",
        "version": "1.0.0",
        "description": "FastAPI server for LightRAG query endpoints",
        "endpoints": {
            "/query": "Non-streaming RAG query",
            "/query/stream": "Streaming RAG query",
            "/query/data": "Data-only RAG query (no LLM generation)",
            "/ingest": "Ingest a single document",
            "/ingest/batch": "Ingest multiple documents",
            "/health": "Health check",
            "/docs": "API documentation",
        },
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "lightrag_fastapi:app",
        host=host,
        port=port,
        reload=True,
    )
