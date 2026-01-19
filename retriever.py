"""
Retriever Agent - Queries the LightRAG knowledge graph and applies Reciprocal Rank Fusion.

This agent is responsible for:
- Receiving subqueries and retrieving context from the knowledge graph
- Applying Reciprocal Rank Fusion (RRF) to merge results from multiple subqueries
- Storing the final context in Redis
- Returning the top-k fused results
"""

import asyncio
import json
import hashlib
from dataclasses import dataclass, field
from typing import Any

import redis.asyncio as redis

from base_agent import BaseAgent
from observation import Observation

# Add light_rag to the path for imports
import sys
sys.path.insert(0, './light_rag')

from lightrag import LightRAG, QueryParam


@dataclass
class RetrievedContext:
    """Represents a retrieved context item with its metadata."""
    content: str
    source_query: str
    entity_name: str | None = None
    entity_type: str | None = None
    description: str | None = None
    source_id: str | None = None
    file_path: str | None = None
    reference_id: str | None = None
    context_type: str = "chunk"  # "entity", "relationship", or "chunk"
    original_rank: int = 0  # rank within its source query results
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "source_query": self.source_query,
            "entity_name": self.entity_name,
            "entity_type": self.entity_type,
            "description": self.description,
            "source_id": self.source_id,
            "file_path": self.file_path,
            "reference_id": self.reference_id,
            "context_type": self.context_type,
            "original_rank": self.original_rank,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievedContext":
        """Create from dictionary."""
        return cls(**data)
    
    def get_unique_id(self) -> str:
        """Generate a unique identifier for deduplication."""
        # Use content hash as the primary identifier
        content_hash = hashlib.md5(self.content.encode()).hexdigest()
        return f"{self.context_type}_{content_hash}"


@dataclass
class RetrieverResult:
    """Result from the retriever agent containing fused contexts."""
    contexts: list[RetrievedContext] = field(default_factory=list)
    rrf_scores: dict[str, float] = field(default_factory=dict)  # context_id -> RRF score
    original_query: str = ""
    subqueries: list[str] = field(default_factory=list)
    step: int = 1
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "contexts": [ctx.to_dict() for ctx in self.contexts],
            "rrf_scores": self.rrf_scores,
            "original_query": self.original_query,
            "subqueries": self.subqueries,
            "step": self.step,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrieverResult":
        """Create from dictionary."""
        return cls(
            contexts=[RetrievedContext.from_dict(ctx) for ctx in data.get("contexts", [])],
            rrf_scores=data.get("rrf_scores", {}),
            original_query=data.get("original_query", ""),
            subqueries=data.get("subqueries", []),
            step=data.get("step", 1),
        )


class RetrieverAgent(BaseAgent):
    """
    Agent responsible for retrieving context from LightRAG knowledge graph.
    
    This agent:
    1. Receives subqueries and retrieves context for each
    2. Applies Reciprocal Rank Fusion (RRF) to merge results
    3. Returns top-k fused results
    4. Stores final context in Redis
    """
    
    # RRF constant (commonly set to 60)
    RRF_K = 60
    
    def __init__(
        self,
        name: str = "RetrieverAgent",
        lightrag: LightRAG | None = None,
        redis_client: redis.Redis | None = None,
        top_k: int = 10,
        query_mode: str = "mix",
        redis_key_prefix: str = "retriever:context:",
        redis_ttl: int = 3600,  # 1 hour default TTL
    ):
        """
        Initialize the RetrieverAgent.
        
        Args:
            name: Agent name
            lightrag: LightRAG instance for querying the knowledge graph
            redis_client: Redis client for storing/retrieving context
            top_k: Number of top results to return after RRF
            query_mode: LightRAG query mode ("local", "global", "hybrid", "mix", "naive")
            redis_key_prefix: Prefix for Redis keys
            redis_ttl: TTL for Redis keys in seconds
        """
        super().__init__(name)
        self.lightrag = lightrag
        self.redis_client = redis_client
        self.top_k = top_k
        self.query_mode = query_mode
        self.redis_key_prefix = redis_key_prefix
        self.redis_ttl = redis_ttl
    
    async def act(self, observation: Observation) -> RetrieverResult:
        """
        Execute the retrieval action based on the observation.
        
        Args:
            observation: Contains the queries and step information
            
        Returns:
            RetrieverResult with fused contexts
        """
        subqueries = observation.subqueries
        original_query = observation.original_query
        step = observation.step
        
        if step == 1:
            return await self._first_step_retrieval(
                subqueries=subqueries,
                original_query=original_query,
            )
        else:
            # TODO: Implement subsequent step retrieval
            # This will involve:
            # - Loading old context from Redis
            # - Merging with new retrieved context
            # - Re-ranking and fusing all contexts
            return await self._subsequent_step_retrieval(
                subqueries=subqueries,
                original_query=original_query,
                step=step,
            )
    
    async def think(self, data: Any) -> dict[str, Any]:
        """
        The retriever agent doesn't need to think - it just retrieves.
        This method is included for interface compatibility.
        
        Args:
            data: Input data (unused)
            
        Returns:
            Empty dict as no thinking is needed
        """
        return {}
    
    async def _first_step_retrieval(
        self,
        subqueries: list[str],
        original_query: str,
    ) -> RetrieverResult:
        """
        Perform first step retrieval:
        1. Query LightRAG for each subquery
        2. Apply RRF to fuse results
        3. Store in Redis
        4. Return top-k results
        
        Args:
            subqueries: List of decomposed subqueries
            original_query: The original user query
            
        Returns:
            RetrieverResult with fused contexts
        """
        if not self.lightrag:
            raise ValueError("LightRAG instance is required for retrieval")
        
        # Retrieve contexts for all subqueries in parallel
        retrieval_tasks = [
            self._retrieve_for_query(query) for query in subqueries
        ]
        query_results = await asyncio.gather(*retrieval_tasks)
        
        # Build ranked lists for RRF
        # Each query produces a ranked list of contexts
        ranked_lists: list[list[RetrievedContext]] = []
        for query, contexts in zip(subqueries, query_results):
            # Assign ranks based on position
            for rank, ctx in enumerate(contexts):
                ctx.original_rank = rank
                ctx.source_query = query
            ranked_lists.append(contexts)
        
        # Apply Reciprocal Rank Fusion
        fused_contexts, rrf_scores = self._reciprocal_rank_fusion(ranked_lists)
        
        # Take top-k results
        top_contexts = fused_contexts[:self.top_k]
        top_scores = {
            ctx.get_unique_id(): rrf_scores[ctx.get_unique_id()]
            for ctx in top_contexts
        }
        
        result = RetrieverResult(
            contexts=top_contexts,
            rrf_scores=top_scores,
            original_query=original_query,
            subqueries=subqueries,
            step=1,
        )
        
        # Store in Redis
        await self._store_in_redis(original_query, result)
        
        return result
    
    async def _subsequent_step_retrieval(
        self,
        subqueries: list[str],
        original_query: str,
        step: int,
    ) -> RetrieverResult:
        """
        Perform subsequent step retrieval (step > 1).
        
        TODO: NOT IMPLEMENTED YET
        This will involve:
        - Loading old context from Redis
        - Retrieving new context for new subqueries
        - Merging old and new contexts
        - Re-ranking and fusing all contexts
        - Storing updated context in Redis
        
        Args:
            subqueries: New subqueries for this step
            original_query: The original user query
            step: Current step number
            
        Returns:
            RetrieverResult with fused contexts
        """
        # TODO: Implement this in the future
        # For now, raise NotImplementedError
        raise NotImplementedError(
            f"Subsequent step retrieval (step={step}) is not implemented yet. "
            "This will be implemented to handle loading old context from Redis, "
            "merging with new context, and re-ranking."
        )
    
    async def _retrieve_for_query(self, query: str) -> list[RetrievedContext]:
        """
        Retrieve context from LightRAG for a single query.
        
        Args:
            query: The query string
            
        Returns:
            List of RetrievedContext items
        """
        if not self.lightrag:
            return []
        
        # Create query parameters - only get context, no LLM generation
        param = QueryParam(
            mode=self.query_mode,
            only_need_context=True,
            only_need_prompt=False,
            top_k=self.top_k * 2,  # Retrieve more to have enough after fusion
        )
        
        # Query LightRAG - this returns only context, not LLM answer
        result = await self.lightrag.aquery_data(query, param)
        
        # Parse the result into RetrievedContext objects
        contexts = self._parse_lightrag_result(result, query)
        
        return contexts
    
    def _parse_lightrag_result(
        self,
        result: dict[str, Any],
        source_query: str,
    ) -> list[RetrievedContext]:
        """
        Parse LightRAG query result into RetrievedContext objects.
        
        Args:
            result: Raw result from LightRAG aquery_data
            source_query: The query that produced this result
            
        Returns:
            List of RetrievedContext items
        """
        contexts: list[RetrievedContext] = []
        
        if result.get("status") != "success":
            return contexts
        
        data = result.get("data", {})
        
        # Parse entities
        for idx, entity in enumerate(data.get("entities", [])):
            ctx = RetrievedContext(
                content=entity.get("description", ""),
                source_query=source_query,
                entity_name=entity.get("entity_name"),
                entity_type=entity.get("entity_type"),
                description=entity.get("description"),
                source_id=entity.get("source_id"),
                file_path=entity.get("file_path"),
                reference_id=entity.get("reference_id"),
                context_type="entity",
                original_rank=idx,
            )
            if ctx.content:  # Only add if there's actual content
                contexts.append(ctx)
        
        # Parse relationships
        for idx, rel in enumerate(data.get("relationships", [])):
            # Construct content from relationship
            content = rel.get("description", "")
            if not content:
                src = rel.get("src_id", "")
                tgt = rel.get("tgt_id", "")
                keywords = rel.get("keywords", "")
                content = f"{src} -> {tgt}: {keywords}"
            
            ctx = RetrievedContext(
                content=content,
                source_query=source_query,
                description=rel.get("description"),
                source_id=rel.get("source_id"),
                file_path=rel.get("file_path"),
                reference_id=rel.get("reference_id"),
                context_type="relationship",
                original_rank=idx,
            )
            if ctx.content:
                contexts.append(ctx)
        
        # Parse chunks
        for idx, chunk in enumerate(data.get("chunks", [])):
            ctx = RetrievedContext(
                content=chunk.get("content", ""),
                source_query=source_query,
                file_path=chunk.get("file_path"),
                reference_id=chunk.get("reference_id"),
                source_id=chunk.get("chunk_id"),
                context_type="chunk",
                original_rank=idx,
            )
            if ctx.content:
                contexts.append(ctx)
        
        return contexts
    
    def _reciprocal_rank_fusion(
        self,
        ranked_lists: list[list[RetrievedContext]],
    ) -> tuple[list[RetrievedContext], dict[str, float]]:
        """
        Apply Reciprocal Rank Fusion to merge multiple ranked lists.
        
        RRF Score = sum(1 / (k + rank_i)) for each list where item appears
        
        Args:
            ranked_lists: List of ranked context lists from different queries
            
        Returns:
            Tuple of (fused sorted list, dict of context_id -> RRF score)
        """
        rrf_scores: dict[str, float] = {}
        context_map: dict[str, RetrievedContext] = {}
        
        for ranked_list in ranked_lists:
            for rank, ctx in enumerate(ranked_list):
                ctx_id = ctx.get_unique_id()
                
                # Calculate RRF contribution from this list
                rrf_contribution = 1.0 / (self.RRF_K + rank + 1)  # +1 because rank is 0-indexed
                
                if ctx_id in rrf_scores:
                    rrf_scores[ctx_id] += rrf_contribution
                else:
                    rrf_scores[ctx_id] = rrf_contribution
                    context_map[ctx_id] = ctx
        
        # Sort by RRF score (descending)
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Build sorted context list
        fused_contexts = [context_map[ctx_id] for ctx_id in sorted_ids]
        
        return fused_contexts, rrf_scores
    
    async def _store_in_redis(
        self,
        original_query: str,
        result: RetrieverResult,
    ) -> None:
        """
        Store the retrieval result in Redis.
        
        Args:
            original_query: The original user query (used for key generation)
            result: The RetrieverResult to store
        """
        if not self.redis_client:
            return
        
        # Generate a key based on the original query
        query_hash = hashlib.md5(original_query.encode()).hexdigest()
        key = f"{self.redis_key_prefix}{query_hash}"
        
        # Serialize and store
        data = json.dumps(result.to_dict())
        await self.redis_client.setex(key, self.redis_ttl, data)
    
    async def _load_from_redis(self, original_query: str) -> RetrieverResult | None:
        """
        Load previous retrieval result from Redis.
        
        Args:
            original_query: The original user query
            
        Returns:
            RetrieverResult if found, None otherwise
        """
        if not self.redis_client:
            return None
        
        query_hash = hashlib.md5(original_query.encode()).hexdigest()
        key = f"{self.redis_key_prefix}{query_hash}"
        
        data = await self.redis_client.get(key)
        if data:
            return RetrieverResult.from_dict(json.loads(data))
        
        return None
    
    def get_context_as_string(self, result: RetrieverResult) -> str:
        """
        Convert the retrieval result to a formatted string for LLM consumption.
        
        Args:
            result: The RetrieverResult
            
        Returns:
            Formatted string with all contexts
        """
        if not result.contexts:
            return "No relevant context found."
        
        sections = []
        
        # Group by context type
        entities = [c for c in result.contexts if c.context_type == "entity"]
        relationships = [c for c in result.contexts if c.context_type == "relationship"]
        chunks = [c for c in result.contexts if c.context_type == "chunk"]
        
        if entities:
            entity_strs = []
            for e in entities:
                entity_str = f"- {e.entity_name}"
                if e.entity_type:
                    entity_str += f" ({e.entity_type})"
                if e.description:
                    entity_str += f": {e.description}"
                entity_strs.append(entity_str)
            sections.append("**Entities:**\n" + "\n".join(entity_strs))
        
        if relationships:
            rel_strs = [f"- {r.content}" for r in relationships]
            sections.append("**Relationships:**\n" + "\n".join(rel_strs))
        
        if chunks:
            chunk_strs = []
            for idx, c in enumerate(chunks, 1):
                chunk_str = f"[{idx}] {c.content}"
                if c.file_path:
                    chunk_str += f"\n   Source: {c.file_path}"
                chunk_strs.append(chunk_str)
            sections.append("**Document Chunks:**\n" + "\n\n".join(chunk_strs))
        
        return "\n\n".join(sections)
