"""
Retriever Agent - Queries the LightRAG knowledge graph and applies Reciprocal Rank Fusion.

This agent is responsible for:
- Receiving subqueries and retrieving context from the knowledge graph
- Applying Reciprocal Rank Fusion (RRF) to merge results from multiple subqueries
- Storing the final context in memory
- Returning the top-k fused results
"""

import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Any

import httpx

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
    4. Stores final context in memory
    """

    # RRF constant (commonly set to 60)
    RRF_K = 60

    def __init__(
        self,
        name: str = "RetrieverAgent",
        lightrag: LightRAG | None = None,
        top_k: int = 10,
        query_mode: str = "mix",
        colbert_base_url: str = "http://localhost:8002",
    ):
        """
        Initialize the RetrieverAgent.

        Args:
            name: Agent name
            lightrag: LightRAG instance for querying the knowledge graph
            top_k: Number of top results to return after RRF
            query_mode: LightRAG query mode ("local", "global", "hybrid", "mix", "naive")
            colbert_base_url: Base URL for the ColBERT service
        """
        super().__init__(name)
        self.lightrag = lightrag
        self.top_k = top_k
        self.query_mode = query_mode
        self.colbert_base_url = colbert_base_url.rstrip("/")
        self._http_client = None

        # In-memory storage for retrieval results (keyed by query hash)
        self._memory: dict[str, RetrieverResult] = {}

    @property
    def http_client(self) -> httpx.AsyncClient:
        """Lazy initialize async HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=60.0)
        return self._http_client
    
    async def act(
        self,
        observation: Observation,
        mode: str | None = 'mix',
    ) -> RetrieverResult:
        """
        Execute the retrieval action based on the observation.

        Args:
            observation: Contains the queries and step information
            mode: LightRAG query mode ("local", "global", "hybrid", "mix", "naive").
                  If None, uses self.query_mode (default from constructor).

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
                mode=mode,
            )
        else:
            return await self._subsequent_step_retrieval(
                subqueries=subqueries,
                original_query=original_query,
                step=step,
                mode=mode,
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
        mode: str | None = None,
    ) -> RetrieverResult:
        """
        Perform first step retrieval:
        1. Query LightRAG for each subquery
        2. Rerank each subquery's results using ColBERT similarity
        3. Apply RRF to fuse the reranked results
        4. Store in memory
        5. Return top-k results

        Args:
            subqueries: List of decomposed subqueries
            original_query: The original user query
            mode: LightRAG query mode. If None, uses self.query_mode.

        Returns:
            RetrieverResult with fused contexts
        """
        if not self.lightrag:
            raise ValueError("LightRAG instance is required for retrieval")

        # Retrieve contexts for all subqueries in parallel
        retrieval_tasks = [
            self._retrieve_for_query(query, mode=mode) for query in subqueries
        ]
        query_results = await asyncio.gather(*retrieval_tasks)

        # Build query-context pairs for reranking
        query_context_pairs: list[tuple[str, list[RetrievedContext]]] = []
        for query, contexts in zip(subqueries, query_results):
            # Assign source query to each context
            for rank, ctx in enumerate(contexts):
                ctx.original_rank = rank
                ctx.source_query = query
            query_context_pairs.append((query, contexts))

        # Rerank with ColBERT, then apply RRF fusion
        fused_contexts, rrf_scores = await self._rerank_and_fuse(query_context_pairs)
        
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
        self._store_in_memory(original_query, result)
        
        return result
    
    async def _subsequent_step_retrieval(
        self,
        subqueries: list[str],
        original_query: str,
        step: int,
        mode: str | None = None,
    ) -> RetrieverResult:
        """
        Perform subsequent step retrieval (step > 1).

        This method:
        1. Loads previous context from memory
        2. Retrieves new context for new subqueries
        3. Merges old and new contexts
        4. Reranks each subquery's contexts using ColBERT similarity
        5. Applies RRF on the reranked combined set
        6. Keeps only top-k results
        7. Stores updated context in memory

        Args:
            subqueries: New subqueries for this step
            original_query: The original user query
            step: Current step number
            mode: LightRAG query mode. If None, uses self.query_mode.

        Returns:
            RetrieverResult with fused contexts
        """
        if not self.lightrag:
            raise ValueError("LightRAG instance is required for retrieval")

        # Step 1: Load previous context from memory
        previous_result = self._load_from_memory(original_query)

        # Step 2: Retrieve new contexts for all new subqueries in parallel
        retrieval_tasks = [
            self._retrieve_for_query(query, mode=mode) for query in subqueries
        ]
        query_results = await asyncio.gather(*retrieval_tasks)

        # Step 3: Build query-context pairs for reranking
        query_context_pairs: list[tuple[str, list[RetrievedContext]]] = []
        all_subqueries = list(subqueries)  # Track all subqueries used

        for query, contexts in zip(subqueries, query_results):
            # Assign source query to each context
            for rank, ctx in enumerate(contexts):
                ctx.original_rank = rank
                ctx.source_query = query
            query_context_pairs.append((query, contexts))

        # Step 4: Include previous contexts as additional query-context pairs
        # This allows RRF to accumulate scores for contexts appearing across steps
        if previous_result and previous_result.contexts:
            # Group previous contexts by their source query to preserve ranking structure
            previous_by_query: dict[str, list[RetrievedContext]] = {}
            for ctx in previous_result.contexts:
                source = ctx.source_query or "previous_step"
                if source not in previous_by_query:
                    previous_by_query[source] = []
                previous_by_query[source].append(ctx)

            # Add each group as a separate query-context pair for reranking
            for source_query, contexts in previous_by_query.items():
                # Sort by original rank to preserve ordering
                contexts.sort(key=lambda x: x.original_rank)
                query_context_pairs.append((source_query, contexts))

            # Merge previous subqueries
            all_subqueries.extend(previous_result.subqueries)

        # Step 5: Rerank with ColBERT, then apply RRF fusion
        fused_contexts, rrf_scores = await self._rerank_and_fuse(query_context_pairs)

        # Step 6: Take top-k results
        top_contexts = fused_contexts[:self.top_k]
        top_scores = {
            ctx.get_unique_id(): rrf_scores[ctx.get_unique_id()]
            for ctx in top_contexts
        }

        # Step 7: Create result with accumulated subqueries
        # Remove duplicate subqueries while preserving order
        seen_queries = set()
        unique_subqueries = []
        for q in all_subqueries:
            if q not in seen_queries:
                seen_queries.add(q)
                unique_subqueries.append(q)

        result = RetrieverResult(
            contexts=top_contexts,
            rrf_scores=top_scores,
            original_query=original_query,
            subqueries=unique_subqueries,
            step=step,
        )

        # Step 8: Store updated result in Redis
        self._store_in_memory(original_query, result)

        return result
    
    async def _retrieve_for_query(
        self,
        query: str,
        mode: str | None = None,
    ) -> list[RetrievedContext]:
        """
        Retrieve context from LightRAG for a single query.

        Args:
            query: The query string
            mode: LightRAG query mode ("local", "global", "hybrid", "mix", "naive").
                  If None, uses self.query_mode (default from constructor).

        Returns:
            List of RetrievedContext items
        """
        if not self.lightrag:
            return []

        # Use provided mode or fall back to instance default
        effective_mode = mode if mode is not None else self.query_mode

        # Create query parameters - only get context, no LLM generation
        param = QueryParam(
            mode=effective_mode,
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

    async def _compute_colbert_similarity(self, query: str, document: str) -> float:
        """
        Compute ColBERT MaxSim similarity between query and document via the ColBERT service.

        Args:
            query: The query text
            document: The document text

        Returns:
            Similarity score between 0 and 1
        """
        response = await self.http_client.post(
            f"{self.colbert_base_url}/v1/similarity",
            json={"query": query, "document": document}
        )
        response.raise_for_status()
        result = response.json()
        return result["similarity"]

    async def _rerank_with_colbert(
        self,
        query: str,
        contexts: list[RetrievedContext],
    ) -> list[RetrievedContext]:
        """
        Rerank contexts using ColBERT similarity scores.

        Args:
            query: The sub-query to compute similarity against
            contexts: List of contexts to rerank

        Returns:
            Contexts sorted by ColBERT similarity (descending)
        """
        if not contexts:
            return []

        # Compute similarity scores in parallel
        similarity_tasks = [
            self._compute_colbert_similarity(query, ctx.content)
            for ctx in contexts
        ]
        scores = await asyncio.gather(*similarity_tasks)

        # Pair contexts with scores
        scored_contexts = list(zip(contexts, scores))

        # Sort by score descending
        scored_contexts.sort(key=lambda x: x[1], reverse=True)

        # Return reranked contexts
        return [ctx for ctx, _ in scored_contexts]

    async def _rerank_and_fuse(
        self,
        query_context_pairs: list[tuple[str, list[RetrievedContext]]],
    ) -> tuple[list[RetrievedContext], dict[str, float]]:
        """
        Rerank each sub-query's contexts with ColBERT, then apply RRF fusion.

        This method:
        1. For each (sub-query, contexts) pair, rerank contexts using ColBERT similarity
        2. Apply Reciprocal Rank Fusion on the reranked lists

        Args:
            query_context_pairs: List of (sub-query, contexts) tuples

        Returns:
            Tuple of (fused sorted list, dict of context_id -> RRF score)
        """
        # Phase 1: Rerank each sub-query's contexts using ColBERT
        rerank_tasks = [
            self._rerank_with_colbert(query, contexts)
            for query, contexts in query_context_pairs
        ]
        reranked_lists = await asyncio.gather(*rerank_tasks)

        # Phase 2: Apply RRF on the reranked lists
        rrf_scores: dict[str, float] = {}
        context_map: dict[str, RetrievedContext] = {}

        for ranked_list in reranked_lists:
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
    
    def _store_in_memory(
        self,
        original_query: str,
        result: RetrieverResult,
    ) -> None:
        """
        Store the retrieval result in memory.

        Args:
            original_query: The original user query (used for key generation)
            result: The RetrieverResult to store
        """
        query_hash = hashlib.md5(original_query.encode()).hexdigest()
        self._memory[query_hash] = result

    def _load_from_memory(self, original_query: str) -> RetrieverResult | None:
        """
        Load previous retrieval result from memory.

        Args:
            original_query: The original user query

        Returns:
            RetrieverResult if found, None otherwise
        """
        query_hash = hashlib.md5(original_query.encode()).hexdigest()
        return self._memory.get(query_hash)

    def clear_memory(self) -> None:
        """Clear all stored retrieval results from memory."""
        self._memory.clear()
    
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

    async def get_global_context(self, query: str) -> str:
        """
        Retrieve global context for the original query before query rewriting.

        This method queries LightRAG in 'global' mode to get high-level knowledge
        that can inform the query rewriter agent about the domain and available
        information in the knowledge graph.

        Args:
            query: The original user query

        Returns:
            Formatted string with global context for the query rewriter
        """
        contexts = await self._retrieve_for_query(query, mode="global")

        if not contexts:
            return "No global context found."

        # Format the global context for the query rewriter
        sections = []

        # Group by context type
        entities = [c for c in contexts if c.context_type == "entity"]
        relationships = [c for c in contexts if c.context_type == "relationship"]

        if entities:
            entity_strs = []
            for e in entities:
                entity_str = f"- {e.entity_name}"
                if e.entity_type:
                    entity_str += f" ({e.entity_type})"
                if e.description:
                    entity_str += f": {e.description}"
                entity_strs.append(entity_str)
            sections.append("**Key Entities:**\n" + "\n".join(entity_strs))

        if relationships:
            rel_strs = [f"- {r.content}" for r in relationships]
            sections.append("**Key Relationships:**\n" + "\n".join(rel_strs))

        return "\n\n".join(sections) if sections else "No global context found."
