"""
Observation data class for multi-agent communication.

This module defines the Observation class that is used to pass data
between agents in the multi-agent LightRAG framework.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Observation:
    """
    Observation passed between agents during the retrieval loop.
    
    This class encapsulates all the information needed for agents
    to perform their tasks at each step of the iterative retrieval process.
    """
    
    # Original user query (preserved throughout the process)
    original_query: str = ""
    
    # Decomposed and/or contextualized subqueries for retrieval
    subqueries: list[str] = field(default_factory=list)
    
    # Current step in the iterative loop (starts at 1)
    step: int = 1
    
    # Maximum number of steps allowed
    max_steps: int = 5
    
    # Retrieved context from previous steps
    retrieved_context: list[dict[str, Any]] = field(default_factory=list)
    
    # Inferred answer from the retrieved context (Step 4)
    inferred_answer: str | None = None
    
    # Inferred query that the context addresses (Step 5)
    inferred_query: str | None = None
    
    # Cosine similarity between original query and inferred query (Step 6)
    similarity_score: float | None = None
    
    # Threshold for convergence
    similarity_threshold: float = 0.85
    
    # Whether convergence has been reached
    converged: bool = False
    
    # Missing context identified by LLM (Step 7)
    missing_context: str | None = None
    
    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def is_converged(self) -> bool:
        """Check if the retrieval process has converged."""
        if self.converged:
            return True
        if self.similarity_score is not None:
            return self.similarity_score >= self.similarity_threshold
        return False
    
    def should_continue(self) -> bool:
        """Check if the loop should continue."""
        return not self.is_converged() and self.step < self.max_steps
    
    def to_dict(self) -> dict[str, Any]:
        """Convert observation to dictionary."""
        return {
            "original_query": self.original_query,
            "subqueries": self.subqueries,
            "step": self.step,
            "max_steps": self.max_steps,
            "retrieved_context": self.retrieved_context,
            "inferred_answer": self.inferred_answer,
            "inferred_query": self.inferred_query,
            "similarity_score": self.similarity_score,
            "similarity_threshold": self.similarity_threshold,
            "converged": self.converged,
            "missing_context": self.missing_context,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Observation":
        """Create observation from dictionary."""
        return cls(
            original_query=data.get("original_query", ""),
            subqueries=data.get("subqueries", []),
            step=data.get("step", 1),
            max_steps=data.get("max_steps", 5),
            retrieved_context=data.get("retrieved_context", []),
            inferred_answer=data.get("inferred_answer"),
            inferred_query=data.get("inferred_query"),
            similarity_score=data.get("similarity_score"),
            similarity_threshold=data.get("similarity_threshold", 0.85),
            converged=data.get("converged", False),
            missing_context=data.get("missing_context"),
            metadata=data.get("metadata", {}),
        )
