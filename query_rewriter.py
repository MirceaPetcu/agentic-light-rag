import json
from re import A
from typing import List

from pydantic import BaseModel, Field
from openai import AsyncOpenAI

from base_agent import BaseAgent
from prompts import QUERY_DECOMPOSITION_PROMPT


class SubQuery(BaseModel):
    """Schema for a single decomposed sub-query."""
    id: int = Field(..., description="Sequential ID starting at 1")
    question: str = Field(..., description="A single, short, fact-focused question")
    canonical_form: str = Field(..., description="Normalized phrasing suitable for retrieval/QA")
    requires_retrieval: bool = Field(..., description="True if the question likely needs external sources")
    evidence_types: List[str] = Field(..., description="Types of evidence needed (e.g., 'official docs', 'statistical data')")
    rationale: str = Field(..., description="One-sentence explanation why this subquery is needed")
    priority: int = Field(..., ge=1, le=5, description="Priority 1-5 (1 = highest)")
    keywords: List[str] = Field(..., description="Terms useful for search/retrieval")


class QueryDecomposition(BaseModel):
    """Schema for the complete query decomposition response."""
    original_query: str = Field(..., description="The original user query")
    subqueries: List[SubQuery] = Field(..., description="List of decomposed sub-queries")


class QueryRewriterAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        base_url: str = "http://localhost:8001/v1",
        api_key: str = "EMPTY",
        model: str = "Qwen/Qwen2-7B-Instruct-AWQ"
    ):
        super().__init__(name)
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = model

    async def _decompose_query(self, query: str) -> QueryDecomposition:
        """Decompose a user query into structured sub-queries using vLLM with Outlines.

        Args:
            query (str): The original user query to decompose.

        Returns:
            QueryDecomposition: A structured object containing the original query
                and a list of decomposed sub-queries.
        """
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": QUERY_DECOMPOSITION_PROMPT},
                {"role": "user", "content": query}
            ],
            temperature=0,
            extra_body={
                "guided_json": QueryDecomposition.model_json_schema()
            }
        )

        response_content = completion.choices[0].message.content
        parsed_response = json.loads(response_content)
        return QueryDecomposition(**parsed_response)

    async def act(self, observation: dict) -> dict:
        """Process an observation and decompose the query.

        Args:
            observation (dict): A dictionary containing the 'query' key.

        Returns:
            dict: A dictionary with 'decomposition' containing the structured result.
        """
        query = observation.get("query", "")
        if not query:
            return {"error": "No query provided in observation"}
        
        decomposition = self._decompose_query(query)
        return {
            "decomposition": decomposition.model_dump(),
            "subqueries": [sq.model_dump() for sq in decomposition.subqueries]
        }

    async def think(self, data: dict) -> dict:
        """Analyze decomposed queries and prioritize them.

        Args:
            data (dict): Data containing decomposition results.

        Returns:
            dict: Prioritized sub-queries ready for retrieval.
        """
        subqueries = data.get("subqueries", [])
        
        # Sort by priority (1 = highest)
        prioritized = sorted(subqueries, key=lambda x: x.get("priority", 5))
        
        # Filter queries that require retrieval
        retrieval_queries = [sq for sq in prioritized if sq.get("requires_retrieval", True)]
        
        return {
            "prioritized_queries": prioritized,
            "retrieval_queries": retrieval_queries,
            "total_queries": len(subqueries),
            "retrieval_count": len(retrieval_queries)
        }