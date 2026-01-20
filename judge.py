import json
from typing import List, Optional

import httpx
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

from base_agent import BaseAgent
from prompts import IDENTIFY_MISSING_CONTEXT_PROMPT, GENERATE_SUBQUERIES_PROMPT


class MissingContext(BaseModel):
    """Schema for identifying missing context."""
    missing_information: List[str] = Field(
        ..., 
        description="List of specific pieces of information that are missing to fully answer the query"
    )
    gaps_analysis: str = Field(
        ..., 
        description="Analysis of why the current context is insufficient"
    )
    suggested_focus_areas: List[str] = Field(
        ..., 
        description="Areas or topics that should be explored to fill the gaps"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence in the gap analysis (0-1)"
    )


class SubQuery(BaseModel):
    """Schema for a single sub-query."""
    id: int = Field(..., description="Sequential ID for the sub-query")
    question: str = Field(..., description="The sub-query question")
    canonical_form: str = Field(..., description="Normalized form suitable for retrieval")
    rationale: str = Field(..., description="Why this sub-query addresses a gap")
    priority: int = Field(..., ge=1, le=5, description="Priority 1-5 (1 = highest)")
    keywords: List[str] = Field(..., description="Keywords for retrieval")


class GeneratedSubQueries(BaseModel):
    """Schema for generated sub-queries from missing context."""
    reasoning: str = Field(
        ..., 
        description="Overall reasoning for the generated sub-queries"
    )
    subqueries: List[SubQuery] = Field(
        ..., 
        description="List of generated sub-queries to fill context gaps"
    )
    


class JudgementResult(BaseModel):
    """Schema for the final judgement result."""
    converged: bool = Field(..., description="Whether the retrieval has converged")
    similarity_score: float = Field(..., description="ColBERT similarity score between original and inferred query")
    threshold: float = Field(..., description="The threshold used for convergence check")
    missing_context: Optional[MissingContext] = Field(
        None, 
        description="Identified missing context if not converged"
    )
    new_subqueries: Optional[GeneratedSubQueries] = Field(
        None, 
        description="Generated sub-queries if not converged"
    )


class JudgeAgent(BaseAgent):
    """Agent that judges whether retrieval has converged and generates new sub-queries if needed.

    This agent performs:
    - Step 6: Compute similarity between original and inferred query using ColBERT
    - Step 7: If similarity <= threshold, identify missing context using LLM
    - Step 8: Generate new sub-queries from missing context using LLM
    """

    def __init__(
        self,
        name: str,
        base_url: str = "http://localhost:8001/v1",
        api_key: str = "EMPTY",
        model: str = "meta-llama/Llama-2-7b-chat-hf",
        colbert_base_url: str = "http://localhost:8002",
        similarity_threshold: float = 0.75,
    ):
        super().__init__(name)
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = model
        self.colbert_base_url = colbert_base_url.rstrip("/")
        self.similarity_threshold = similarity_threshold
        self._http_client = None

    @property
    def http_client(self) -> httpx.AsyncClient:
        """Lazy initialize async HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=60.0)
        return self._http_client

    async def _compute_colbert_similarity(self, query1: str, query2: str) -> float:
        """Compute ColBERT MaxSim similarity between two queries via the ColBERT service.

        Args:
            query1: The original user query.
            query2: The inferred query from context.

        Returns:
            float: Similarity score between 0 and 1.
        """
        response = await self.http_client.post(
            f"{self.colbert_base_url}/v1/similarity",
            json={"query": query1, "document": query2}
        )
        response.raise_for_status()
        result = response.json()
        return result["similarity"]

    async def _identify_missing_context(
        self, 
        original_query: str, 
        inferred_query: str, 
        inferred_answer: str, 
        context: str
    ) -> MissingContext:
        """Identify what context is missing to fully answer the original query.

        Args:
            original_query: The original user query.
            inferred_query: The query inferred from context.
            inferred_answer: The answer inferred from context.
            context: The retrieved knowledge graph context.

        Returns:
            MissingContext: Structured analysis of missing information.
        """
        user_content = f"""Original Query: {original_query}

Inferred Query (what the context actually addresses): {inferred_query}

Inferred Answer (what the context actually answers): {inferred_answer}

Retrieved Context:
{context}"""

        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": IDENTIFY_MISSING_CONTEXT_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0,
            extra_body={
                "guided_json": MissingContext.model_json_schema()
            }
        )

        response_content = completion.choices[0].message.content
        parsed_response = json.loads(response_content)
        return MissingContext(**parsed_response)

    async def _generate_subqueries(
        self, 
        original_query: str, 
        inferred_query: str, 
        inferred_answer: str, 
        context: str,
        missing_context: MissingContext
    ) -> GeneratedSubQueries:
        """Generate new sub-queries to fill the gaps in retrieved context.

        Args:
            original_query: The original user query.
            inferred_query: The query inferred from context.
            inferred_answer: The answer inferred from context.
            context: The retrieved knowledge graph context.
            missing_context: The identified missing context.

        Returns:
            GeneratedSubQueries: New sub-queries to retrieve missing information.
        """
        user_content = f"""Original Query: {original_query}

Inferred Query (what the context actually addresses): {inferred_query}

Inferred Answer (what the context actually answers): {inferred_answer}

Retrieved Context:
{context}

Missing Information Analysis:
- Gaps: {missing_context.gaps_analysis}
- Missing pieces: {', '.join(missing_context.missing_information)}
- Suggested focus areas: {', '.join(missing_context.suggested_focus_areas)}"""

        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": GENERATE_SUBQUERIES_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0,
            extra_body={
                "guided_json": GeneratedSubQueries.model_json_schema()
            }
        )

        response_content = completion.choices[0].message.content
        parsed_response = json.loads(response_content)
        return GeneratedSubQueries(**parsed_response)

    async def judge(
        self,
        original_query: str,
        inferred_query: str,
        inferred_answer: str,
        context: str,
        threshold: Optional[float] = None
    ) -> JudgementResult:
        """Judge whether the retrieval has converged and generate new sub-queries if needed.

        Args:
            original_query: The original user query.
            inferred_query: The query inferred from context by the DeducerAgent.
            inferred_answer: The answer inferred from context by the DeducerAgent.
            context: The retrieved knowledge graph context.
            threshold: Optional override for similarity threshold.

        Returns:
            JudgementResult: The judgement including convergence status and new sub-queries if needed.
        """
        threshold = threshold or self.similarity_threshold

        # Step 6: Compute ColBERT similarity
        similarity_score = await self._compute_colbert_similarity(original_query, inferred_query)

        # Check convergence
        if similarity_score > threshold:
            return JudgementResult(
                converged=True,
                similarity_score=similarity_score,
                threshold=threshold,
                missing_context=None,
                new_subqueries=None
            )

        # Step 7: Identify missing context
        missing_context = await self._identify_missing_context(
            original_query, inferred_query, inferred_answer, context
        )

        # Step 8: Generate new sub-queries based on missing context
        new_subqueries = await self._generate_subqueries(
            original_query, inferred_query, inferred_answer, context, missing_context
        )

        return JudgementResult(
            converged=False,
            similarity_score=similarity_score,
            threshold=threshold,
            missing_context=missing_context,
            new_subqueries=new_subqueries
        )

    async def act(self, observation: dict) -> dict:
        """Process an observation and perform judgement.

        Args:
            observation (dict): A dictionary containing:
                - original_query: The original user query
                - inferred_query: The query inferred from context
                - inferred_answer: The answer inferred from context
                - context: The retrieved knowledge graph context
                - threshold (optional): Override for similarity threshold

        Returns:
            dict: The judgement result as a dictionary.
        """
        original_query = observation.get("original_query", "")
        inferred_query = observation.get("inferred_query", "")
        inferred_answer = observation.get("inferred_answer", "")
        context = observation.get("context", "")
        threshold = observation.get("threshold")

        if not all([original_query, inferred_query, context]):
            return {"error": "Missing required fields: original_query, inferred_query, context"}

        result = await self.judge(
            original_query=original_query,
            inferred_query=inferred_query,
            inferred_answer=inferred_answer,
            context=context,
            threshold=threshold
        )

        return result.model_dump()

    async def think(self, data: dict) -> dict:
        """Analyze the judgement results and provide insights.

        Args:
            data (dict): Data containing judgement results.

        Returns:
            dict: Analysis of the judgement.
        """
        converged = data.get("converged", False)
        similarity_score = data.get("similarity_score", 0.0)
        threshold = data.get("threshold", self.similarity_threshold)
        
        analysis = {
            "converged": converged,
            "similarity_score": similarity_score,
            "threshold": threshold,
            "similarity_gap": threshold - similarity_score if not converged else 0,
            "convergence_status": "Retrieval converged" if converged else "More context needed"
        }

        if not converged:
            missing_context = data.get("missing_context", {})
            new_subqueries = data.get("new_subqueries", {})
            
            analysis["num_missing_pieces"] = len(missing_context.get("missing_information", []))
            analysis["num_new_subqueries"] = len(new_subqueries.get("subqueries", []))
            analysis["focus_areas"] = missing_context.get("suggested_focus_areas", [])
            analysis["next_queries"] = [
                sq.get("question", "") 
                for sq in new_subqueries.get("subqueries", [])
            ]

        return analysis
