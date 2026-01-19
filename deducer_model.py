import asyncio
import json
from typing import List, Optional

from pydantic import BaseModel, Field
from openai import AsyncOpenAI

from base_agent import BaseAgent
from prompts import INFER_ANSWER_PROMPT, INFER_QUERY_PROMPT


class InferredAnswer(BaseModel):
    """Schema for the inferred answer from context."""
    answer: str = Field(..., description="The answer that can be derived from the context")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the answer (0-1)")
    supporting_evidence: List[str] = Field(..., description="Key pieces of evidence from the context supporting this answer")
    reasoning: str = Field(..., description="Brief explanation of how the answer was derived")


class InferredQuery(BaseModel):
    """Schema for the inferred original query from context."""
    query: str = Field(..., description="The most likely original query that this context addresses")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the inferred query (0-1)")
    query_type: str = Field(..., description="Type of query (e.g., 'factual', 'comparative', 'procedural', 'exploratory')")
    key_topics: List[str] = Field(..., description="Main topics/entities the query focuses on")
    reasoning: str = Field(..., description="Brief explanation of how the query was inferred")


class DeducerAgent(BaseAgent):
    """Agent that infers both the answer and original query from retrieved context.
    
    This agent performs two parallel LLM calls:
    - Step 4: Infer what answer could be found from the retrieved context
    - Step 5: Infer what was the original query that was addressed from the retrieved context
    """

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

    async def _infer_answer(self, context: str) -> InferredAnswer:
        """Infer what answer could be found from the retrieved context.

        Args:
            context (str): The retrieved knowledge graph context.

        Returns:
            InferredAnswer: A structured object containing the inferred answer.
        """
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": INFER_ANSWER_PROMPT},
                {"role": "user", "content": f"Context:\n{context}"}
            ],
            temperature=0,
            extra_body={
                "guided_json": InferredAnswer.model_json_schema()
            }
        )

        response_content = completion.choices[0].message.content
        parsed_response = json.loads(response_content)
        return InferredAnswer(**parsed_response)

    async def _infer_query(self, context: str) -> InferredQuery:
        """Infer what was the original query addressed from the retrieved context.

        Args:
            context (str): The retrieved knowledge graph context.

        Returns:
            InferredQuery: A structured object containing the inferred query.
        """
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": INFER_QUERY_PROMPT},
                {"role": "user", "content": f"Context:\n{context}"}
            ],
            temperature=0,
            extra_body={
                "guided_json": InferredQuery.model_json_schema()
            }
        )

        response_content = completion.choices[0].message.content
        parsed_response = json.loads(response_content)
        return InferredQuery(**parsed_response)

    async def deduce(self, context: str) -> dict:
        """Perform parallel inference of both answer and query from context.

        Args:
            context (str): The retrieved knowledge graph context.

        Returns:
            dict: A dictionary containing both the inferred answer and inferred query.
        """
        # Run both inferences in parallel since they don't interfere
        inferred_answer, inferred_query = await asyncio.gather(
            self._infer_answer(context),
            self._infer_query(context)
        )

        return {
            "inferred_answer": inferred_answer.model_dump(),
            "inferred_query": inferred_query.model_dump()
        }

    async def act(self, observation: dict) -> dict:
        """Process an observation containing context and deduce answer and query.

        Args:
            observation (dict): A dictionary containing the 'context' key.

        Returns:
            dict: A dictionary with 'inferred_answer' and 'inferred_query'.
        """
        context = observation.get("context", "")
        if not context:
            return {"error": "No context provided in observation"}

        return await self.deduce(context)

    async def think(self, data: dict) -> dict:
        """Analyze the deduction results and provide insights.

        Args:
            data (dict): Data containing deduction results.

        Returns:
            dict: Analysis of the deduction including confidence assessment.
        """
        inferred_answer = data.get("inferred_answer", {})
        inferred_query = data.get("inferred_query", {})

        answer_confidence = inferred_answer.get("confidence", 0.0)
        query_confidence = inferred_query.get("confidence", 0.0)

        # Calculate overall confidence
        overall_confidence = (answer_confidence + query_confidence) / 2

        return {
            "overall_confidence": overall_confidence,
            "answer_confidence": answer_confidence,
            "query_confidence": query_confidence,
            "is_high_confidence": overall_confidence >= 0.7,
            "inferred_query_text": inferred_query.get("query", ""),
            "inferred_answer_text": inferred_answer.get("answer", ""),
            "key_topics": inferred_query.get("key_topics", []),
            "supporting_evidence": inferred_answer.get("supporting_evidence", [])
        }
