"""
Response Agent - Generates comprehensive responses with citations using vLLM.

This agent is responsible for:
- Taking the full context (retrieved information, inferred answers, etc.)
- Generating a well-structured response to the user's query
- Citing all information used in the response
- Providing a references section with source details
"""

import json
from typing import List, Optional

from pydantic import BaseModel, Field
from openai import AsyncOpenAI

from base_agent import BaseAgent
from prompts import RESPONSE_GENERATION_PROMPT


class Citation(BaseModel):
    """Schema for a citation reference."""
    id: int = Field(..., description="Citation number (1, 2, 3, ...)")
    source_type: str = Field(..., description="Type of source (entity, relationship, chunk, evidence)")
    source_name: str = Field(..., description="Name/identifier of the source")
    excerpt: str = Field(..., description="Brief excerpt or summary from the source")
    file_path: Optional[str] = Field(None, description="File path if available")


class GeneratedResponse(BaseModel):
    """Schema for the generated response with citations."""
    response: str = Field(..., description="The full response with inline citations [1], [2], etc.")
    citations: List[Citation] = Field(..., description="List of citation references")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in the response completeness")
    limitations: List[str] = Field(..., description="Any limitations or gaps in the response")
    key_points: List[str] = Field(..., description="Main points covered in the response")


class ResponseAgent(BaseAgent):
    """Agent that generates comprehensive responses with citations using vLLM.
    
    This agent takes the full context including:
    - Original user query
    - Retrieved context from knowledge graph
    - Inferred answers and queries from the DeducerAgent
    - Supporting evidence and key topics
    
    And produces a well-structured response with proper citations.
    """

    def __init__(
        self,
        name: str,
        base_url: str = "http://localhost:8001/v1",
        api_key: str = "EMPTY",
        model: str = "Qwen/Qwen2-7B-Instruct-AWQ"
    ):
        """Initialize the ResponseAgent.
        
        Args:
            name: Name of the agent.
            base_url: Base URL for the vLLM API.
            api_key: API key for authentication.
            model: Model identifier to use for generation.
        """
        super().__init__(name)
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = model

    def _build_context_string(self, observation: dict) -> str:
        """Build a comprehensive context string from the observation.
        
        Args:
            observation: Dictionary containing all context information.
            
        Returns:
            Formatted context string for the LLM.
        """
        parts = []
        
        # Original query
        original_query = observation.get("original_query", "")
        if original_query:
            parts.append(f"## Original User Query\n{original_query}")
        
        # Retrieved contexts
        contexts = observation.get("contexts", [])
        if contexts:
            parts.append("\n## Retrieved Context")
            for i, ctx in enumerate(contexts, 1):
                ctx_parts = [f"\n### Context Item {i}"]
                
                if isinstance(ctx, dict):
                    if ctx.get("entity_name"):
                        ctx_parts.append(f"**Entity:** {ctx['entity_name']}")
                    if ctx.get("entity_type"):
                        ctx_parts.append(f"**Type:** {ctx['entity_type']}")
                    if ctx.get("context_type"):
                        ctx_parts.append(f"**Source Type:** {ctx['context_type']}")
                    if ctx.get("file_path"):
                        ctx_parts.append(f"**File:** {ctx['file_path']}")
                    if ctx.get("content"):
                        ctx_parts.append(f"**Content:**\n{ctx['content']}")
                    if ctx.get("description"):
                        ctx_parts.append(f"**Description:**\n{ctx['description']}")
                else:
                    ctx_parts.append(str(ctx))
                
                parts.append("\n".join(ctx_parts))
        
        # Inferred answer information
        inferred_answer = observation.get("inferred_answer", {})
        if inferred_answer:
            parts.append("\n## Inferred Answer Analysis")
            if inferred_answer.get("answer"):
                parts.append(f"**Answer:** {inferred_answer['answer']}")
            if inferred_answer.get("confidence"):
                parts.append(f"**Confidence:** {inferred_answer['confidence']}")
            if inferred_answer.get("reasoning"):
                parts.append(f"**Reasoning:** {inferred_answer['reasoning']}")
            
            evidence = inferred_answer.get("supporting_evidence", [])
            if evidence:
                parts.append("**Supporting Evidence:**")
                for ev in evidence:
                    parts.append(f"- {ev}")
        
        # Inferred query information
        inferred_query = observation.get("inferred_query", {})
        if inferred_query:
            parts.append("\n## Inferred Query Analysis")
            if inferred_query.get("query"):
                parts.append(f"**Inferred Query:** {inferred_query['query']}")
            if inferred_query.get("query_type"):
                parts.append(f"**Query Type:** {inferred_query['query_type']}")
            
            topics = inferred_query.get("key_topics", [])
            if topics:
                parts.append(f"**Key Topics:** {', '.join(topics)}")
        
        # Additional analysis
        analysis = observation.get("analysis", {})
        if analysis:
            parts.append("\n## Additional Analysis")
            if analysis.get("key_topics"):
                parts.append(f"**Key Topics:** {', '.join(analysis['key_topics'])}")
            if analysis.get("supporting_evidence"):
                parts.append("**Supporting Evidence:**")
                for ev in analysis['supporting_evidence']:
                    parts.append(f"- {ev}")
        
        # RRF scores if available
        rrf_scores = observation.get("rrf_scores", {})
        if rrf_scores:
            parts.append("\n## Relevance Scores")
            sorted_scores = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            for ctx_id, score in sorted_scores:
                parts.append(f"- {ctx_id}: {score:.4f}")
        
        return "\n".join(parts)

    async def generate_response(self, observation: dict) -> GeneratedResponse:
        """Generate a comprehensive response with citations.
        
        Args:
            observation: Dictionary containing all context information including:
                - original_query: The user's original question
                - contexts: List of retrieved context items
                - inferred_answer: Results from answer inference
                - inferred_query: Results from query inference
                - analysis: Additional analysis data
                
        Returns:
            GeneratedResponse: Structured response with citations.
        """
        # Build the full context string
        context_string = self._build_context_string(observation)
        
        # Make the LLM call with structured output
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": RESPONSE_GENERATION_PROMPT},
                {"role": "user", "content": context_string}
            ],
            temperature=0.1,  # Low temperature for factual responses
            max_tokens=4096,  # Allow for comprehensive responses
            extra_body={
                "guided_json": GeneratedResponse.model_json_schema()
            }
        )
        
        response_content = completion.choices[0].message.content
        parsed_response = json.loads(response_content)
        return GeneratedResponse(**parsed_response)

    def format_response_with_references(self, generated: GeneratedResponse) -> str:
        """Format the response with a proper references section.
        
        Args:
            generated: The generated response object.
            
        Returns:
            Formatted string with response and references.
        """
        parts = [generated.response]
        
        if generated.citations:
            parts.append("\n\n---\n## References\n")
            for citation in sorted(generated.citations, key=lambda c: c.id):
                ref_line = f"[{citation.id}] **{citation.source_name}** "
                ref_line += f"({citation.source_type})"
                if citation.file_path:
                    ref_line += f" - _{citation.file_path}_"
                ref_line += f"\n    > {citation.excerpt}"
                parts.append(ref_line)
        
        if generated.limitations:
            parts.append("\n\n---\n## Limitations\n")
            for limitation in generated.limitations:
                parts.append(f"- {limitation}")
        
        return "\n".join(parts)

    async def act(self, observation: dict) -> dict:
        """Process an observation and generate a response with citations.
        
        Args:
            observation: Dictionary containing the full context.
            
        Returns:
            Dictionary with the generated response and metadata.
        """
        # Validate that we have some context to work with
        if not observation:
            return {
                "error": "No observation provided",
                "response": None
            }
        
        has_content = (
            observation.get("contexts") or 
            observation.get("original_query") or
            observation.get("inferred_answer")
        )
        
        if not has_content:
            return {
                "error": "Insufficient context to generate response",
                "response": None
            }
        
        # Generate the response
        generated = await self.generate_response(observation)
        
        # Format the full response with references
        formatted_response = self.format_response_with_references(generated)
        
        return {
            "response": formatted_response,
            "raw_response": generated.response,
            "citations": [c.model_dump() for c in generated.citations],
            "confidence": generated.confidence,
            "limitations": generated.limitations,
            "key_points": generated.key_points,
            "citation_count": len(generated.citations)
        }

    async def think(self, data: dict) -> dict:
        """Analyze the response quality and provide insights.
        
        Args:
            data: Data containing the generated response.
            
        Returns:
            Analysis of the response quality.
        """
        citations = data.get("citations", [])
        confidence = data.get("confidence", 0.0)
        limitations = data.get("limitations", [])
        key_points = data.get("key_points", [])
        
        # Analyze citation coverage
        citation_types = {}
        for citation in citations:
            source_type = citation.get("source_type", "unknown")
            citation_types[source_type] = citation_types.get(source_type, 0) + 1
        
        # Determine response quality
        is_well_cited = len(citations) >= 2
        is_high_confidence = confidence >= 0.7
        has_limitations = len(limitations) > 0
        
        quality_score = 0.0
        if is_well_cited:
            quality_score += 0.3
        if is_high_confidence:
            quality_score += 0.4
        if key_points:
            quality_score += 0.2
        if has_limitations:  # Being transparent about limitations is good
            quality_score += 0.1
        
        return {
            "quality_score": quality_score,
            "is_well_cited": is_well_cited,
            "is_high_confidence": is_high_confidence,
            "citation_count": len(citations),
            "citation_types": citation_types,
            "key_points_count": len(key_points),
            "limitations_count": len(limitations),
            "recommendation": self._get_quality_recommendation(quality_score, is_well_cited, is_high_confidence)
        }

    def _get_quality_recommendation(
        self, 
        quality_score: float, 
        is_well_cited: bool, 
        is_high_confidence: bool
    ) -> str:
        """Generate a quality recommendation based on analysis.
        
        Args:
            quality_score: Overall quality score.
            is_well_cited: Whether the response has sufficient citations.
            is_high_confidence: Whether the confidence is high.
            
        Returns:
            Recommendation string.
        """
        if quality_score >= 0.8:
            return "Response is comprehensive and well-supported."
        elif quality_score >= 0.5:
            issues = []
            if not is_well_cited:
                issues.append("add more citations")
            if not is_high_confidence:
                issues.append("gather more context to improve confidence")
            return f"Response is acceptable but could be improved: {', '.join(issues)}."
        else:
            return "Response needs significant improvement. Consider retrieving more context."


# Convenience function for standalone usage
async def generate_cited_response(
    observation: dict,
    base_url: str = "http://localhost:8001/v1",
    api_key: str = "EMPTY",
    model: str = "meta-llama/Llama-2-7b-chat-hf"
) -> dict:
    """Generate a response with citations from the given observation.
    
    Args:
        observation: Dictionary containing the full context.
        base_url: vLLM API base URL.
        api_key: API key for authentication.
        model: Model to use for generation.
        
    Returns:
        Dictionary with response and metadata.
    """
    agent = ResponseAgent(
        name="response_generator",
        base_url=base_url,
        api_key=api_key,
        model=model
    )
    return await agent.act(observation)
