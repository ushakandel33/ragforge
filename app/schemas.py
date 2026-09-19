"""
Pydantic models used for request/response validation and for forcing the
LLM into structured (valid JSON) output.
"""
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User's question")
    use_rag: bool = Field(True, description="Whether to retrieve context before answering")
    session_id: Optional[str] = Field(None, description="Optional session id for future memory support")


class Source(BaseModel):
    text: str
    metadata: dict[str, Any] = {}
    score: float


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]
    result: Any = None


class ChatResponse(BaseModel):
    """
    Guaranteed-shape structured response returned by the API, regardless of
    which underlying provider answered the request.
    """
    answer: str
    sources: list[Source] = []
    tool_calls: list[ToolCall] = []
    provider_used: Literal["gemini", "groq", "local", "fallback_static"]
    cached: bool = False


class AgentRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User's question")
    use_rag: bool = Field(True, description="Allow the agent to retrieve document evidence")
    session_id: Optional[str] = Field(None, description="Optional session id for future memory support")


AgentActionName = Literal[
    "RETRIEVE", "CALCULATE", "GET_TIME", "VERIFY", "REVISE", "ASK_USER", "FINALIZE"
]


class AgentDecision(BaseModel):
    action: AgentActionName
    reason: str = ""
    query: Optional[str] = None
    expression: Optional[str] = None
    answer: Optional[str] = None
    evidence_sufficient: bool = False


class AgentResponse(BaseModel):
    answer: str
    status: Literal["completed", "clarification_required", "degraded", "failed"]
    iterations: int
    actions: list[AgentActionName] = []
    tools_used: list[str] = []
    failures: list[str] = []
    sources: list[Source] = []
    provider_used: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    token_usage_note: Optional[str] = None


class IngestResponse(BaseModel):
    filename: str
    chunks_created: int
    status: Literal["success", "failed"]


class HealthResponse(BaseModel):
    status: str
    providers: dict[str, bool]
