"""
Agent State Schema for Agentic StateGraph Workflows.
Defines typed state attributes, execution history, memory context, and citation evidence.
"""

from typing import Dict, List, Any, Optional, TypedDict


class AgentState(TypedDict):
    # User Request Context
    session_id: str
    user_role: str
    raw_query: str
    sanitized_query: str
    detected_pii: Dict[str, str]

    # Agentic Workflow & Routing State
    intent: str
    step_count: int
    max_steps: int
    is_safe: bool
    security_flag: Optional[str]

    # Tool Execution Context
    selected_tools: List[Dict[str, Any]]
    tool_outputs: List[Dict[str, Any]]

    # Response & Grounding Verification
    retrieved_contexts: List[Dict[str, Any]]
    grounded_answer: str
    citations: List[Dict[str, Any]]
    grounding_score: float
    confidence: float
    is_hallucinated: bool
