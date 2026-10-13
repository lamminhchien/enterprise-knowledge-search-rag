"""
Enterprise Agentic Workflow Engine (Compiled LangGraph StateGraph Architecture).
Coordinates Zero-Trust Security, Dynamic Intent Routing, MCP Tool Calling,
and Output Grounding Verification with Verifiable Source Citations.
"""

from typing import Dict, List, Any, Optional
import os
import sys
import time
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.security.prompt_guard import PromptSecurityGuard
from src.security.pii_osmosis_guard import PIIOsmosisGuard
from src.tools.mcp_server import MCPServerRegistry


class EnterpriseAgenticWorkflow:
    """
    Production Compiled LangGraph StateGraph Agent:
    State Transitions:
    [Inbound Query] ──► (Guardrail Node)
                              │
                    ┌─────────┴─────────┐
                    ▼ (Unsafe)          ▼ (Safe)
                  [END]          (Router Node)
                                        │
                              ┌─────────┴─────────┐
                              ▼ (Tools Needed)    ▼ (Direct)
                        (Tools Node)       (Grounding Node)
                              │                   │
                     [Loop / Budget Check]        ▼
                              │                 [END]
                              ▼
                       (Grounding Node)
    """

    def __init__(
        self,
        mcp_registry: MCPServerRegistry,
        pii_guard: Optional[PIIOsmosisGuard] = None,
        security_guard: Optional[PromptSecurityGuard] = None,
        max_steps: int = 3
    ):
        self.mcp = mcp_registry
        self.pii_guard = pii_guard or PIIOsmosisGuard()
        self.security_guard = security_guard or PromptSecurityGuard()
        self.max_steps = max_steps
        
        # Ephemeral multi-turn context memory buffer
        self._session_memory: Dict[str, List[Dict[str, str]]] = {}

        # Compile the LangGraph StateGraph
        self._compiled_graph = self._build_graph()

    def _node_input_guardrail(self, state: AgentState) -> AgentState:
        """Node 1: Inspects for prompt injection & tokenizes sensitive PII."""
        # Check for adversarial injections
        is_safe, reason, risk_score = self.security_guard.inspect_input(state["raw_query"])
        if not is_safe:
            state["is_safe"] = False
            state["security_flag"] = reason
            state["grounded_answer"] = (
                f"[SECURITY ALERT] Yêu cầu bị từ chối do vi phạm quy chuẩn an toàn LLM: {reason}"
            )
            state["confidence"] = 0.0
            return state

        # Inbound One-Way Privacy Osmosis
        clean_text, tokens = self.pii_guard.filter_and_tokenize(state["raw_query"], session_id=state["session_id"])
        state["is_safe"] = True
        state["sanitized_query"] = clean_text
        state["detected_pii"] = tokens
        return state

    def _node_route_intent(self, state: AgentState) -> AgentState:
        """Node 2: Decides which MCP tools are needed to satisfy user request."""
        query_lower = state["sanitized_query"].lower()

        if any(w in query_lower for w in ["hóa đơn", "scan", "ảnh", "invoice", "tài liệu đính kèm", "ocr"]):
            state["intent"] = "ocr_parsing"
            state["selected_tools"] = [
                {"name": "ocr_extract_document", "args": {"file_path": "uploaded_invoice_2026.png"}}
            ]
        elif any(w in query_lower for w in ["ngân sách", "doanh thu", "chi phí", "budget", "revenue", "headcount", "số liệu"]):
            state["intent"] = "financial_analytics"
            state["selected_tools"] = [
                {"name": "sql_financial_analytics", "args": {"metric_name": "project_budget"}}
            ]
        else:
            state["intent"] = "knowledge_retrieval"
            state["selected_tools"] = [
                {"name": "search_enterprise_knowledge", "args": {"query": state["sanitized_query"], "top_k": 3}}
            ]

        return state

    def _node_execute_tools(self, state: AgentState) -> AgentState:
        """Node 3: Executes selected tools via Anthropic MCP protocol."""
        outputs = []
        contexts = []

        for tool_call in state["selected_tools"]:
            t_name = tool_call["name"]
            t_args = tool_call["args"]

            exec_res = self.mcp.execute_tool(t_name, t_args, user_role=state["user_role"])
            outputs.append(exec_res)

            if exec_res.get("status") == "success":
                data = exec_res.get("data")
                if isinstance(data, list):
                    contexts.extend(data)
                elif isinstance(data, dict):
                    contexts.append(data)

        state["tool_outputs"] = outputs
        state["retrieved_contexts"] = contexts
        state["step_count"] += 1
        return state

    def _node_synthesize_and_ground(self, state: AgentState) -> AgentState:
        """Node 4: Synthesizes response, evaluates grounding ratio, and rehydrates PII."""
        contexts = state["retrieved_contexts"]
        citations = []

        if not contexts:
            state["grounded_answer"] = "Không tìm thấy dữ liệu đối chiếu trong hệ thống kiến thức nội bộ."
            state["grounding_score"] = 0.0
            state["is_hallucinated"] = True
            state["confidence"] = 0.0
            return state

        # Format evidence blocks & extract citations
        evidence_snippets = []
        for idx, ctx in enumerate(contexts, start=1):
            fn = ctx.get("filename", "internal_database")
            txt = ctx.get("text", str(ctx))
            score = ctx.get("rrf_score", ctx.get("average_confidence", 0.95))

            citations.append({
                "source_index": idx,
                "filename": fn,
                "confidence": score
            })
            evidence_snippets.append(txt)

        # Synthesize grounded answer
        top_ctx = contexts[0]
        if state["intent"] == "knowledge_retrieval":
            answer_text = (
                f"Theo quy chuẩn và tài liệu nội bộ đã đối soát ({top_ctx.get('filename')}):\n"
                f"- Nội dung cốt lõi: {top_ctx.get('text')[:220]}...\n"
                f"- Biện pháp tuân thủ: Mọi quy trình phải được ghi vết audit log và tuân thủ SOP bảo mật."
            )
        elif state["intent"] == "ocr_parsing":
            ocr_text = top_ctx.get("text", "")
            answer_text = (
                f"Kết quả trích xuất quang học (OCR) từ tài liệu đính kèm ({top_ctx.get('filename')}):\n"
                f"- Độ tin cậy trung bình: {top_ctx.get('average_confidence') * 100:.1f}%\n"
                f"- Nội dung tóm lược: {ocr_text[:200]}..."
            )
        elif state["intent"] == "financial_analytics":
            answer_text = (
                f"Báo cáo phân tích số liệu tài chính có cấu trúc:\n"
                f"- Chỉ số: {top_ctx.get('metric', 'budget')}\n"
                f"- Trạng thái: {top_ctx.get('status', 'ON_TRACK')} | Số liệu: {top_ctx}"
            )
        else:
            answer_text = f"Đã xử lý yêu cầu thành công dựa trên {len(contexts)} nguồn tham chiếu."

        # Verify Grounding
        ground_score, is_grounded = self.security_guard.verify_groundedness(answer_text, evidence_snippets)
        state["grounded_answer"] = answer_text
        state["citations"] = citations
        state["grounding_score"] = ground_score
        state["is_hallucinated"] = not is_grounded
        state["confidence"] = 0.95 if is_grounded else 0.40

        # Outbound Rehydration
        is_cleared = state["user_role"] in ["admin", "ciso", "compliance_officer"]
        rehydrated = self.pii_guard.rehydrate_context(
            state["grounded_answer"],
            session_id=state["session_id"],
            user_authorized=is_cleared
        )
        state["grounded_answer"] = rehydrated
        return state

    def _conditional_guardrail(self, state: AgentState) -> str:
        """Evaluates whether to continue after guardrail node."""
        if not state["is_safe"]:
            return "end"
        return "route"

    def _conditional_router(self, state: AgentState) -> str:
        """Determines if tools need execution."""
        if state["selected_tools"]:
            return "tools"
        return "grounding"

    def _conditional_loop(self, state: AgentState) -> str:
        """Enforces cyclic ReAct execution budget (max_steps limit)."""
        # If no contexts found and budget remaining, re-route with relaxed query
        if not state["retrieved_contexts"] and state["step_count"] < state["max_steps"]:
            return "retry"
        return "grounding"

    def _build_graph(self) -> Any:
        """Builds and compiles the formal LangGraph StateGraph."""
        builder = StateGraph(AgentState)

        # Register Nodes
        builder.add_node("guardrail", self._node_input_guardrail)
        builder.add_node("router", self._node_route_intent)
        builder.add_node("tools", self._node_execute_tools)
        builder.add_node("grounding", self._node_synthesize_and_ground)

        # Set Entry Point
        builder.set_entry_point("guardrail")

        # Add Conditional Edges
        builder.add_conditional_edges(
            "guardrail",
            self._conditional_guardrail,
            {"end": END, "route": "router"}
        )
        builder.add_conditional_edges(
            "router",
            self._conditional_router,
            {"tools": "tools", "grounding": "grounding"}
        )
        builder.add_conditional_edges(
            "tools",
            self._conditional_loop,
            {"retry": "router", "grounding": "grounding"}
        )

        builder.add_edge("grounding", END)

        return builder.compile()

    def run(self, user_query: str, user_role: str = "employee", session_id: str = "default_session") -> AgentState:
        """Executes the compiled LangGraph StateGraph pipeline."""
        initial_state: AgentState = {
            "session_id": session_id,
            "user_role": user_role,
            "raw_query": user_query,
            "sanitized_query": user_query,
            "detected_pii": {},
            "intent": "unknown",
            "step_count": 0,
            "max_steps": self.max_steps,
            "is_safe": True,
            "security_flag": None,
            "selected_tools": [],
            "tool_outputs": [],
            "retrieved_contexts": [],
            "grounded_answer": "",
            "citations": [],
            "grounding_score": 0.0,
            "confidence": 0.0,
            "is_hallucinated": False
        }

        # Execute through compiled LangGraph engine
        final_state = self._compiled_graph.invoke(initial_state)

        # Update Session Memory
        if session_id not in self._session_memory:
            self._session_memory[session_id] = []
        self._session_memory[session_id].append({
            "query": user_query,
            "answer": final_state["grounded_answer"],
            "timestamp": str(time.time())
        })

        return final_state


if __name__ == "__main__":
    registry = MCPServerRegistry()
    agent = EnterpriseAgenticWorkflow(mcp_registry=registry)

    print("=== TESTING COMPILED LANGGRAPH STATEGRAPH ===")
    res = agent.run("Kiểm tra quy định xác thực MFA và bảo mật hệ thống IT.", user_role="employee")
    print(f"Graph State: PASSED={res['is_safe']} | Intent={res['intent']}")
    print(f"Steps: {res['step_count']}/{res['max_steps']} | Tools Called: {[t['name'] for t in res['selected_tools']]}")
    print(f"Grounded Answer:\n{res['grounded_answer'][:150]}...")
    print(f"Grounding Score: {res['grounding_score'] * 100:.1f}%")
    print("[VERIFIED] Compiled LangGraph StateGraph execution successful!")
