import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from agent.graph.base_graph import build_graph
from agent.memory.memory import format_memory_for_prompt, save_turn
from agent.types import ReasoningState

router = APIRouter()
agent = build_graph()

# In-memory state store keyed by session_id
_state_store: dict[str, dict] = {}

class AgentRequest(BaseModel):
    input: str = Field(..., description="User's latest message")
    session_id: str = Field(default="default", description="Unique session identifier for state persistence")

class AgentResponse(BaseModel):
    response: str
    intent: str
    type: str
    node: str
    context: list 

@router.post("/reasoned", response_model=AgentResponse)
async def run_agent_reasoning(request: AgentRequest):
    sid = request.session_id
    stored = _state_store.get(sid)

    if stored:
        state = ReasoningState(**stored)
        state.user_input = request.input
        state.history = format_memory_for_prompt(sid)
    else:
        state = ReasoningState(
            user_input=request.input,
            type="",
            intent="",
            node="",
            context=[],
            response="",
            history=format_memory_for_prompt(sid),
            ado_context=None,
            web_result=None,
            bug_template=None,
            story_template=None, 
        )

    try:
        result = agent.invoke(state)
        if not isinstance(result, ReasoningState):
            try:
                result = ReasoningState(**result)
            except Exception as e:
                print(" FastAPI Result Parse Error:", e)
                import traceback; traceback.print_exc()
                raise HTTPException(
                    status_code=500,
                    detail=f"Internal agent error: Could not parse agent state ({e})."
                )
    except Exception as e:
        print("FastAPI Exception:", e)
        import traceback; traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal agent pipeline error: {e}"
        )

    _state_store[sid] = result.dict()
    save_turn(request.input, result.response, sid)

    return {
        "response": result.response,
        "intent": result.intent,
        "type": result.type,
        "node": result.node,
        "context": list(result.context) if isinstance(result.context, (list, tuple)) else [result.context] if result.context else [],
}
