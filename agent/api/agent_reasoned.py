import os
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from agent.graph.base_graph import build_graph
from agent.memory.memory import format_memory_for_prompt, save_turn
from agent.types import ReasoningState
from typing import AsyncGenerator, Dict, Any, Optional

router = APIRouter()
agent = build_graph()

# In-memory state store keyed by session_id (for demo/dev only; swap for Redis in prod)
_state_store: Dict[str, Dict[str, Any]] = {}

class AgentRequest(BaseModel):
    input: str = Field(..., description="User's latest message")
    session_id: str = Field("default", description="Unique session identifier for state persistence")
    stream: Optional[bool] = Field(False, description="Enable streaming response")

class AgentResponse(BaseModel):
    response: str
    intent: str
    type: str
    node: str
    context: list

async def _invoke_agent(state: ReasoningState) -> ReasoningState:
    try:
        result = agent.invoke(state)
        if not isinstance(result, ReasoningState):
            result = ReasoningState(**result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent pipeline error: {e}")

async def _stream_agent(state: ReasoningState) -> AsyncGenerator[bytes, None]:
    # Placeholder: adapt this to your graph's streaming API
    for chunk in agent.invoke_stream(state):  # assume generator of partial states
        try:
            state = ReasoningState(**chunk) if not isinstance(chunk, ReasoningState) else chunk
            yield (state.response + "\n").encode("utf-8")
        except Exception:
            continue

@router.post("/reasoned", response_model=AgentResponse)
async def run_agent_reasoning(request: AgentRequest, _: Request):
    sid = request.session_id
    stored = _state_store.get(sid)

    # Load or initialize state
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
        )

    # Streaming path
    if request.stream:
        async def streamer():
            # call streaming invoke generator
            async for data in _stream_agent(state):
                # persist partial memory if needed
                yield data
            # final persist
            final = await _invoke_agent(state)
            _state_store[sid] = final.dict()
            save_turn(request.input, final.response, sid)

        return StreamingResponse(streamer(), media_type="text/plain")

    # Non-streaming
    result = await _invoke_agent(state)
    _state_store[sid] = result.dict()
    save_turn(request.input, result.response, sid)

    return AgentResponse(
        response=result.response,
        intent=result.intent,
        type=result.type,
        node=result.node,
        context=result.context,
    )
