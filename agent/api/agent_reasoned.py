import json
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from agent.graph.base_graph import build_graph
from agent.memory.memory import format_memory_for_prompt, save_turn
from agent.types import ReasoningState

logger = logging.getLogger(__name__)
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
async def run_agent_reasoning(request: AgentRequest) -> dict:
    sid = request.session_id
    stored = _state_store.get(sid)

    if stored:
        state = ReasoningState(**stored)
        state.user_input = request.input
        state.history = format_memory_for_prompt(sid)
        logger.debug(f"Loaded stored state for session {sid}")
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
        logger.debug(f"Created new state for session {sid}")

    try:
        result = await agent.invoke(state) if callable(getattr(agent.invoke, "__await__", None)) else agent.invoke(state)
        if not isinstance(result, ReasoningState):
            result = ReasoningState(**result)
        logger.debug(f"Agent invocation successful for session {sid}")
    except Exception as e:
        logger.exception(f"Agent pipeline error for session {sid}: {e}")
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

@router.post("/reasoned/stream")
async def run_agent_reasoning_stream(request: AgentRequest) -> StreamingResponse:
    sid = request.session_id
    stored = _state_store.get(sid)

    if stored:
        state = ReasoningState(**stored)
        state.user_input = request.input
        state.history = format_memory_for_prompt(sid)
        logger.debug(f"Loaded stored state for streaming session {sid}")
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
        logger.debug(f"Created new state for streaming session {sid}")

    async def event_generator():
        try:
            stream_method = agent.stream
            if callable(getattr(stream_method, "__aiter__", None)):
                async for step_dict in stream_method(state):
                    try:
                        step_data = next(iter(step_dict.values())) if len(step_dict) == 1 else step_dict
                        step = ReasoningState(**step_data)
                        if step.thought:
                            yield f"data: {json.dumps({'type': 'thought', 'content': step.thought})}\n\n"
                        if step.response:
                            yield f"data: {json.dumps({'type': 'response', 'content': step.response})}\n\n"
                    except Exception as e:
                        logger.error(f"Error parsing step dict in async stream: {e}")
            else:
                for step_dict in stream_method(state):
                    try:
                        step_data = next(iter(step_dict.values())) if len(step_dict) == 1 else step_dict
                        step = ReasoningState(**step_data)
                        if step.thought:
                            yield f"data: {json.dumps({'type': 'thought', 'content': step.thought})}\n\n"
                        if step.response:
                            yield f"data: {json.dumps({'type': 'response', 'content': step.response})}\n\n"
                    except Exception as e:
                        logger.error(f"Error parsing step dict in sync stream: {e}")

            yield "data: [DONE]\n\n"
            logger.debug(f"Streaming completed for session {sid}")
        except Exception as e:
            logger.exception(f"Error in streaming: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
