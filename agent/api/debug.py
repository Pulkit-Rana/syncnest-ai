# agent/api/chat_debug.py

from fastapi import APIRouter
from agent.memory.memory import load_conversation_history
from langchain_core.messages import HumanMessage

router = APIRouter()

@router.get("/chat/debug")
def debug_chat():
    """
    Returns the full conversation history for debugging purposes.
    Each message is labeled by role ('user' or 'assistant').
    """
    messages = load_conversation_history()
    trace = [
        {
            "role": "user" if isinstance(m, HumanMessage) else "assistant",
            "content": m.content.strip()
        }
        for m in messages
    ]
    return {
        "history": trace,
        "turns": len(trace)
    }
