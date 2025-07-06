# agent/api/chat_reset.py

from fastapi import APIRouter
from agent.memory.memory import reset_memory

router = APIRouter()

@router.post("/reset")
async def reset_conversation():
    """
    Resets the conversation memory for the current session.
    """
    reset_memory()
    return {"message": "✅ Memory has been cleared."}
