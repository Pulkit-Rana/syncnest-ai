from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Union

_memory_store: dict[str, ConversationBufferMemory] = {}

def get_memory(session_id: str = "default") -> ConversationBufferMemory:
    """Returns (or creates) a session-specific memory buffer."""
    if session_id not in _memory_store:
        _memory_store[session_id] = ConversationBufferMemory(
            memory_key="history",
            return_messages=True,
            input_key="input",
        )
    return _memory_store[session_id]

def load_conversation_history(session_id: str = "default") -> List[Union[HumanMessage, AIMessage]]:
    """Returns raw message history for a session as LangChain message objects."""
    return get_memory(session_id).chat_memory.messages

def save_turn(user: str, ai: str, session_id: str = "default"):
    """Appends a user→agent turn to the session's memory buffer."""
    memory = get_memory(session_id)
    if user:
        memory.chat_memory.add_user_message(user)
    if ai:
        memory.chat_memory.add_ai_message(ai)

def reset_memory(session_id: str = "default"):
    """Clears the entire memory buffer for a session."""
    get_memory(session_id).clear()

def format_memory_for_prompt(session_id: str = "default") -> str:
    """
    Formats a session's memory history into a plain-text string for LLM injection.
    Each turn is prepended with 'User:' or 'Agent:'.
    """
    memory = get_memory(session_id)
    lines = []
    for msg in memory.chat_memory.messages:
        role = "User" if isinstance(msg, HumanMessage) else "Agent"
        content = msg.content.strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()
 