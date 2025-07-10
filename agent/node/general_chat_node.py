import logging
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from agent.utils.llm_response import call_llm
from agent.memory.memory import save_turn
from agent.types import ReasoningState
from tavily import TavilyClient
import os

logger = logging.getLogger(__name__)

# (Optional) Keywords to gently nudge if classifier missed a product question (paranoia catch)
PRODUCT_TRIGGER_WORDS = [
    "login", "dashboard", "feature", "app", "button", "upload", "ui", "form", "page", "account",
    "profile", "report", "error", "issue", "workflow", "search", "submit", "reset", "settings"
]

def general_chat_node():
    def run(state: ReasoningState) -> ReasoningState:
        query = state.user_input.strip()
        history = state.history or ""

        # 💬 System prompt for off-topic/casual chat
        system_prompt = (
            "You are a helpful, friendly assistant for general, non-product questions. "
            "Respond conversationally and humanly. If you don't know something, say: 'I don't know.'"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Chat so far:\n{history}\n\nUser now asked:\n{query}")
        ]

        try:
            answer = call_llm(messages).strip()
        except Exception as e:
            logger.error(f"LLM call failed in general_chat_node: {e}")
            answer = "Sorry, I'm having trouble thinking right now. Please try again!"

        # If LLM admits it doesn't know, fallback to web search
        if not answer or "i don't know" in answer.lower() or "not sure" in answer.lower():
            response = run_web_search(query)
            state.intent = "web_search"
            state.node = "web_search"
        else:
            # Optional: gentle nudge if the classifier let a product question slip through
            if any(word in query.lower() for word in PRODUCT_TRIGGER_WORDS):
                answer += (
                    "\n\n(If this is about your app, workflow, or a feature, please rephrase or try again for more tailored help!)"
                )
            response = answer
            state.intent = "general_chat"
            state.node = "general_chat"

        save_turn(query, response)
        state.response = response
        return state

    return RunnableLambda(run)

# 🌐 Internet Search Fallback (unchanged)
def run_web_search(query: str) -> str:
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        result = client.search(query=query, max_results=1)
        top = result["results"][0] if result.get("results") else None
        if top:
            snippet = top.get("answer") or top.get("content", "")
            url = top.get("url", "")
            return f"🔎 {snippet}\n(Source: {url})"
        else:
            return "❓ I couldn’t find anything helpful online."
    except Exception as e:
        return f"⚠️ Web search failed: {str(e)}"
