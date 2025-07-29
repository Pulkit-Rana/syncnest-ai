import logging
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from agent.utils.llm_response import call_llm
from agent.memory.memory import save_turn
from agent.types import ReasoningState
from tavily import TavilyClient
import os

logger = logging.getLogger(__name__)

PRODUCT_TRIGGER_WORDS = [
    "login", "dashboard", "feature", "app", "button", "upload", "ui", "form", "page", "account",
    "profile", "report", "error", "issue", "workflow", "search", "submit", "reset", "settings"
]

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
            return " I couldn’t find anything helpful online."
    except Exception as e:
        return f"Web search failed: {str(e)}"

def general_chat_node():
    def run(state: ReasoningState) -> ReasoningState:
        query = state.user_input.strip()
        history = state.history or ""

        state.thought = "Preparing system prompt for general, non-product chat..."  # [Step 1]

        system_prompt = (
            "You are a helpful, friendly assistant for general, non-product questions. "
            "Respond conversationally and humanly. If you don't know something, say: 'I don't know.'"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Chat so far:\n{history}\n\nUser now asked:\n{query}")
        ]

        try:
            state.thought = "Calling LLM for a general chat response..."  # [Step 2]
            answer = call_llm(messages).strip()
            state.thought = f"LLM responded: {answer[:50]}..."  # partial response for debug
        except Exception as e:
            logger.error(f"LLM call failed in general_chat_node: {e}")
            state.thought = f"LLM call failed: {str(e)}"
            answer = "Sorry, I'm having trouble thinking right now. Please try again!"

        if not answer or "i don't know" in answer.lower() or "not sure" in answer.lower():
            state.thought = "LLM was uncertain. Running web search fallback..."
            response = run_web_search(query)
            state.intent = "web_search"
            state.node = "web_search"
        else:
            if any(word in query.lower() for word in PRODUCT_TRIGGER_WORDS):
                state.thought = "Detected possible product keyword in general chat."
                answer += (
                    "\n\n(If this is about your app, workflow, or a feature, please rephrase or try again for more tailored help!)"
                )
            else:
                state.thought = "General chat completed. Returning answer."
            response = answer
            state.intent = "general_chat"
            state.node = "general_chat"

        save_turn(query, response)
        state.response = response
        return state

    return RunnableLambda(run)
