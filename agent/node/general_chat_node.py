from langchain_core.runnables import RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from agent.utils.llm_response import call_llm
from agent.memory.memory import save_turn
from agent.types import ReasoningState
from tavily import TavilyClient
import os

def general_chat_node():
    def run(state: ReasoningState) -> ReasoningState:
        query = state.user_input.strip()
        history = state.history or ""

        # 🔍 Keyword-based forced web search
        search_triggers = [
            "top news", "latest news", "trending", "headlines", "breaking news",
            "today", "weather", "price of", "who won", "current events",
            "capital of", "population", "stock price", "temperature"
        ]

        if state.type == "web_search" or any(keyword in query.lower() for keyword in search_triggers):
            return run_web_search(state, query, fallback=True)

        # 💬 Ask LLM using chat history
        system_prompt = (
            "You are a smart and friendly assistant. Respond conversationally like a helpful human teammate.\n"
            "You can use the full chat history. If you are unsure or don't know something, say: 'I don't know.'"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Chat so far:\n{history}\n\nUser now asked:\n{query}")
        ]

        answer = call_llm(messages)

        # 🤖 If LLM admits it doesn't know → fallback to search
        if "i don't know" in answer.lower() or "not sure" in answer.lower():
            return run_web_search(state, query, fallback=True)

        # ✅ LLM responded well
        save_turn(query, answer)
        state.response = answer
        state.intent = "general_chat"
        state.node = "general_chat"
        return state

    return RunnableLambda(run)

# 🌐 Internet Search Fallback
def run_web_search(state: ReasoningState, query: str, fallback: bool = False) -> ReasoningState:
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        result = client.search(query=query, max_results=1)
        top = result["results"][0] if result.get("results") else None

        if top:
            snippet = top.get("answer") or top.get("content", "")
            url = top.get("url", "")
            response = f"🔎 {snippet}\n(Source: {url})"
        else:
            response = "❓ I couldn’t find anything helpful online."

    except Exception as e:
        response = f"⚠️ Web search failed: {str(e)}"

    save_turn(state.user_input, response)
    state.response = response
    state.intent = "web_search"
    state.node = "web_search"
    return state