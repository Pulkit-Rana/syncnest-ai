from langchain_core.runnables import RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from agent.utils.llm_response import call_llm
from agent.memory.memory import save_turn
from agent.types import ReasoningState
from tavily import TavilyClient
import os

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
        return f" Web search failed: {str(e)}"

def general_chat_node():
    def run(state: ReasoningState) -> ReasoningState:
        query = state.user_input.strip()
        history = state.history or ""

        state.thought = "Preparing system prompt for general chat."

        system_prompt = (
            "You are a smart and friendly assistant. Respond conversationally like a helpful human teammate.\n"
            "You can use the full chat history. If you are unsure or don't know something, say: 'I don't know.'"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Chat so far:\n{history}\n\nUser now asked:\n{query}")
        ]

        try:
            state.thought = "Calling LLM for general chat response."
            answer = call_llm(messages).strip()
            state.thought = f"LLM response received: {answer[:50]}..."
        except Exception as e:
            state.thought = f"LLM call failed with error: {e}"
            answer = "Sorry, I'm having trouble thinking right now. Please try again!"

        if not answer or "i don't know" in answer.lower() or "not sure" in answer.lower():
            state.thought = "LLM uncertain, falling back to web search."
            response = run_web_search(query)
            state.intent = "web_search"
            state.node = "web_search"
        else:
            response = answer
            state.intent = "general_chat"
            state.node = "general_chat"
            if any(word in query.lower() for word in ["login", "dashboard", "app", "feature", "button", "page", "story", "bug"]):
                state.thought = "Detected product-related keywords in general chat input."

        save_turn(query, response)
        state.response = response
        return state

    return RunnableLambda(run)
