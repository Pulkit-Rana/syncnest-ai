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

        # System prompt for LLM chat
        system_prompt = (
            "You are a smart and friendly assistant. Respond conversationally like a helpful human teammate.\n"
            "You can use the full chat history. If you are unsure or don't know something, say: 'I don't know.'"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Chat so far:\n{history}\n\nUser now asked:\n{query}")
        ]

        answer = call_llm(messages).strip()

        # Fallback logic: If LLM can't answer, try web search
        if not answer or "i don't know" in answer.lower() or "not sure" in answer.lower():
            response = run_web_search(query)
            # You may also want to set state.intent = "web_search" here for audit trail:
            state.intent = "web_search"
        else:
            response = answer
            state.intent = "general_chat"

        save_turn(query, response)
        state.response = response
        return state

    return RunnableLambda(run)
