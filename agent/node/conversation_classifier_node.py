from langchain_core.runnables import RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from agent.utils.llm_response import call_llm
from agent.types import ReasoningState

ALLOWED_TYPES = {
    "greeting", "farewell", "general_chat", "web_search", "product_question", "bug_log"
}
BUG_KEYWORDS = [
    "log a bug", "log bug", "create a bug", "create a story", "log it",
    "raise bug", "file bug", "bug report", "register bug", "log issue",
    "file issue", "register issue"
]

REFINED_PROMPT = """
You are an AI assistant for a webapp/product. Classify the user's latest message into ONE of these types ONLY:

- greeting: For greetings like 'Hi', 'Hello', 'Namaste', or similar.
- farewell: For goodbyes like 'Bye', 'See you', or similar.
- general_chat: For messages that are NOT about this product’s features, UI, usage, issues, or functionality. Includes jokes, small talk, general facts, tech outside this product, or non-product questions.
- web_search: Only use if the user EXPLICITLY requests to search or look up something on the web.
- product_question: Use for ANY message about this product’s features, functionality, usage, settings, screens, bugs, enhancements, configuration, or troubleshooting.
- bug_log: Use when the user explicitly asks to log a bug or create a story in the system.

RULES:
- For anything not in allowed types, default to 'product_question'.

Reply with the label only, in lowercase. Do not explain or elaborate.
"""

def conversation_classifier_node():
    def classify(state: ReasoningState) -> ReasoningState:
        user_input = state.user_input.strip().lower()
        history = "\n".join(state.history) if isinstance(state.history, list) else (state.history or "")

        # Keyword detection first (bulletproof for bug log intent)
        if any(kw in user_input for kw in BUG_KEYWORDS):
            label = "bug_log"
        else:
            messages = [
                SystemMessage(content=REFINED_PROMPT),
                HumanMessage(content=f"Chat so far:\n{history}\n\nUser now said:\n{state.user_input}")
            ]
            # Retry loop to ensure valid output
            for _ in range(2):
                label = call_llm(messages).strip().lower().rstrip(".")
                if label in ALLOWED_TYPES:
                    break
                label = "product_question"

        state.type = label
        state.intent = label
        return state

    return RunnableLambda(classify)
