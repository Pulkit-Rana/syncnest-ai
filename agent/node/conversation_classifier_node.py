import logging
from langchain_core.runnables import RunnableLambda
from agent.types import ReasoningState
from agent.memory.memory import format_memory_for_prompt
from agent.utils.llm_response import call_llm

logger = logging.getLogger(__name__) 

CONFIRM_KEYWORDS_STORY = [
    "log it", "submit", "submit story", "log story", "create story",
    "raise story", "file story", "raise ticket", "log ticket", "add as story",
    "make a story", "please file a story", "new story"
]
CONFIRM_KEYWORDS_BUG = [
    "log it", "log this bug", "submit", "submit bug", "raise bug", "confirm bug", "file bug", "add as bug",
    "please file a bug", "new bug", "add this as a bug", "report this bug"
]

INTENT_CLASSIFIER_PROMPT = """
You are an elite AI intent classifier. Given the conversation so far and the latest user message, classify the intent as one of:

- product_question (user asks about their web app, UI, features, workflow, problems, follow-up on a bug/story, or mentions anything related to the app)
- bug_log (user wants to report/log a bug or defect, or says anything about 'log bug', 'file bug', 'add this as bug', etc.)
- story_log (user wants to create/log a user story, or says 'add story', 'log story', 'file story', etc.)
- general_chat (open-ended, clearly non-product questions, jokes, weather, casual)
- greeting (hello/hi/thanks/goodbye)
- clarify (uncertain, ambiguous, or you need more info)

## Examples:
(Product Questions, Bug Logging, Story Logging, General Chat, Greeting...)
## Instructions:
- If the user's message is about the app, its features, workflows, issues, bugs, or stories, or could be a follow-up (like "What is the status of that bug?"), **always prefer 'product_question'** if unsure.
- Only use 'general_chat' if it's clearly NOT about the product or work context.
- If user is trying to confirm or submit a bug/story (see confirmation examples above), route to 'bug_log' or 'story_log' as appropriate.
- If you are unsure or the question is ambiguous, reply ONLY with 'clarify'.

Conversation history:
{history}

User message:
"{user_input}"

Respond with ONLY the label above. If you are not sure, reply 'clarify'.
"""

def conversation_classifier_node():
    def classify(state: ReasoningState) -> ReasoningState:
        user_input = state.user_input.strip().lower()
        state.node = "conversation_classifier"

        if getattr(state, "bug_template", None):
            if any(k in user_input for k in CONFIRM_KEYWORDS_BUG):
                state.intent = "bug_log"
                logger.info("Sticky bug_log intent [confirmation detected]")
                return state
        # Sticky intent for story confirmation
        if getattr(state, "story_template", None):
            if any(k in user_input for k in CONFIRM_KEYWORDS_STORY):
                state.intent = "story_log"
                logger.info("Sticky story_log intent [confirmation detected]")
                return state

        #  Always format memory for the current session for context
        session_id = getattr(state, "session_id", "default")
        history = format_memory_for_prompt(session_id)

        # Build LLM prompt and classify with bulletproof instruction and context
        prompt = INTENT_CLASSIFIER_PROMPT.format(
            history=history,
            user_input=state.user_input
        )
        llm_input = [{"role": "user", "content": prompt}]
        try:
            label = call_llm(llm_input).strip().lower()
        except Exception as e:
            logger.error(f"Classifier LLM call failed: {e}")
            label = "clarify"

        allowed_labels = [
            "product_question", "bug_log", "story_log", "general_chat", "greeting"
        ]

        # Force aggressive product bias if uncertain (guarantees never misroutes to general_chat)
        if label not in allowed_labels:
            logger.warning(f"LLM gave unclear label '{label}', falling back to clarify/product_question fallback logic.")
            if any(w in user_input for w in ["login", "dashboard", "app", "feature", "button", "page", "story", "bug", "profile", "account", "report", "issue", "submit"]):
                label = "product_question"
            else:
                label = "clarify"

        if label == "clarify":
            state.intent = "clarify"
            state.response = (
                "Can you clarify your request? Are you asking about your product, reporting a bug, logging a user story, or just chatting?"
            )
            logger.warning(f" Ambiguous intent: asking for clarification. user_input: '{user_input}'")
            return state

        # Final intent set and debug log for traceability
        state.intent = label
        logger.info(f" Final classified intent: {state.intent} | user_input: '{state.user_input}' | node: '{state.node}'")
        return state

    return RunnableLambda(classify)
