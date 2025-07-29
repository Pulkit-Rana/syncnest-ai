import logging
from langchain_core.messages import HumanMessage
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
GREETING_KEYWORDS = [
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "thanks", "thank you", "bye", "goodbye", "see you", "take care", "welcome"
]
# Add product context keywords for fallback bias
PRODUCT_KEYWORDS = [
    "entity", "page", "visible", "dashboard", "login", "bug", "story", "feature", "button", "screen", "form",
    "profile", "app", "application", "submit", "status", "error", "issue", "report", "not working", "unable"
]

INTENT_CLASSIFIER_PROMPT = """
You are an elite AI intent classifier for an AI-powered workflow/chat agent. Given the conversation and latest user message, assign the **single best intent** from the following options:

- product_question: The user asks about the web app, features, workflow, issues, bug status, improvements, usage, or mentions *anything* product- or work-related.
- bug_log: The user reports, logs, files, or confirms a bug, or says things like 'log bug', 'add bug', 'report bug', 'this is a bug', 'submit bug'.
- story_log: The user wants to create or confirm a user story, with phrases like 'add story', 'log story', 'file story', 'new story'.
- general_chat: The user is casual, makes jokes, asks about the weather, AI, random facts, or unrelated topics. **Never use if there's any work/app/product context.**
- greeting: The user says 'hi', 'hello', 'good morning', 'good night', 'thanks', 'goodbye', 'bye', or similar short social phrases.
- clarify: If the request is **ambiguous**, unclear, or you genuinely cannot decide after checking all options, respond ONLY with 'clarify'.

## Examples:
- "How do I add a filter to the dashboard?" → product_question
- "Please log a bug: the upload button crashes" → bug_log
- "Create a new story for user onboarding" → story_log
- "What's the weather?" → general_chat
- "Hi!" → greeting
- "Thanks, bye!" → greeting
- "Can you help me?" → clarify

## Rules:
- If in doubt between 'general_chat' and 'product_question', **always pick 'product_question'**.
- Use 'greeting' for any social, short, or trivial greeting/farewell/thank you—even if the user just says 'hi' or 'ok thanks'.
- Use 'bug_log' or 'story_log' if the user is clearly confirming, logging, or submitting one of those, even if it’s just 'submit', 'log it', or similar.
- Use 'clarify' **ONLY** if you genuinely cannot infer intent, or user is totally ambiguous.
- Never invent new labels. Only use the options above.

## Conversation so far:
{history}

## Latest user message:
"{user_input}"

Respond with ONE label only from: product_question, bug_log, story_log, general_chat, greeting, clarify.
"""

def conversation_classifier_node():
    def classify(state: ReasoningState) -> ReasoningState:
        user_input = state.user_input.strip().lower()
        state.node = "conversation_classifier"
        state.thought = f"Classifying user input: '{user_input}'..."

        # Sticky confirmation if already in bug/story template flow
        if getattr(state, "bug_template", None):
            if any(k in user_input for k in CONFIRM_KEYWORDS_BUG):
                state.thought = "Detected confirmation keywords for bug logging."
                state.intent = "bug_log"
                logger.info("Sticky bug_log intent [confirmation detected]")
                return state
        if getattr(state, "story_template", None):
            if any(k in user_input for k in CONFIRM_KEYWORDS_STORY):
                state.thought = "Detected confirmation keywords for story logging."
                state.intent = "story_log"
                logger.info("Sticky story_log intent [confirmation detected]")
                return state

        # Robust greeting detection
        if any(
            (user_input == kw or user_input.startswith(kw + " ") or user_input.endswith(" " + kw))
            for kw in GREETING_KEYWORDS
        ) or user_input in GREETING_KEYWORDS:
            state.thought = "Detected greeting/farewell keyword."
            state.intent = "greeting"
            logger.info(f"Detected greeting/farewell intent: '{user_input}'")
            return state

        # LLM-based classification
        session_id = getattr(state, "session_id", "default")
        history = format_memory_for_prompt(session_id)
        state.thought = "Invoking LLM for intent classification..."

        prompt = INTENT_CLASSIFIER_PROMPT.format(
            history=history,
            user_input=state.user_input
        )
        # FIX: Use HumanMessage for LangChain, not raw dict!
        llm_input = [HumanMessage(content=prompt)]
        logger.debug(f"Classifier prompt (len={len(prompt)}): {prompt[:200].replace(chr(10),' ')}...")

        try:
            label = call_llm(llm_input).strip().lower()
            state.thought = f"LLM classified input as '{label}'."
        except Exception as e:
            logger.error(f"Classifier LLM call failed: {e}")
            state.thought = "LLM call failed, falling back to clarify."
            label = "clarify"

        allowed_labels = [
            "product_question", "bug_log", "story_log", "general_chat", "greeting", "clarify"
        ]

        if label not in allowed_labels:
            logger.warning(f"LLM gave unclear label '{label}', forcing bias fallback.")
            state.thought = f"Unknown label '{label}', using bias fallback."
            # Strong fallback bias for product keywords
            if any(w in user_input for w in PRODUCT_KEYWORDS):
                label = "product_question"
                state.thought = "Biasing to product_question based on product keywords."
            elif any(k in user_input for k in GREETING_KEYWORDS):
                label = "greeting"
                state.thought = "Biasing to greeting based on keywords."
            else:
                label = "clarify"
                state.thought = "Could not determine intent, using clarify."

        # --- Bulletproof fallback for clarify (product context should never clarify) ---
        if label == "clarify":
            if any(w in user_input for w in PRODUCT_KEYWORDS):
                state.intent = "product_question"
                state.thought = "LLM returned clarify, but product keyword detected. Forcing product_question."
                logger.warning("LLM returned clarify, but keyword biasing to product_question.")
                return state
            state.intent = "clarify"
            state.response = (
                "Can you clarify your request? Are you asking about your product, reporting a bug, logging a user story, or just chatting?"
            )
            logger.warning(f"Ambiguous intent: asking for clarification. user_input: '{user_input}'")
            state.thought = "Intent is ambiguous, asking user for clarification."
            return state

        # Otherwise, return LLM-detected intent
        state.intent = label
        logger.info(f"Final classified intent: {state.intent} | user_input: '{state.user_input}' | node: '{state.node}'")
        state.thought = f"Final classified intent: {state.intent}"
        return state

    return RunnableLambda(classify)
