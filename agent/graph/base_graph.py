import logging
from langgraph.graph import StateGraph
from agent.types import ReasoningState

# Core nodes
from agent.node.conversation_classifier_node import conversation_classifier_node
from agent.node.greeting_node import greeting_node
from agent.node.farewell_node import farewell_node
from agent.node.general_chat_node import general_chat_node
from agent.node.product_question_node import product_question_node

# Optional
try:
    from agent.node.web_search_node import web_search_node
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False

# Bug flow
from agent.node.bug_template_builder_node import bug_template_builder_node
from agent.node.bug_submission_node import bug_submission_node

# Story flow
from agent.node.story_template_builder_node import story_template_builder_node
from agent.node.story_submission_node import story_submission_node

# --- Fallback node (add this node to your agent.node package if not present) ---
def fallback_node():
    def node(state: ReasoningState):
        state.response = (
            "Sorry, I didn’t understand that. "
            "Can you rephrase or specify your question?"
        )
        return state
    return node

logger = logging.getLogger(__name__)

def build_graph():
    workflow = StateGraph(ReasoningState)

    # Classifier
    workflow.add_node("classifier", conversation_classifier_node())

    # Core conversation paths
    workflow.add_node("greeting", greeting_node())
    workflow.add_node("farewell", farewell_node())
    workflow.add_node("general_chat", general_chat_node())
    workflow.add_node("product_question", product_question_node())
    if WEB_SEARCH_AVAILABLE:
        workflow.add_node("web_search", web_search_node())

    # Bug flow
    workflow.add_node("bug_template_builder", bug_template_builder_node())
    workflow.add_node("bug_submission", bug_submission_node())

    # Story flow
    workflow.add_node("story_template_builder", story_template_builder_node())
    workflow.add_node("story_submission", story_submission_node())

    # Fallback node
    workflow.add_node("fallback", fallback_node())

    # Entry point
    workflow.set_entry_point("classifier")

    # Router logic (hardened)
    def route(state):
        logger.info(f"[Router] intent='{getattr(state, 'intent', None)}' | state={state}")

        if state.intent == "bug_log":
            if not getattr(state, "bug_template", None):
                logger.info("[Router] Routing to bug_template_builder")
                return "bug_template_builder"
            logger.info("[Router] Routing to bug_submission")
            return "bug_submission"

        if state.intent == "story_log":
            if not getattr(state, "story_template", None):
                logger.info("[Router] Routing to story_template_builder")
                return "story_template_builder"
            logger.info("[Router] Routing to story_submission")
            return "story_submission"

        mapping = {
            "greeting": "greeting",
            "farewell": "farewell",
            "general_chat": "general_chat",
            "product_question": "product_question"
        }
        if WEB_SEARCH_AVAILABLE:
            mapping["web_search"] = "web_search"

        if state.intent in mapping:
            logger.info(f"[Router] Routing to {mapping[state.intent]}")
            return mapping[state.intent]

        # Robust fallback: route unknown/ambiguous intents to fallback node
        logger.warning(f"[Router] Unknown or ambiguous intent: {state.intent}. Routing to fallback node.")
        return "fallback"

    workflow.add_conditional_edges("classifier", route)
    return workflow.compile()
