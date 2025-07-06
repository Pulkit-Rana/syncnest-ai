from langgraph.graph import StateGraph
from agent.types import ReasoningState
from agent.node.conversation_classifier_node import conversation_classifier_node
from agent.node.greeting_node import greeting_node
from agent.node.farewell_node import farewell_node
from agent.node.general_chat_node import general_chat_node
from agent.node.product_question_node import product_question_node

try:
    from agent.node.web_search_node import web_search_node
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False

from agent.node.bug_template_builder_node import bug_template_builder_node
from agent.node.bug_submission_node import bug_submission_node

def build_graph():
    workflow = StateGraph(ReasoningState)
    workflow.add_node("classifier", conversation_classifier_node())
    workflow.add_node("greeting", greeting_node())
    workflow.add_node("farewell", farewell_node())
    workflow.add_node("general_chat", general_chat_node())
    workflow.add_node("product_question", product_question_node())
    if WEB_SEARCH_AVAILABLE:
        workflow.add_node("web_search", web_search_node())
    workflow.add_node("bug_template_builder", bug_template_builder_node())
    workflow.add_node("bug_submission", bug_submission_node())
    workflow.set_entry_point("classifier")
    def route(state):
        if state.intent == "bug_log" and not getattr(state, "bug_template", None):
            return "bug_template_builder"
        if state.intent == "bug_log" and getattr(state, "bug_template", None):
            return "bug_submission"
        mapping = {
            "greeting": "greeting",
            "farewell": "farewell",
            "general_chat": "general_chat",
            "product_question": "product_question"
        }
        if WEB_SEARCH_AVAILABLE:
            mapping["web_search"] = "web_search"
        return mapping.get(state.type, "general_chat")
    workflow.add_conditional_edges("classifier", route)
    return workflow.compile()
