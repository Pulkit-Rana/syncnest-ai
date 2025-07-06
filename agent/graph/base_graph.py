from langgraph.graph import StateGraph
from agent.types import ReasoningState
from agent.node.conversation_classifier_node import conversation_classifier_node
from agent.node.greeting_node import greeting_node
from agent.node.farewell_node import farewell_node
from agent.node.general_chat_node import general_chat_node
from agent.node.product_question_node import product_question_node

# Optional web search node
try:
    from agent.node.web_search_node import web_search_node
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False

from agent.node.bug_template_builder_node import bug_template_builder_node
from agent.node.bug_submission_node import bug_submission_node

import copy
from typing import Generator

def build_graph():
    # Initialize the state graph
    workflow = StateGraph(ReasoningState)

    # --- Dynamic Node Registration ---
    nodes = [
        ("classifier", conversation_classifier_node()),
        ("greeting", greeting_node()),
        ("farewell", farewell_node()),
        ("general_chat", general_chat_node()),
        ("product_question", product_question_node()),
    ]
    if WEB_SEARCH_AVAILABLE:
        nodes.append(("web_search", web_search_node()))
    nodes.extend([
        ("bug_template_builder", bug_template_builder_node()),
        ("bug_submission", bug_submission_node()),
    ])

    for name, node in nodes:
        workflow.add_node(name, node)

    # Entry point
    workflow.set_entry_point("classifier")

        # --- Conditional edges via route function ---
    def route(state: ReasoningState):
        # Bug logging flow
        if state.intent == "bug_log" and not getattr(state, "bug_template", None):
            return "bug_template_builder"
        if state.intent == "bug_log" and getattr(state, "bug_template", None):
            return "bug_submission"
        # Type-based routing
        mapping = {
            "greeting": "greeting",
            "farewell": "farewell",
            "general_chat": "general_chat",
            "product_question": "product_question",
        }
        if WEB_SEARCH_AVAILABLE:
            mapping["web_search"] = "web_search"
        # Default fallback
        return mapping.get(state.type, "general_chat")

    workflow.add_conditional_edges("classifier", route)

    # Compile the graph
    graph = workflow.compile()

    # --- Ensure Immutable State Inputs ---
    original_invoke = graph.invoke
    def invoke(state: ReasoningState) -> ReasoningState:
        return original_invoke(copy.deepcopy(state))
    graph.invoke = invoke

    # --- Streaming Support Stub ---
    def invoke_stream(state: ReasoningState) -> Generator[ReasoningState, None, None]:
        final = graph.invoke(state)
        yield final
    graph.invoke_stream = invoke_stream

    return graph
