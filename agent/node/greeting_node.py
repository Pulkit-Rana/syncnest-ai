from langchain_core.runnables import RunnableLambda
from agent.types import ReasoningState
from agent.memory.memory import save_turn

def greeting_node():
    def run(state: ReasoningState) -> ReasoningState:
        # Refined greeting to match the agent's scope (Q&A + ADO bug/story logging)
        response = (
            "Hello! 👋 Welcome to the Product Support Assistant. "
            "I can help you with questions about the app's features and workflows, "
            "troubleshoot issues, and log bugs or user stories in Azure DevOps. "
            "What can I do for you today?"
        )
        # Save conversation turn and set state
        save_turn(state.user_input, response)
        state.response = response
        state.node = "greeting"
        return state

    return RunnableLambda(run)
