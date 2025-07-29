from langchain_core.runnables import RunnableLambda
from agent.types import ReasoningState
from agent.memory.memory import save_turn

def greeting_node():
    def run(state: ReasoningState) -> ReasoningState:
        # Set thought for reasoning stream
        state.thought = "Detected greeting. Preparing welcome message for the user."
        
        response = (
            "Hello! 👋 Welcome to the Product Support Assistant. "
            "I can help you with questions about the app's features and workflows, "
            "troubleshoot issues, and log bugs or user stories in Azure DevOps. "
            "What can I do for you today?"
        )
        save_turn(state.user_input, response)
        state.response = response
        state.node = "greeting"
        return state

    return RunnableLambda(run)
