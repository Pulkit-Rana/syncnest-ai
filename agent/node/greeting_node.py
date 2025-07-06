from langchain_core.runnables import RunnableLambda
from agent.types import ReasoningState
from agent.memory.memory import save_turn

def greeting_node():
    def run(state: ReasoningState) -> ReasoningState:
        response = (
            "Hi there! 👋 It's great to see you. How can I assist you today? "
            "Feel free to ask about app issues, workflows, or anything DevOps-related."
        )
        save_turn(state.user_input, response)
        state.response = response
        state.node = "greeting" 
        return state

    return RunnableLambda(run)
