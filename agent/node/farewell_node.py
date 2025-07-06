from langchain_core.runnables import RunnableLambda
from agent.types import ReasoningState
from agent.memory.memory import save_turn

def farewell_node():
    def run(state: ReasoningState) -> ReasoningState:
        response = "Thanks for stopping by! 👋 If you need help later, just ping me anytime."
        save_turn(state.user_input, response)
        state.response = response
        return state
 
    return RunnableLambda(run)
