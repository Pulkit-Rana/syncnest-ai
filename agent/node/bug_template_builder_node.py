import json
import re
from langchain_core.runnables import RunnableLambda
from agent.utils.llm_response import call_llm
from agent.types import ReasoningState

def bug_template_builder_node():
    def handle(state: ReasoningState) -> ReasoningState:
        if state.intent != "bug_log" or state.bug_template is not None:
            return state

        state.node = "bug_template_builder"
        user_desc = state.user_input.strip()
        context_items = []
        if getattr(state, "ado_context", None):
            context_items = [f"- {item.get('title')} (ID: {item.get('id')})" for item in state.ado_context[:5]]

        prompt = (
            "You are an expert product support engineer. Generate a bug report template in JSON using the user's exact description below. "
            "Do NOT ask any follow-up questions. Populate all fields; if unsure, use allowed default values: "
            "For priority use 2, for severity use '3 - Medium', for repro_steps use 'No steps provided'. "
            "Reply ONLY with raw, valid JSON. "
            "\n\nRequired JSON keys: title, description, repro_steps, priority, severity."
            f"\n\nUser Description:\n{user_desc}\n"
            f"\nContext:{'\n' + '\n'.join(context_items) if context_items else ' None'}\n\n"
            "Return ONLY the JSON object, no explanation."
        )

        keys = ["title", "description", "repro_steps", "priority", "severity"]
        result_str = call_llm(prompt).strip()

        for attempt in range(2):
            match = re.search(r'\{[\s\S]*\}', result_str)
            json_str = match.group(0) if match else result_str
            try:
                result_json = json.loads(json_str)
                # Normalize keys and defaults — THIS IS THE CRUCIAL PART!
                for k in keys:
                    val = result_json.get(k, "").strip() if isinstance(result_json.get(k), str) else result_json.get(k, "")
                    if not val or val == "N/A":
                        if k == "priority":
                            result_json[k] = "2"
                        elif k == "severity":
                            result_json[k] = "3 - Medium"
                        elif k == "repro_steps":
                            result_json[k] = "No steps provided"
                        else:
                            result_json[k] = "N/A"
                    else:
                        result_json[k] = val
                state.bug_template = result_json
                pretty = json.dumps(result_json, indent=2)
                state.response = (
                    "Here’s your auto-generated bug template. **Reply 'log it' to submit as a bug**, or reply with any edits to update the template:\n\n"
                    + pretty
                )
                return state
            except Exception:
                # Retry with a forceful, minimal prompt
                result_str = call_llm(
                    "Return only valid JSON for the previous bug template request. "
                    "The JSON MUST have these keys: title, description, repro_steps, priority, severity. "
                    "Use allowed default values for missing fields: priority=2, severity='3 - Medium', repro_steps='No steps provided'."
                ).strip()

        # If all fails, fallback error
        state.bug_template = None
        state.response = (
            "Sorry, I couldn't auto-generate a bug report from your description right now. "
            "Would you like to provide the bug details (title, description, repro_steps, priority, severity) directly, or should I connect you to support? "
            "If you want, just reply with a quick bug description in this format:\n"
            "Title: ...\nDescription: ...\nRepro Steps: ...\nPriority: ...\nSeverity: ..."
        )
        return state

    return RunnableLambda(handle)
