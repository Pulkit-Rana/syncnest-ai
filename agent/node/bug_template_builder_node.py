import json
import re
import logging
from langchain_core.runnables import RunnableLambda
from agent.utils.llm_response import call_llm
from agent.types import ReasoningState
from agent.vector.qdrant_client import search_similar

logger = logging.getLogger(__name__)

YES_KEYWORDS = [
    "yes", "show me", "details", "see it", "more info", "see details", "show details", "yep", "of course",
    "log", "log it", "please log", "create bug", "file bug", "new bug", "add bug"
]

def bug_template_builder_node():
    def handle(state: ReasoningState) -> ReasoningState:
        if state.intent != "bug_log" or state.bug_template is not None:
            return state

        user_desc = state.user_input.strip()
        user_reply = user_desc.lower()

        state.thought = "Checking if user wants details on last similar bug."
        # 1. YES/DETAILS on last_entity
        if any(kw in user_reply for kw in YES_KEYWORDS) and getattr(state, "last_entity", None):
            entity = state.last_entity
            state.thought = f"Providing details for last similar bug: {entity.get('title', '')}"
            state.response = (
                f"Here are the details for the similar bug:\n"
                f"• Title: {entity.get('title', '')}\n"
                f"• Status: {entity.get('status', '')}\n"
                f"• ID: {entity.get('id', '')}\n"
                f"Description: {entity.get('description', '') or 'No further description available.'}\n\n"
                "Would you like to log a new bug anyway? If yes, just say 'log bug' or describe your new bug."
            )
            return state

        state.thought = "Searching for similar bugs in vector database."
        # 2. Search for similar bugs
        similar = search_similar(user_desc, top_k=5)
        if similar:
            for item in similar:
                sim = item.get("similarity", 0)
                title_match = item.get("title", "").lower() in user_desc.lower()
                if sim >= 0.93 or title_match:
                    state.thought = f"Found similar bug: {item.get('title', '')} with similarity {sim:.2f}"
                    state.last_entity = item
                    state.response = (
                        f"It looks like a similar bug already exists:\n"
                        f"• Title: {item.get('title', '')}\n"
                        f"• Status: {item.get('status', '')}\n"
                        f"• ID: {item.get('id', '')}\n"
                        "Would you like to see more details, update this, or log a new bug anyway?"
                    )
                    return state

        state.thought = "Generating new bug report template using LLM."
        # 3. Build new bug template using LLM
        prompt = (
            "You are an expert QA engineer. Given the user's description, generate a clear, actionable bug report template in JSON. "
            "Do NOT ask follow-up questions. If unsure, use defaults: priority=2, severity='3 - Medium', repro_steps='No steps provided'. "
            "Reply ONLY with raw, valid JSON.\n\n"
            "Required keys: title, description, repro_steps, priority, severity."
            f"\n\nUser Description:\n{user_desc}\n\n"
            "Return ONLY the JSON object, no explanation."
        )
        keys = ["title", "description", "repro_steps", "priority", "severity"]
        result_str = call_llm(prompt).strip()

        for attempt in range(2):
            match = re.search(r'\{[\s\S]*\}', result_str)
            json_str = match.group(0) if match else result_str
            try:
                result_json = json.loads(json_str)
                # Normalize fields
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
                state.thought = "Successfully generated bug template JSON."
                state.response = (
                    "Here’s your auto-generated **bug template**. "
                    "**Reply 'log it' to submit as a bug**, or reply with any edits to update the template. "
                    "If you want to add or edit fields, just say what needs to change!\n\n"
                    + pretty
                )
                return state
            except Exception as e:
                logger.error(f"BugTemplateBuilder JSON parse failed: {e} | Output was: {result_str}")
                state.thought = "Failed to parse JSON from LLM response; retrying with stricter prompt."
                # Retry with a stricter prompt
                result_str = call_llm(
                    "Return only valid JSON for the previous bug template request. "
                    "The JSON MUST have these keys: title, description, repro_steps, priority, severity. "
                    "Use allowed default values for missing fields: priority=2, severity='3 - Medium', repro_steps='No steps provided'."
                ).strip()

        state.thought = "Failed to generate valid bug template after retries."
        state.bug_template = None
        state.response = (
            "Sorry, I couldn't auto-generate a bug report from your description right now. "
            "Would you like to provide the bug details (title, description, repro_steps, priority, severity) directly, or should I connect you to support? "
            "If you want, just reply with a quick bug description in this format:\n"
            "Title: ...\nDescription: ...\nRepro Steps: ...\nPriority: ...\nSeverity: ..."
        )
        return state

    return RunnableLambda(handle)

