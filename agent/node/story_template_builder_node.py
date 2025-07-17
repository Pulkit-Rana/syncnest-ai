import json
import re
import logging
# from langchain_core.runnables import RunnableLambda
from agent.utils.llm_response import call_llm
from agent.types import ReasoningState
from agent.vector.qdrant_client import search_similar

logger = logging.getLogger(__name__)

YES_KEYWORDS = [
    "yes", "show me", "details", "see it", "more info", "see details", "show details", "yep", "of course", "log", "log it", "please log", "create story", "file story", "new story", "add story"
]

def story_template_builder_node():
    def handle(state: ReasoningState) -> ReasoningState:
        # --- Only build if correct intent and no template yet ---
        if state.intent != "story_log" or state.story_template is not None:
            return state

        user_desc = state.user_input.strip()
        user_reply = user_desc.lower()

        # === 1. YES/DETAILS/SHOW on last_entity ===
        if any(kw in user_reply for kw in YES_KEYWORDS) and getattr(state, "last_entity", None):
            entity = state.last_entity
            state.response = (
                f"Here are the details for the similar story:\n"
                f"• Title: {entity.get('title', '')}\n"
                f"• Status: {entity.get('status', '')}\n"
                f"• ID: {entity.get('id', '')}\n"
                f"Description: {entity.get('description', '') or 'No further description available.'}\n\n"
                "Would you like to log a new story anyway? If yes, just say 'log story' or describe your new story."
            )
            return state

        # --- 2. Search for similar stories ---
        similar = search_similar(user_desc, top_k=5)
        if similar:
            for item in similar:
                # Stricter similarity threshold to avoid false positives
                sim = item.get("similarity", 0)
                title_match = item.get("title", "").lower() in user_desc.lower()
                if sim >= 0.93 or title_match:
                    state.last_entity = item
                    state.response = (
                        f"It looks like a similar story already exists:\n"
                        f"• Title: {item.get('title', '')}\n"
                        f"• Status: {item.get('status', '')}\n"
                        f"• ID: {item.get('id', '')}\n"
                        "Would you like to see more details, update this, or log a new story anyway?"
                    )
                    return state

        # --- 3. If not, build new story template using LLM ---
        prompt = (
            "You are an expert product manager. Given the user's description, generate a clear, actionable user story template in JSON. "
            "Do NOT ask follow-up questions. If unsure, use defaults: acceptance_criteria='N/A', story_points=1. "
            "Reply ONLY with raw, valid JSON.\n\n"
            "Required keys: title, description, acceptance_criteria, story_points."
            f"\n\nUser Description:\n{user_desc}\n\n"
            "Return ONLY the JSON object, no explanation."
        )
        keys = ["title", "description", "acceptance_criteria", "story_points"]
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
                        if k == "story_points":
                            result_json[k] = 1
                        else:
                            result_json[k] = "N/A"
                    elif k == "story_points":
                        try:
                            result_json[k] = int(val)
                        except Exception:
                            result_json[k] = 1
                    else:
                        result_json[k] = val
                state.story_template = result_json
                pretty = json.dumps(result_json, indent=2)
                state.response = (
                    "Here’s your auto-generated **story template**. "
                    "**Reply 'log it' to submit as a story**, or reply with any edits to update the template. "
                    "If you want to add or edit fields, just say what needs to change!\n\n"
                    + pretty
                )
                return state
            except Exception as e:
                logger.error(f"StoryTemplateBuilder JSON parse failed: {e} | Output was: {result_str}")
                # Retry with a stricter prompt
                result_str = call_llm(
                    "Return only valid JSON for the previous story template request. "
                    "The JSON MUST have these keys: title, description, acceptance_criteria, story_points. "
                    "Use allowed default values: acceptance_criteria='N/A', story_points=1."
                ).strip()

        state.story_template = None
        state.response = (
            "Sorry, I couldn't auto-generate a story from your description right now. "
            "Would you like to provide the story fields directly instead?\n\n"
            "**Title:**\n**Description:**\n**Acceptance Criteria:**\n**Story Points:**"
        )
        return state

    return handle
