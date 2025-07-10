import logging
from langchain_core.runnables import RunnableLambda
from agent.vector.ado_client import ADOClient
from agent.types import ReasoningState

logger = logging.getLogger(__name__)

CONFIRM_KEYWORDS_STORY = [
    "log it", "submit", "submit story", "log story", "create story",
    "raise story", "file story", "raise ticket", "log ticket", "add as story",
    "make a story", "please file a story", "new story"
]

def story_submission_node():
    def handle(state: ReasoningState) -> ReasoningState:
        logger.info(f"[StorySubmission] called with intent='{state.intent}' | user_input='{state.user_input}'")

        # Only allow for correct intent and present template
        if state.intent != "story_log" or not state.story_template:
            logger.warning(f"[StorySubmission] Invalid state: intent={state.intent}, story_template={state.story_template}")
            return state

        user_reply = state.user_input.strip().lower()
        if not any(k in user_reply for k in CONFIRM_KEYWORDS_STORY):
            logger.info("[StorySubmission] No submit confirmation found in user reply.")
            state.response = (
                "To submit this story, please confirm by saying something like 'log it' or 'submit story'."
            )
            return state

        state.node = "story_submission_node"
        tpl = state.story_template or {}

        # Apply normalization/defaults
        title = tpl.get("title", "").strip() or "Untitled Story"
        description = tpl.get("description", "").strip() or "No description provided"
        criteria = tpl.get("acceptance_criteria", "").strip() or "N/A"
        points_raw = tpl.get("story_points", 1)
        try:
            story_points = float(points_raw)
        except Exception:
            story_points = 1

        # ADO field mapping
        fields = {
            "System.Title": title,
            "System.Description": description,
            "Microsoft.VSTS.Common.AcceptanceCriteria": criteria,
            "Microsoft.VSTS.Scheduling.StoryPoints": story_points,
        }

        logger.info(f"[StorySubmission] Submitting to ADO: fields={fields}")
        try:
            client = ADOClient()
            result = client.create_work_item("User Story", fields)
            state.response = (
                f"✅ Story successfully logged in Azure DevOps!\n"
                f"• ID: {result.get('id')}\n"
                f"• Title: {result.get('title')}\n"
                f"• Link: {result.get('url') or 'N/A'}"
            )
            state.story_template = None  # Clear for next session!
            logger.info(f"[StorySubmission] ADO create success: {result}")
        except Exception as e:
            state.response = (
                "❌ Failed to submit the story to ADO. Please try again later or contact support.\n"
                f"Error: {e}\n"
                f"Story details: {tpl}"
            )
            logger.error(f"[StorySubmission] ADO create failed: {e}")

        return state

    return RunnableLambda(handle)
