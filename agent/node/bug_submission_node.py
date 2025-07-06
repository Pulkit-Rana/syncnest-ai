from langchain_core.runnables import RunnableLambda
from agent.vector.ado_client import ADOClient
from agent.types import ReasoningState

CONFIRM_KEYWORDS = [
    "log it", "submit", "submit bug", "log bug", "create bug", "raise bug", "file bug", "raise ticket", "log ticket"
]

def bug_submission_node():
    def handle(state: ReasoningState) -> ReasoningState:
        if state.intent != "bug_log" or not state.bug_template:
            return state

        user_reply = state.user_input.strip().lower()
        if not any(k in user_reply for k in CONFIRM_KEYWORDS):
            return state

        state.node = "bug_submission_node"
        tpl = state.bug_template

        # Apply defaults here as well — edge-case double safety
        priority = tpl.get("priority", "2")
        try:
            priority_int = int(priority)
        except Exception:
            priority_int = 2  # fallback

        severity = tpl.get("severity", "3 - Medium")
        if not severity or severity == "N/A":
            severity = "3 - Medium"

        # Map to ADO field names
        fields = {
            "System.Title": tpl.get("title", ""),
            "System.Description": tpl.get("description", ""),
            "Microsoft.VSTS.TCM.ReproSteps": tpl.get("repro_steps", "No steps provided"),
            "Microsoft.VSTS.Common.Priority": priority_int,
            "Microsoft.VSTS.Common.Severity": severity,
        }

        try:
            client = ADOClient()
            result = client.create_work_item("Bug", fields)
            state.response = (
                f"✅ Bug successfully logged in Azure DevOps!\n"
                f"• ID: {result.get('id')}\n"
                f"• Title: {result.get('title')}\n"
                f"• Link: {result.get('url') or 'N/A'}"
            )
            state.bug_template = None
        except Exception as e:
            state.response = (
                "❌ Failed to submit the bug to ADO. Please try again later or contact support.\n"
                f"Error: {e}\n"
                f"Bug details: {tpl}"
            )
        return state

    return RunnableLambda(handle)
