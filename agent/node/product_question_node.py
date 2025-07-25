import os
from langchain_core.runnables import RunnableLambda
from agent.utils.llm_response import call_llm
from agent.types import ReasoningState
from agent.vector.ado_client import ADOClient
from agent.vector.qdrant_client import search_similar

YES_KEYWORDS = [
    "yes", "show me", "details", "see it", "more info", "see details", "show details", "yep", "of course", "log", "log it", "please log", "create bug", "file bug", "new bug", "add bug"
]
BUG_KEYWORDS = ["bug", "issue", "defect", "error", "not working", "fail", "unable"]
STORY_KEYWORDS = ["story", "feature", "enhancement", "request"]

SIMILARITY_THRESHOLD = 0.93

def product_question_node():
    def handle(state: ReasoningState) -> ReasoningState:
        user_input = state.user_input.strip()
        history = state.history or ""
        user_reply = user_input.lower()
        session_last = getattr(state, "last_entity", None)

        # 1. YES/DETAILS follow-up for last_entity
        if any(kw in user_reply for kw in YES_KEYWORDS) and session_last:
            entity = session_last
            state.node = "product_question"
            state.intent = "product_question"
            state.response = (
                f"Here are the details for '{entity.get('title', '')}' (ID: {entity.get('id', '')}):\n"
                f"Type: {entity.get('work_item_type', 'Item')}\n"
                f"Status: {entity.get('status', 'Unknown')}\n"
                f"Description: {entity.get('description', '') or 'No further description available.'}\n"
                "Would you like to log a new bug or story about this, update it, or ask something else?"
            )
            return state

        # 2. Strong vector match
        semantic_results = search_similar(user_input, top_k=5)
        state.ado_context = semantic_results if isinstance(semantic_results, list) else []
        most_similar = None
        user_words = set(w.lower() for w in user_input.split() if len(w) > 2)
        if semantic_results:
            for item in semantic_results:
                sim = item.get('similarity', 0)
                title = item.get("title", "").lower()
                title_words = set(w for w in title.split())
                if sim >= SIMILARITY_THRESHOLD or (user_words and user_words.issubset(title_words)):
                    most_similar = item
                    break

        if most_similar:
            state.last_entity = most_similar
            state.node = "product_question"
            title = most_similar.get("title", "")
            entity_id = most_similar.get("id", "")
            status = most_similar.get("status", "Unknown")
            entity_type = most_similar.get("work_item_type", "Item")
            state.response = (
                f"It looks like a similar {entity_type.lower()} already exists:\n"
                f"• Title: {title}\n"
                f"• Status: {status}\n"
                f"• ID: {entity_id}\n"
                "Would you like to see more details, update this, or log a new one anyway?"
            )
            return state

        # 3. ADO keyword match (strict)
        ADO_ORG = os.environ.get("ADO_ORGANIZATION")
        ADO_PROJECT = os.environ.get("ADO_PROJECT")
        ADO_PAT = os.environ.get("ADO_PAT")
        ado_client = ADOClient(ADO_ORG, ADO_PROJECT, ADO_PAT)
        ado_results = ado_client.search_stories(user_input, top_k=5)
        # PATCH: flatten ado_context as a list
        if ado_results and isinstance(ado_results, dict):
            combined = []
            for k in ['stories', 'bugs', 'features', 'wikis']:
                combined.extend(ado_results.get(k, []))
            state.ado_context = combined
        else:
            state.ado_context = []
        found_match = None
        for item in state.ado_context:
            title_words = set(w.lower() for w in item.get("title", "").split())
            if user_words and user_words.issubset(title_words):
                found_match = item
                break
        if found_match:
            state.last_entity = found_match
            state.node = "product_question"
            title = found_match.get("title", "")
            entity_id = found_match.get("id", "")
            status = found_match.get("status", "Unknown")
            entity_type = found_match.get("work_item_type", "Item")
            state.response = (
                f"A similar {entity_type.lower()} already exists in Azure DevOps:\n"
                f"• Title: {title}\n"
                f"• Status: {status}\n"
                f"• ID: {entity_id}\n"
                "Would you like to see more details, update this, or log a new one anyway?"
            )
            return state

        # 4. No match found: offer to log as bug/story, or answer with LLM using context if any exists
        is_bug = any(kw in user_reply for kw in BUG_KEYWORDS)
        is_story = any(kw in user_reply for kw in STORY_KEYWORDS)
        context_blocks = []
        if semantic_results:
            for item in semantic_results:
                src = item.get("source", "")
                if src == "work_item":
                    block = (
                        f"WORK ITEM:\nTitle: {item.get('title', '')} (ID: {item.get('id', '')}, Type: {item.get('work_item_type', '')})\n"
                        f"Description: {item.get('description', '')}"
                    )
                elif src == "wiki":
                    block = (
                        f"WIKI PAGE:\nTitle: {item.get('title', '')}\nExcerpt: {item.get('description', '')[:700]}"
                    )
                else:
                    block = f"OTHER:\n{item}"
                context_blocks.append(block)
        semantic_context = "\n\n".join(context_blocks) or "No relevant work items, bugs, stories, or wiki pages were found."

        # LLM prompt
        prompt = (
            "You are a highly skilled, empathetic AI product specialist for this web application. "
            "Use the CONTEXT to answer user questions or requests, or offer to log a new bug/story if nothing relevant is found.\n"
            "Be specific and helpful. If you are unsure, clarify or ask for more info, but always offer the next step.\n\n"
            f"Chat so far:\n{history}\n\n"
            f"User's latest question:\n{user_input}\n\n"
            f"---\nCONTEXT:\n{semantic_context}\n---"
        )

        answer = call_llm(prompt)
        state.node = "product_question"

        if answer and answer.strip() and "not found" not in answer.lower():
            state.response = (
                answer.strip() +
                ("\n\nWould you like me to log this as a bug?" if is_bug else "") +
                ("\n\nWould you like me to log this as a user story?" if is_story else "") +
                ("\n\nOr would you like to clarify, edit, or ask something else?")
            )
            return state

        state.response = (
            "I couldn't find any relevant existing bugs, stories, or docs for your request.\n"
            "Would you like to log this as a new bug or story? If yes, just say 'log bug' or 'log story' and I'll build a template for you."
        )
        state.intent = "clarify"
        return state

    return RunnableLambda(handle)
