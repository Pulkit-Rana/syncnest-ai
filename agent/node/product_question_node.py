import os
from langchain_core.runnables import RunnableLambda
from agent.utils.llm_response import call_llm
from agent.types import ReasoningState
from agent.vector.ado_client import ADOClient
from agent.vector.qdrant_client import search_similar

def product_question_node():
    def handle(state: ReasoningState) -> ReasoningState:
        user_input = state.user_input
        history = state.history or ""

        ADO_ORG = os.environ.get("ADO_ORGANIZATION")
        ADO_PROJECT = os.environ.get("ADO_PROJECT")
        ADO_PAT = os.environ.get("ADO_PAT")

        # Set node name early for traceability
        state.node = "product_question"
        state.intent = "product_question"

        if not (ADO_ORG and ADO_PROJECT and ADO_PAT):
            state.response = (
                "ADO configuration missing in environment. Please contact the administrator."
            )
            return state

        # --- 1. Semantic Search (Qdrant vector DB) ---
        semantic_results = search_similar(user_input, top_k=5)  # richer synthesis
        state.ado_context = semantic_results

        if semantic_results:
            context_blocks = []
            for item in semantic_results:
                src = item.get("source", "")
                if src == "work_item":
                    block = (
                        f"WORK ITEM:\n"
                        f"Title: {item.get('title', '')} (ID: {item.get('id', '')}, Type: {item.get('type', '')})\n"
                        f"Description: {item.get('description', '')}"
                    )
                elif src == "wiki":
                    block = (
                        f"WIKI PAGE:\n"
                        f"Title: {item.get('title', '')}\n"
                        f"Excerpt: {item.get('description', '')[:700]}"
                    )
                else:
                    block = f"OTHER:\n{item}"
                context_blocks.append(block)
            semantic_context = "\n\n".join(context_blocks)

            # --- ELITE PROMPT ---
            prompt = (
                "You are a highly skilled AI product specialist for this web application, "
                "acting as a bridge between the user and the development team.\n\n"
                "When a user reports an issue or asks a question:\n"
                "- Read the WORK ITEMS and WIKI PAGES context below. Do NOT simply repeat the text.\n"
                "- **Diagnose the problem** using context — make educated guesses, relate requirements, anticipate pain points.\n"
                "- If the answer or fix is clear, explain it directly. Otherwise, ask concise, targeted follow-up questions as an expert engineer/support would.\n"
                "- If a known workaround/feature is documented, explain it and cite the relevant doc/story by title.\n"
                "- If context is missing/unclear, say so, and suggest the user provide more info or offer to escalate/log a bug/feature.\n"
                "- Always be specific, empathetic, and solution-oriented.\n"
                "- Close with a clear next step for user or agent.\n\n"
                "Below is the chat and product context. Your response should be actionable, concise, and human.\n\n"
                f"Chat so far:\n{history}\n\n"
                f"User's latest question:\n{user_input}\n\n"
                f"---\nCONTEXT:\n{semantic_context}\n---"
            )

            answer = call_llm(prompt)
            state.response = answer.strip()
            return state

        # --- 2. Classic ADO Keyword Search as Fallback ---
        ado_client = ADOClient(ADO_ORG, ADO_PROJECT, ADO_PAT)
        ado_results = ado_client.search_stories(user_input, top_k=5)
        state.ado_context = ado_results

        work_items = [item for item in ado_results if item.get("source") == "work_item"]
        wiki_pages = [item for item in ado_results if item.get("source") == "wiki"]

        if work_items or wiki_pages:
            context_blocks = []
            for item in work_items:
                block = (
                    f"WORK ITEM:\n"
                    f"Title: {item['title']} (ID: {item['id']}, Type: {item.get('work_item_type', '')})\n"
                    f"Description: {item.get('description', '')}"
                )
                context_blocks.append(block)
            for item in wiki_pages:
                block = (
                    f"WIKI PAGE:\n"
                    f"Title: {item['title']}\n"
                    f"Excerpt: {item.get('description', '')[:700]}"
                )
                context_blocks.append(block)
            classic_context = "\n\n".join(context_blocks)

            prompt = (
                "You are a highly skilled AI product specialist for this web application.\n\n"
                "- Use the WORK ITEMS and WIKI PAGES below to diagnose and answer the user's question.\n"
                "- Don't just repeat content — reason through the context, connect details, and give actionable advice.\n"
                "- If docs/stories cover the issue, explain clearly and cite sources. If not, ask concise follow-up questions or suggest escalation/logging as a bug/feature.\n"
                "- Always close with a specific next step for the user or agent.\n\n"
                f"Chat so far:\n{history}\n\n"
                f"User's latest question:\n{user_input}\n\n"
                f"---\nCONTEXT:\n{classic_context}\n---"
            )

            answer = call_llm(prompt)
            state.response = answer.strip()
            return state

        # --- 3. Nothing found anywhere ---
        state.response = (
            "I couldn't find any relevant user story, bug, or documentation in ADO or the Wiki for your question. "
            "Would you like me to help you log this as a new bug or feature request, or do you want to provide more details? "
            "Let me know how you’d like to proceed, and I’ll guide you further."
        )
        return state

    return RunnableLambda(handle)
