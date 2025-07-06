import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

# ✅ Shared LLM endpoint
_llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id=os.getenv("LLM_MODEL"),
        token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        task="text-generation",
        temperature=0.5,
        max_new_tokens=256,
    )
)

def call_llm(user_input) -> str:
    """
    Takes a list of formatted ChatMessages (e.g., from ChatPromptTemplate).
    Returns cleaned response.
    """
    return _llm.invoke(user_input).content.strip()
