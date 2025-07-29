import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

# Create endpoint and model instances (token automatically picked from env var)
llm = HuggingFaceEndpoint(
    repo_id=os.getenv("LLM_MODEL"),
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

def call_llm(messages, stream=False):
    """
    messages: List of ChatMessages (e.g., HumanMessage, SystemMessage).
    Returns LLM response content as string, or yields tokens if stream=True.
    """
    if not stream:
        response = model.invoke(messages)
        return response.content.strip()
    else:
        for chunk in model.stream(messages):
            yield chunk.content 

