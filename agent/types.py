from pydantic import BaseModel, Field
from typing import List, Any, Dict, Optional

class ReasoningState(BaseModel):
    user_input: str
    type: str = ""
    intent: str = ""
    node: str = "" 
    context: List[Any] = Field(default_factory=list)
    response: str = ""
    history: str = ""
    ado_context: Optional[List[Dict[str, Any]]] = None
    web_result: Optional[Any] = None
    bug_template: Optional[Dict[str, Any]] = None
