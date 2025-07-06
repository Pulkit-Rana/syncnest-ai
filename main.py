from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from agent.api.memory import router as memory_router
from agent.api.debug import router as debug_router
from agent.api.agent_reasoned import router as reasoned_router
from agent.api.qdrant_debug import router as qdrant_debug_router


app = FastAPI(title="AI Reasoning Agent")

# ✅ Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Register endpoints
app.include_router(memory_router, prefix="/chat")     
app.include_router(debug_router, prefix="/chat")      
app.include_router(reasoned_router, prefix="/chat")
app.include_router(qdrant_debug_router, prefix="/chat")  