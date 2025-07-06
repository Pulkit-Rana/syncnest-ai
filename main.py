import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# Import routers
from agent.api.memory import router as memory_router
from agent.api.debug import router as debug_router
from agent.api.agent_reasoned import router as reasoned_router
from agent.api.qdrant_debug import router as qdrant_debug_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO
    # Initialize resources (e.g., Redis connection, telemetry clients)
    # Example: await init_redis()
    yield
    # Clean up resources (e.g., close DB connections)
    # Example: await close_redis()

app = FastAPI(title="AI Reasoning Agent", lifespan=lifespan)

# CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Exception Handlers ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )

# Register endpoints under /chat
app.include_router(memory_router, prefix="/chat")
app.include_router(debug_router, prefix="/chat")
app.include_router(reasoned_router, prefix="/chat")
app.include_router(qdrant_debug_router, prefix="/chat")

# Placeholder for streaming endpoints:
# Define StreamingResponse endpoints in your router modules using the "yield" pattern,
# then mount them here or dynamically include a streaming router if needed.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=bool(os.getenv("DEV_MODE", True)),
    )
