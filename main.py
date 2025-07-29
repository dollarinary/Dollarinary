from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from trainable_ai import HybridDrawingAI

app = FastAPI()
ai = HybridDrawingAI()

# Enable CORS for any frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/suggest")
def suggest(prompt: str = Query(..., description="Prompt for drawing suggestion")):
    response = ai.suggest(prompt)
    return {"response": response}
