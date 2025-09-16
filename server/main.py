from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import re

BASE_DIR = Path(__file__).parent
VIDEOS_DIR = BASE_DIR / "static" / "signs"

# Optional: your real translator if available
_translate_to_gloss = None
try:
    from services.text_to_gloss_translation import translate_to_gloss as _translate_to_gloss
except Exception as e:
    print(f">>> translate_to_gloss not loaded: {e}")

app = FastAPI(title="rhymASL API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GlossRequest(BaseModel):
    text: str

def token_to_file(token: str) -> str | None:
    """Hard-code: map TOKEN -> L_<token>.mp4 (lowercased)."""
    fname = f"L_{token.lower()}.mp4"
    return fname if (VIDEOS_DIR / fname).exists() else None

@app.get("/health")
def health():
    return {
        "status": "ok",
        "gloss_fn_loaded": _translate_to_gloss is not None,
        "videos_dir": str(VIDEOS_DIR),
        "videos_dir_exists": VIDEOS_DIR.exists(),
    }

@app.post("/gloss")
def gloss(req: GlossRequest):
    txt = req.text.strip()
    if not txt:
        raise HTTPException(status_code=400, detail="text required")
    try:
        g = _translate_to_gloss(txt) if _translate_to_gloss else txt.upper().replace(" ", " ")
        # tokens: split on spaces, hyphens, underscores
        tokens = [t for t in re.split(r"[ \-\_]+", g) if t]
        files, missing = [], []
        for tok in tokens:
            f = token_to_file(tok)
            (files if f else missing).append(f or tok)
        return {"input": txt, "gloss": g, "videos": files, "missing": missing}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static videos
if VIDEOS_DIR.exists():
    app.mount("/videos", StaticFiles(directory=VIDEOS_DIR), name="videos")
    print(f">>> Serving videos from: {VIDEOS_DIR}")
else:
    print(f"WARNING: Videos directory not found at {VIDEOS_DIR}.")
