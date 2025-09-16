from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from fastapi.staticfiles import StaticFiles
import pandas as pd
import re
import unicodedata

# ---- Import your logic modules ----
# Adjust function names after you confirm them in your files.
try:
    from services.text_to_gloss_translation import translate_to_gloss as _translate_to_gloss
except Exception:
    print(">>> Translator loaded?", _translate_to_gloss is not None)
    _translate_to_gloss = None

try:
    from services.ASLsimilarity import find_similar as _find_similar
except Exception:
    print(">>> Translator loaded?", _translate_to_gloss is not None)
    _find_similar = None

app = FastAPI(title="rhymASL API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).parent / "data"
LEX_PATH = DATA_DIR / "ASL-LEX.csv"
LEX_ADD_PATH = DATA_DIR / "ASL-LEX_add_translation.csv"

lex_df = None
lex_add_df = None
if LEX_PATH.exists():
    try:
        lex_df = pd.read_csv(LEX_PATH)
    except Exception as e:
        print("Failed to load ASL-LEX.csv:", e)
if LEX_ADD_PATH.exists():
    try:
        lex_add_df = pd.read_csv(LEX_ADD_PATH)
    except Exception as e:
        print("Failed to load ASL-LEX_add_translation.csv:", e)

class GlossRequest(BaseModel):
    text: str

class SimilarityRequest(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"status": "ok", "lex_loaded": bool(lex_df is not None)}

@app.post("/gloss")
def gloss(req: GlossRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text required")
    if _translate_to_gloss is None:
        # Fallback placeholder
        return {"input": req.text, "gloss": req.text.upper().replace(" ", "-")}
    try:
        result = _translate_to_gloss(req.text)
        return {"input": req.text, "gloss": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/similarity")
def similarity(req: SimilarityRequest):
    if _find_similar is None:
        # Safe fallback if your real function isn’t wired yet
        sample = (lex_df.head(5).to_dict(orient="records") if lex_df is not None else [])
        return {"query": req.query, "results": sample}
    try:
        result = _find_similar(req.query, lex_df, lex_add_df)
        return {"query": req.query, "results": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
app.mount("/videos", StaticFiles(directory="ASL_LEX_MP4"), name="videos")
