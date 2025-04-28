import os
import openai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .db import sb
from .schemas import SavePayload

# ─────────── ENV / OpenAI
load_dotenv(".env.local", override=True)
openai.api_key = os.environ["OPENAI_API_KEY"]

app = FastAPI(title="RAM Engine")

# ─────────── CORS
origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# ─────────── helpers
def _extract_id(res) -> str | None:
    if hasattr(res, "data") and res.data:
        row = res.data[0]
    elif isinstance(res, list) and res:
        row = res[0]
    else:
        return None
    return row.get("id") if isinstance(row, dict) else None

# ─────────── Request schemas
class EmbedPayload(BaseModel):
    text: str = Field(..., description="The chat block text to embed & save")
    newpart: bool = Field(False, description="If true, force a new part; else continue same part")

# ─────────── Routes

@app.get("/health")
async def health():
    return {"ok": True}

@app.get("/debug/env")
async def debug_env():
    sk = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    return {"service_key": sk[:20] or "MISSING"}

@app.post("/save")
async def save(payload: SavePayload):
    # legacy save endpoint (expects full payload, incl. part/chapter)
    res = sb.table("messages").insert(payload.dict()).execute()
    row_id = _extract_id(res)
    if row_id is None:
        raise HTTPException(500, "Insert failed (no id returned)")
    return {"inserted": True, "id": row_id}

@app.post("/embed-save")
async def embed_and_save(payload: EmbedPayload):
    text = payload.text

    # 1️⃣ Fetch the last row's part & chapter
    try:
        last = (
            sb.table("messages")
            .select("part,chapter")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        last_row = None
        if hasattr(last, "data") and last.data:
            last_row = last.data[0]
        elif isinstance(last, list) and last:
            last_row = last[0]
    except Exception as e:
        raise HTTPException(500, f"Error fetching last part/chapter: {e}")

    # 2️⃣ Compute the new part & chapter
    if last_row:
        prev_part = last_row["part"]
        prev_chap = last_row["chapter"]
    else:
        prev_part = 0
        prev_chap = 0

    if payload.newpart:
        part = prev_part + 1
        chapter = 1
    else:
        if prev_part == 0:
            part, chapter = 1, 1
        else:
            part = prev_part
            chapter = prev_chap + 1

    # 3️⃣ Generate embedding
    try:
        emb_resp = openai.embeddings.create(
            model=os.environ["OPENAI_MODEL_EMBED"],
            input=text,
        )
        emb = emb_resp.data[0].embedding
    except Exception as e:
        raise HTTPException(502, f"Embedding failed: {e}")

    # 4️⃣ Convert to pgvector literal
    vect_str = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"

    # 5️⃣ Insert into Supabase
    try:
        res = (
            sb.table("messages")
            .insert({
                "text": text,
                "part": part,
                "chapter": chapter,
                "embedding": vect_str,
            })
            .execute()
        )
    except Exception as e:
        raise HTTPException(500, f"Supabase insert failed: {e}")

    row_id = _extract_id(res)
    if row_id is None:
        raise HTTPException(500, "Insert failed (no id returned)")

    return {"inserted": True, "id": row_id, "part": part, "chapter": chapter}
