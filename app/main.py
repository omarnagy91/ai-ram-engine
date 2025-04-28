import os
import openai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .db import sb
from .schemas import SavePayload

# ─────────── ENV + OpenAI
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

# ─────────── Routes
@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/debug/env")
async def debug_env():
    svc = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "MISSING")[:20]
    return {"service_key": svc}


@app.post("/save")
async def save(payload: SavePayload):
    """Insert already-computed embedding row."""
    res = sb.table("messages").insert(payload.dict()).execute()

    if res.status_code >= 400 or res.data is None:
        raise HTTPException(500, str(res.data))

    return {"inserted": True, "id": res.data[0]["id"]}


@app.post("/embed-save")
async def embed_and_save(body: dict):
    text    = body["text"]
    part    = body["part"]
    chapter = body["chapter"]

    # 1️⃣  Create embedding
    emb_resp = openai.embeddings.create(
        model=os.environ["OPENAI_MODEL_EMBED"],
        input=text,
    )
    vector   = emb_resp.data[0].embedding        # list[float]

    # 2️⃣  pgvector literal → "[0.123, …]"
    vect_str = "[" + ",".join(f"{x:.6f}" for x in vector) + "]"

    # 3️⃣  Insert row
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

    if res.status_code >= 400 or res.data is None:
        raise HTTPException(500, str(res.data))

    return {"inserted": True, "id": res.data[0]["id"]}
