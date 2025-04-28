import os, openai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
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
    """
    supabase-py 2.x returns a Postgrest Response that *no longer*
    exposes .status_code / .error.  
    Instead we trust that .execute() raised no exception, then pull
    the row id from whatever structure came back.
    """
    if hasattr(res, "data") and res.data:
        row = res.data[0]
    elif isinstance(res, list) and res:
        row = res[0]
    else:
        return None
    return row.get("id") if isinstance(row, dict) else None

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
    res = sb.table("messages").insert(payload.dict()).execute()
    row_id = _extract_id(res)
    if row_id is None:
        raise HTTPException(500, "Insert failed (no id returned)")
    return {"inserted": True, "id": row_id}

@app.post("/embed-save")
async def embed_and_save(body: dict):
    text    = body["text"]
    part    = body["part"]
    chapter = body["chapter"]

    # 1️⃣  OpenAI embedding
    emb = openai.embeddings.create(
        model=os.environ["OPENAI_MODEL_EMBED"],
        input=text,
    ).data[0].embedding          # list[float]

    # 2️⃣  list → pgvector literal
    vect_str = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"

    # 3️⃣  insert into Supabase
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

    row_id = _extract_id(res)
    if row_id is None:
        raise HTTPException(500, "Insert failed (no id returned)")
    return {"inserted": True, "id": row_id}
