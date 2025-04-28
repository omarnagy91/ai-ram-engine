from pydantic import BaseModel, conlist

class SavePayload(BaseModel):
    text: str
    part: int
    chapter: int
    embedding: conlist(float, min_length=1536, max_length=1536)
