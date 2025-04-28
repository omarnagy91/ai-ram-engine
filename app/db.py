import os
from dotenv import load_dotenv
from supabase_py import create_client

load_dotenv(".env.local", override=True)

sb = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"]
)
