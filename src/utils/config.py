from __future__ import annotations
import os
from dotenv import load_dotenv, find_dotenv

def load_env() -> None:
    """
    Load environment variables from .env and print where it was loaded from.
    Works in Codespaces, Docker, Render/Railway.
    """
    # Allow override via ENV_FILE; otherwise auto-detect
    env_path = os.environ.get("ENV_FILE") or find_dotenv(usecwd=True)
    if env_path and os.path.exists(env_path):
        load_dotenv(env_path, override=False)
        print(f"Loaded environment from {env_path}")
    else:
        load_dotenv(override=False)
        print("Loaded environment from .env (default search)")
