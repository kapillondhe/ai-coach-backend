
import logging
from functools import lru_cache

import jwt
from fastapi import Depends, HTTPException, Request

from app.core.config import get_settings

logger = logging.getLogger(__name__)

@lru_cache
def _jwks_client(supabase_url: str) -> "jwt.PyJWKClient":
    return jwt.PyJWKClient(f"{supabase_url}/auth/v1/.well-known/jwks.json")


def _verify(token: str, supabase_url: str, audience: str) -> str | None:
    try:
        signing_key = _jwks_client(supabase_url).get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256", "RS256"],
            audience=audience,
        )
    except jwt.PyJWTError:
        logger.info("Rejected invalid/expired Supabase JWT; proceeding as anonymous", exc_info=True)
        return None

    sub = payload.get("sub")
    return str(sub) if sub else None


def get_current_user_id(request: Request) -> str | None:
    settings = get_settings()
    if not settings.supabase_url:
        return None

    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        return None

    token = auth_header[len("bearer ") :].strip()
    if not token:
        return None

    return _verify(token, settings.supabase_url, settings.supabase_jwt_audience)


def require_user_id(user_id: str | None = Depends(get_current_user_id)) -> str:
    if user_id is None:
        raise HTTPException(status_code=401, detail="Sign-in required")
    return user_id
