"""
Authentication module for Air Quality Forecasting app.

JWT-based auth with bcrypt password hashing.
"""

import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from database import (
    create_user,
    get_user_by_email,
    get_user_by_id,
    get_user_by_username,
)

# JWT Configuration — persist secret so it survives restarts during dev
_SECRET_FILE = os.path.join(os.path.dirname(__file__), ".jwt_secret")

def _load_or_create_secret() -> str:
    env = os.environ.get("JWT_SECRET")
    if env:
        return env
    if os.path.exists(_SECRET_FILE):
        with open(_SECRET_FILE, "r") as f:
            return f.read().strip()
    secret = secrets.token_hex(32)
    with open(_SECRET_FILE, "w") as f:
        f.write(secret)
    return secret

JWT_SECRET = _load_or_create_secret()
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 72

security = HTTPBearer(auto_error=False)

# Validation
EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
USERNAME_REGEX = re.compile(r"^[a-zA-Z0-9_]{3,30}$")
MIN_PASSWORD_LENGTH = 6


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def create_token(user_id: int) -> str:
    payload = {
        "sub": str(user_id),
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def register_user(username: str, email: str, password: str) -> dict:
    """Register a new user. Returns user dict + token."""
    # Validate
    if not USERNAME_REGEX.match(username):
        raise HTTPException(
            status_code=400,
            detail="Username must be 3-30 characters, letters/numbers/underscores only",
        )
    if not EMAIL_REGEX.match(email):
        raise HTTPException(status_code=400, detail="Invalid email address")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Password must be at least {MIN_PASSWORD_LENGTH} characters",
        )

    # Check duplicates
    if get_user_by_email(email):
        raise HTTPException(status_code=409, detail="Email already registered")
    if get_user_by_username(username):
        raise HTTPException(status_code=409, detail="Username already taken")

    # Create
    pw_hash = hash_password(password)
    user_id = create_user(username, email, pw_hash)
    token = create_token(user_id)

    return {
        "user": {"id": user_id, "username": username, "email": email.lower()},
        "token": token,
    }


def login_user(email: str, password: str) -> dict:
    """Authenticate a user. Returns user dict + token."""
    user = get_user_by_email(email)
    if not user or not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_token(user["id"])
    return {
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
        },
        "token": token,
    }


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """Dependency: extract and validate the current user from JWT."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_token(credentials.credentials)
    user = get_user_by_id(int(payload["sub"]))
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[dict]:
    """Dependency: return user if authenticated, None otherwise."""
    if credentials is None:
        return None
    try:
        payload = decode_token(credentials.credentials)
        return get_user_by_id(int(payload["sub"]))
    except HTTPException:
        return None
