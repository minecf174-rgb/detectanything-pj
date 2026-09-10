from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
from secrets import token_urlsafe
from uuid import uuid4

import jwt
from fastapi import HTTPException, status
from jwt import InvalidTokenError
from pwdlib import PasswordHash
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config.settings import get_settings
from app.modules.auth.models import AuthSession
from app.modules.auth.schemas import ChangePasswordRequest, LoginRequest, RegisterRequest
from app.modules.projects.models import User


password_hasher = PasswordHash.recommended()


class AuthService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.settings = get_settings()

    def register(self, data: RegisterRequest) -> tuple[User, dict]:
        email = data.email.lower()
        existing = self.db.scalar(select(User).where(func.lower(User.email) == email))
        if existing is not None:
            if existing.password_hash is not None or not self.settings.auth_allow_legacy_account_claim:
                raise HTTPException(status_code=409, detail="Email already exists")
            existing.password_hash = password_hasher.hash(data.password)
            existing.email = email
            existing.username = data.username or existing.username
            existing.is_active = True
            user = existing
        else:
            user = User(
                email=email,
                username=data.username,
                password_hash=password_hasher.hash(data.password),
                is_active=True,
            )
            self.db.add(user)
        try:
            self.db.flush()
            tokens = self._create_session_and_tokens(user)
            self.db.commit()
        except IntegrityError as exc:
            self.db.rollback()
            raise HTTPException(status_code=409, detail="Email already exists") from exc
        self.db.refresh(user)
        return user, tokens

    def login(self, data: LoginRequest) -> tuple[User, dict]:
        user = self.db.scalar(select(User).where(func.lower(User.email) == data.email.lower()))
        if user is None or user.password_hash is None:
            raise self._invalid_credentials()
        if not user.is_active:
            raise HTTPException(status_code=403, detail="User account is disabled")
        if not password_hasher.verify(data.password, user.password_hash):
            raise self._invalid_credentials()
        user.last_login_at = datetime.now(timezone.utc)
        tokens = self._create_session_and_tokens(user)
        self.db.commit()
        self.db.refresh(user)
        return user, tokens

    def refresh(self, refresh_token: str) -> tuple[User, dict]:
        token_hash = self._hash_token(refresh_token)
        auth_session = self.db.scalar(
            select(AuthSession).where(AuthSession.token_hash == token_hash)
        )
        now = datetime.now(timezone.utc)
        if (
            auth_session is None
            or auth_session.revoked_at is not None
            or self._as_utc(auth_session.expires_at) <= now
        ):
            raise self._invalid_credentials()
        user = self.db.get(User, auth_session.user_id)
        if user is None or not user.is_active:
            raise self._invalid_credentials()

        auth_session.revoked_at = now
        auth_session.last_used_at = now
        tokens = self._create_session_and_tokens(user)
        self.db.commit()
        self.db.refresh(user)
        return user, tokens

    def logout(self, refresh_token: str) -> None:
        auth_session = self.db.scalar(
            select(AuthSession).where(AuthSession.token_hash == self._hash_token(refresh_token))
        )
        if auth_session is not None and auth_session.revoked_at is None:
            auth_session.revoked_at = datetime.now(timezone.utc)
            self.db.commit()

    def change_password(self, user: User, data: ChangePasswordRequest) -> None:
        if user.password_hash is None or not password_hasher.verify(data.current_password, user.password_hash):
            raise self._invalid_credentials()
        user.password_hash = password_hasher.hash(data.new_password)
        self.db.execute(
            update(AuthSession)
            .where(AuthSession.user_id == user.user_id, AuthSession.revoked_at.is_(None))
            .values(revoked_at=datetime.now(timezone.utc))
        )
        self.db.commit()

    def user_from_access_token(self, token: str) -> User:
        try:
            payload = jwt.decode(token, self.settings.auth_jwt_secret, algorithms=["HS256"])
            user_id = int(payload["sub"])
            if payload.get("type") != "access":
                raise ValueError("Wrong token type")
        except (InvalidTokenError, KeyError, TypeError, ValueError) as exc:
            raise self._invalid_credentials() from exc
        user = self.db.get(User, user_id)
        if user is None or not user.is_active:
            raise self._invalid_credentials()
        return user

    def _create_session_and_tokens(self, user: User) -> dict:
        now = datetime.now(timezone.utc)
        refresh_token = token_urlsafe(48)
        self.db.add(
            AuthSession(
                session_id=uuid4().hex,
                user_id=user.user_id,
                token_hash=self._hash_token(refresh_token),
                expires_at=now + timedelta(days=self.settings.auth_refresh_token_days),
            )
        )
        access_expiration = now + timedelta(minutes=self.settings.auth_access_token_minutes)
        access_token = jwt.encode(
            {
                "sub": str(user.user_id),
                "type": "access",
                "iat": now,
                "exp": access_expiration,
                "jti": uuid4().hex,
            },
            self.settings.auth_jwt_secret,
            algorithm="HS256",
        )
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_in": int((access_expiration - now).total_seconds()),
        }

    @staticmethod
    def _hash_token(token: str) -> str:
        return sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)

    @staticmethod
    def _invalid_credentials() -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
