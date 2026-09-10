from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.core.database.session import get_db
from app.modules.auth.dependencies import get_current_user
from app.modules.auth.schemas import (
    AuthTokenRead,
    AuthUserRead,
    ChangePasswordRequest,
    LoginRequest,
    LogoutRequest,
    RefreshRequest,
    RegisterRequest,
)
from app.modules.auth.service import AuthService
from app.modules.projects.models import User


router = APIRouter()
CurrentUser = Annotated[User, Depends(get_current_user)]
Database = Annotated[Session, Depends(get_db)]


def _token_response(user: User, tokens: dict) -> AuthTokenRead:
    return AuthTokenRead(user=user, **tokens)


@router.post("/register", response_model=AuthTokenRead, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Database):
    user, tokens = AuthService(db).register(payload)
    return _token_response(user, tokens)


@router.post("/login", response_model=AuthTokenRead)
def login(payload: LoginRequest, db: Database):
    user, tokens = AuthService(db).login(payload)
    return _token_response(user, tokens)


@router.post("/refresh", response_model=AuthTokenRead)
def refresh(payload: RefreshRequest, db: Database):
    user, tokens = AuthService(db).refresh(payload.refresh_token)
    return _token_response(user, tokens)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(payload: LogoutRequest, db: Database):
    AuthService(db).logout(payload.refresh_token)


@router.get("/me", response_model=AuthUserRead)
def me(current_user: CurrentUser):
    return current_user


@router.post("/change-password", status_code=status.HTTP_204_NO_CONTENT)
def change_password(payload: ChangePasswordRequest, current_user: CurrentUser, db: Database):
    AuthService(db).change_password(current_user, payload)
