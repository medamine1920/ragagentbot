from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import Request, HTTPException, status, Response
from fastapi.security import OAuth2PasswordRequestForm
from passlib.context import CryptContext
from pydantic import BaseModel

from config import settings  # ✅ Import settings only once
from services.cassandra_connector import CassandraConnector


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # Or read from settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserInDB(BaseModel):
    username: str
    email: Optional[str] = None
    his_job: Optional[str] = None


class Token(BaseModel):
    access_token: str
    token_type: str
    the_user: str
    his_job: str


class AuthService:
    def __init__(self, db: CassandraConnector):
        self.db = db

    def hash_password(self, password: str) -> str:
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    def authenticate_user(self, username: str, password: str) -> Optional[UserInDB]:
        user = self.db.get_user(username)
        if not user or not self.verify_password(password, user.password):
            return None
        return user

    async def login_for_access_token(self, response: Response, form_data: OAuth2PasswordRequestForm) -> Token:
        user = self.authenticate_user(form_data.username, form_data.password)
        if not user:
            raise HTTPException(status_code=401, detail="Incorrect username or password")

        access_token = self.create_access_token(
            data={"sub": user.username, "role": user.his_job},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        response.set_cookie(key="auth_token", value=access_token, httponly=True)
        return Token(access_token=access_token, token_type="bearer", the_user=user.username, his_job=user.his_job)

    async def logout(self, response: Response):
        response.delete_cookie("auth_token")

    async def get_current_user(self, request: Request) -> UserInDB:
        token = request.cookies.get("auth_token")
        if not token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            role: str = payload.get("role")
            if not username or not role:
                raise HTTPException(status_code=401, detail="Invalid token")
            return UserInDB(username=username, his_job=role)
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
        
    def authenticate_user(self, username: str, password: str) -> Optional[UserInDB]:
        user_data = self.db.get_user(username)
        print(f"[DEBUG] User from DB: {user_data}")
        if not user_data:
            print("[DEBUG] User not found.")
            return None
        if not verify_password(password, user_data["password"]):
            print("[DEBUG] Password mismatch.")
            return None
        print("[DEBUG] User authenticated successfully.")
        return UserInDB(username=user_data["username"], his_job=user_data["his_job"])
