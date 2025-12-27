# backend/config.py
import os
from datetime import timedelta

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "dev-jwt-secret-change-me")

    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        "mysql+pymysql://root:@localhost/fitquest"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # 🔐 JWT config
    JWT_TOKEN_LOCATION = ["headers"]     # where to look for tokens
    JWT_HEADER_NAME = "Authorization"    # header name
    JWT_HEADER_TYPE = "Bearer"           # expected prefix
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(days=7)  # dev: 7 days
