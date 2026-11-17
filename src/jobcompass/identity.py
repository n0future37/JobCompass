"""本地账号、服务端会话与画像。仅保存加盐密码摘要和会话令牌摘要。"""

import hashlib
import hmac
import secrets
import time
import uuid

from sqlalchemy import JSON, Column, Float, String, Table, delete, select, update

from .storage import metadata

users = Table(
    "rag_users",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("username", String(64), unique=True, nullable=False),
    Column("password_hash", String(256), nullable=False),
    Column("profile", JSON, nullable=False),
)
sessions = Table(
    "rag_sessions",
    metadata,
    Column("token_hash", String(64), primary_key=True),
    Column("user_id", String(36), nullable=False),
    Column("expires_at", Float(53), nullable=False),
)


def password_hash(password, salt=None):
    salt = salt or secrets.token_hex(16)
    value = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 600_000).hex()
    return f"pbkdf2_sha256$600000${salt}${value}"


def password_matches(password, encoded):
    try:
        _, _, salt, _ = encoded.split("$")
        return hmac.compare_digest(password_hash(password, salt), encoded)
    except (ValueError, TypeError):
        return False


class Identity:
    def __init__(self, store):
        self.store = store

    def register(self, username, password):
        from sqlalchemy.exc import IntegrityError

        user_id = str(uuid.uuid4())
        try:
            with self.store.engine.begin() as c:
                c.execute(
                    users.insert().values(
                        id=user_id,
                        username=username.casefold(),
                        password_hash=password_hash(password),
                        profile={"text": ""},
                    )
                )
        except IntegrityError:
            raise ValueError("用户名已存在") from None
        return user_id

    def login(self, username, password):
        with self.store.engine.connect() as c:
            user = c.execute(select(users).where(users.c.username == username.casefold())).mappings().first()
        # 不存在的用户名也计算一次摘要，避免明显的时序差别。
        valid = (
            password_matches(password, user["password_hash"])
            if user
            else bool(password_hash(password) and False)
        )
        if not valid:
            raise ValueError("用户名或密码错误")
        token = secrets.token_urlsafe(32)
        with self.store.engine.begin() as c:
            c.execute(delete(sessions).where(sessions.c.expires_at < time.time()))
            c.execute(
                sessions.insert().values(
                    token_hash=hashlib.sha256(token.encode()).hexdigest(),
                    user_id=user["id"],
                    expires_at=time.time() + 86400 * 7,
                )
            )
        return token

    def current(self, token):
        if not token or len(token) > 100:
            return None
        with self.store.engine.connect() as c:
            return (
                c.execute(
                    select(users.c.id, users.c.username, users.c.profile)
                    .join(sessions, users.c.id == sessions.c.user_id)
                    .where(
                        sessions.c.token_hash == hashlib.sha256(token.encode()).hexdigest(),
                        sessions.c.expires_at > time.time(),
                    )
                )
                .mappings()
                .first()
            )

    def logout(self, token):
        with self.store.engine.begin() as c:
            c.execute(
                delete(sessions).where(sessions.c.token_hash == hashlib.sha256(token.encode()).hexdigest())
            )

    def profile(self, user_id, text):
        with self.store.engine.begin() as c:
            row = c.execute(select(users.c.profile).where(users.c.id == user_id).with_for_update()).one()
            profile = dict(row[0], text=text)
            c.execute(update(users).where(users.c.id == user_id).values(profile=profile))
