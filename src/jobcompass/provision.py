"""使用本机管理员配置创建RAG专属账号；运行时使用受限账号。"""

import logging
import secrets
from pathlib import Path

from .settings import ROOT, Settings, env_file


def provision():
    import pymysql
    import yaml
    from pymilvus import MilvusClient

    logging.getLogger("pymilvus").setLevel(logging.CRITICAL)
    local = env_file(ROOT / ".env.local")
    for stage in ("DEV", "TEST"):
        local.setdefault(f"MYSQL_{stage}_PASSWORD", secrets.token_urlsafe(28))
        local.setdefault(f"MILVUS_{stage}_PASSWORD", secrets.token_urlsafe(28))
    # 仅复用非敏感的供应商设置，密钥仍由配置指向的环境变量读取。
    original = env_file(Path("configs/provider.local.env"))
    for key in ("API_BASE", "API_KEY_ENV", "MODEL", "API_TYPE", "REASONING_EFFORT"):
        if key in original:
            local.setdefault(key, original[key])
    (ROOT / ".env.local").write_text("\n".join(f"{k}={v}" for k, v in local.items()) + "\n", encoding="utf-8")
    admin_password = env_file(Path("configs/mysql.admin.env"))["MYSQL_ROOT_PASSWORD"]
    connection = pymysql.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password=admin_password,
        connect_timeout=5,
        autocommit=True,
    )
    try:
        with connection.cursor() as cursor:
            for stage in ("dev", "test"):
                database, user = f"jobcompass_{stage}", f"jc_rag_{stage}"
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{database}` CHARACTER SET utf8mb4")
                cursor.execute(
                    f"CREATE USER IF NOT EXISTS '{user}'@'%%' IDENTIFIED BY %s",
                    (local[f"MYSQL_{stage.upper()}_PASSWORD"],),
                )
                cursor.execute(
                    f"GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER, INDEX, DROP, REFERENCES "
                    f"ON `{database}`.* TO '{user}'@'%'"
                )
    finally:
        connection.close()
    config = yaml.safe_load(Path("configs/milvus.admin.yaml").read_text(encoding="utf-8"))
    password = config["common"]["security"]["defaultRootPassword"]
    client = MilvusClient(uri="http://127.0.0.1:19530", user="root", password=str(password), timeout=15)
    try:
        databases = client.list_databases()
        users, roles = (
            client.list_users(),
            [x["role_name"] if isinstance(x, dict) else x for x in client.list_roles()],
        )
        for stage in ("dev", "test"):
            database, user = f"jobcompass_{stage}", f"jc_rag_{stage}"
            if database not in databases:
                client.create_database(db_name=database)
            if user not in users:
                client.create_user(user_name=user, password=local[f"MILVUS_{stage.upper()}_PASSWORD"])
            if user not in roles:
                client.create_role(role_name=user)
            client.grant_privilege_v2(
                role_name=user,
                privilege="CollectionAdmin",
                collection_name="*",
                db_name=database,
            )
            client.grant_privilege_v2(
                role_name=user,
                privilege="CreateCollection",
                collection_name="*",
                db_name=database,
            )
            client.grant_privilege_v2(
                role_name=user,
                privilege="DropCollection",
                collection_name="*",
                db_name=database,
            )
            client.grant_role(user_name=user, role_name=user)
    finally:
        client.close()
    checks = []
    from sqlalchemy import text

    from .storage import engine

    for stage in ("dev", "test"):
        settings = Settings(stage)
        e = engine(settings)
        with e.connect() as c:
            assert c.execute(text("SELECT DATABASE()")).scalar_one() == settings.db_name
        e.dispose()
        c = MilvusClient(
            uri="http://127.0.0.1:19530",
            user=settings.db_user,
            password=settings.milvus_password,
            db_name=settings.db_name,
            timeout=15,
        )
        c.list_collections()
        c.close()
        checks.append({"environment": stage, "mysql": "connected", "milvus": "connected"})
    return checks
