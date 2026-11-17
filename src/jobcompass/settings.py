"""项目配置；只在本机解析凭据，不输出连接密码。"""

import os
import re
import tomllib
from pathlib import Path

ROOT = Path(os.getenv("JOBCOMPASS_ROOT", Path(__file__).resolve().parents[2]))


def env_file(path):
    result = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                result[key.strip()] = value.strip().strip("\"'")
    return result


class Settings:
    def __init__(self, stage=None):
        self.stage = stage or os.getenv("JOBCOMPASS_ENV", "dev")
        if self.stage not in {"dev", "test"}:
            raise ValueError("环境只能为dev或test")
        self.values = env_file(ROOT / ".env.local") | dict(os.environ)
        self.config = tomllib.loads((ROOT / "configs/rag.toml").read_text(encoding="utf-8"))
        self.db_name = f"jobcompass_{self.stage}"
        self.db_user = f"jc_rag_{self.stage}"
        self.mysql_password = self.values.get(f"MYSQL_{self.stage.upper()}_PASSWORD", "")
        self.milvus_password = self.values.get(f"MILVUS_{self.stage.upper()}_PASSWORD", "")
        self.redis_password = env_file(Path("configs/redis.local.env")).get("REDIS_PASSWORD")
        self.redis_db = 0 if self.stage == "dev" else 1
        self.prefix = f"jobcompass:{self.stage}:rag:v1:"
        namespace = self.values.get("JOBCOMPASS_RUN_NAMESPACE", "")
        if namespace:
            if self.stage != "test" or not re.fullmatch(r"[a-z0-9_]{1,32}", namespace):
                raise ValueError("运行命名空间仅支持test环境的1至32位小写字母数字下划线")
            self.prefix += namespace + ":"
        self.mysql_host = self.values.get("MYSQL_HOST", "127.0.0.1")
        self.redis_host = self.values.get("REDIS_HOST", "127.0.0.1")
        self.milvus_uri = self.values.get("MILVUS_URI", "http://127.0.0.1:19530")
        self.redis_password = self.values.get("REDIS_PASSWORD", self.redis_password)
        for section in ("embedding", "reranker"):
            if self.values.get(section.upper() + "_PATH"):
                self.config[section]["path"] = self.values[section.upper() + "_PATH"]

    @property
    def mysql_url(self):
        from sqlalchemy import URL

        return URL.create(
            "mysql+pymysql",
            username=self.db_user,
            password=self.mysql_password,
            host=self.mysql_host,
            port=3306,
            database=self.db_name,
            query={"charset": "utf8mb4"},
        )

    @property
    def api_config(self):
        return {
            key: self.values.get(key, "")
            for key in (
                "API_BASE",
                "API_KEY_ENV",
                "MODEL",
                "API_TYPE",
                "REASONING_EFFORT",
                "ENABLE_THINKING",
                "TEMPERATURE",
            )
        }


def project_path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p
