"""Config (configs/objsearch.yaml), paths and .env loading."""
import functools
import hashlib
import json
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "configs", "objsearch.yaml")
SECRET_ENV = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "SKETCHFAB_TOKEN",
              "GOOGLE_CLOUD_VISION_KEY", "SNAPEDA_TOKEN", "ULTRALIBRARIAN_TOKEN")


def load_dotenv(path=os.path.join(ROOT, ".env")):
    """Minimal KEY=VALUE loader; never overrides variables already set."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


@functools.lru_cache(maxsize=1)
def load():
    load_dotenv()
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_dir():
    d = os.environ.get("OBJSEARCH_DATA") or load()["data_dir"]
    return d if os.path.isabs(d) else os.path.join(ROOT, d)


def db_path():
    return os.path.join(data_dir(), "objsearch.db")


def config_hash(cfg=None):
    return hashlib.sha1(json.dumps(cfg or load(), sort_keys=True).encode()).hexdigest()[:12]


def secrets():
    """Current secret values, for log redaction."""
    return [os.environ[k] for k in SECRET_ENV if len(os.environ.get(k, "")) >= 8]
