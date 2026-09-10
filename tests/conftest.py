import os
import tempfile
from pathlib import Path


test_root = Path(tempfile.mkdtemp(prefix="cv-platform-tests-"))
os.environ["DATABASE_URL"] = f"sqlite:///{test_root / 'test.sqlite3'}"
os.environ["STORAGE_ROOT"] = str(test_root / "storage")
os.environ["TEMP_ROOT"] = str(test_root / "temp")
os.environ["AUTH_JWT_SECRET"] = "test-secret-that-is-long-enough-for-jwt-signing"


def pytest_sessionstart(session):
    from alembic import command
    from alembic.config import Config

    config = Config(str(Path(__file__).parents[1] / "alembic.ini"))
    command.upgrade(config, "head")
