import os
import time
from pathlib import Path

class LocalStore:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def make_path(self, job_id: str, filename: str) -> str:
        p = self.base_dir / job_id
        p.mkdir(parents=True, exist_ok=True)
        return str(p / filename)

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def delete_path(self, path: str) -> None:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
        try:
            d = Path(path).parent
            if d.exists() and d.is_dir():
                d.rmdir()
        except Exception:
            pass

    def cleanup_older_than(self, ttl_seconds: int) -> int:
        now = time.time()
        deleted = 0
        for job_dir in self.base_dir.glob("*"):
            if not job_dir.is_dir():
                continue
            try:
                mtime = job_dir.stat().st_mtime
                if (now - mtime) > ttl_seconds:
                    for f in job_dir.glob("*"):
                        self.delete_path(str(f))
                    try:
                        job_dir.rmdir()
                    except Exception:
                        pass
                    deleted += 1
            except Exception:
                continue
        return deleted
