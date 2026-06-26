"""In-memory cache for ingestion pipelines using file hashes."""

import hashlib
from pathlib import Path
from schemas.mri import MRIData


class MRICache:
    """Caching layer preventing redundant loading of previously parsed MRI paths."""

    def __init__(self):
        self._cache = {}

    def compute_hash(self, path: Path) -> str:
        """Computes quick hash combining file metadata and first 1MB of binary content."""
        if not path.exists():
            return ""
        
        stat = path.stat()
        sha256 = hashlib.sha256()
        # Mix file size and modification timestamp
        sha256.update(f"{stat.st_size}_{stat.st_mtime}".encode("utf-8"))
        
        # Read first 1MB content if it is a file
        if path.is_file():
            try:
                with open(path, "rb") as f:
                    sha256.update(f.read(1024 * 1024))
            except Exception:
                pass
        
        return sha256.hexdigest()

    def get(self, hash_key: str) -> MRIData | None:
        """Retrieves cached MRIData if it exists."""
        return self._cache.get(hash_key)

    def set(self, hash_key: str, data: MRIData) -> None:
        """Stores MRIData instance matching the generated hash key."""
        if hash_key:
            self._cache[hash_key] = data

    def clear(self) -> None:
        """Clears all stored entries."""
        self._cache.clear()
