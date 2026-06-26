"""Common generic platform helper functions."""

import uuid


def generate_unique_id(prefix: str = "id") -> str:
    """Generates unique UUID formatted string with given prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"
