def wkey(prefix: str, name: str) -> str:
    return f"{prefix}__{name}" if prefix else name

