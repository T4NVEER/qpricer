"""Shared input validation helpers."""


def require_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")


def require_non_negative(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}")
