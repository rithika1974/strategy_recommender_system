"""Base agent interface."""


class BaseAgent:
    """Shared base class for analysis agents."""

    def __init__(self, name: str):
        self.name = name
