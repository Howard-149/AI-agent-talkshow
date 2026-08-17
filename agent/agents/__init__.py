"""Role agents (host/guest/commentator) and factory for single-participant handoff."""

from .factory import build_agent
from .host import HostAgent

__all__ = ["HostAgent", "build_agent"]
