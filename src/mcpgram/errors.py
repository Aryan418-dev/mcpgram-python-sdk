"""
Thrown for any non-2xx response from the MCPGRAM API, except the 502
"tool executed but the tool itself failed" case -- that comes back as a
normal ExecuteResult with status="error" instead of raising, since it's an
expected outcome an agent should be able to branch on, not an exceptional
one. Mirrors src/errors.ts in the JS SDK.
"""
from __future__ import annotations


class PlatformApiError(Exception):
    def __init__(self, message: str, status: int, body: object = None, retry_after_ms: float | None = None):
        super().__init__(message)
        self.status = status
        self.body = body
        self.retry_after_ms = retry_after_ms
