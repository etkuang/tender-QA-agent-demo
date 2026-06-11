# coding: utf-8
# @Author: Wang Qingkang

import os


class FrontendSettings:
    """Frontend runtime settings loaded from environment variables."""

    BACKEND_URL = os.environ["BACKEND_URL"]

    @property
    def INTERNAL_TIMEOUT(self):
        """Timeout for Frontend -> Backend metadata requests."""
        return None if os.environ.get("DEBUG_MODE") == "true" else 1.0

    @property
    def STREAM_TIMEOUT(self):
        """Timeout for Frontend -> Backend streaming requests."""
        return None if os.environ.get("DEBUG_MODE") == "true" else 300.0


settings = FrontendSettings()