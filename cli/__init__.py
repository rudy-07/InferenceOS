"""
InferenceOS CLI Package
======================
Complete, modern command line interface and interactive runtime environment for InferenceOS.
"""

__version__ = "1.0.0"

import sys
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
