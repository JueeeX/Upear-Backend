"""Standalone entry point for the SafeHarness Tool Gateway process.

Kept as a module (``python -m tool_gateway_main``) because the demo launcher
passes process arguments through PowerShell native quoting, which shreds
inline ``-c`` snippets containing spaces.
"""

import sys

from core.safeharness_tool_gateway import run_gateway_server

if __name__ == "__main__":
    sys.exit(run_gateway_server())
