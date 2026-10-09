"""MCP servers package.

This package contains standalone Model Context Protocol (MCP) stdio servers.
They are intentionally kept separate from the FastAPI app so importing/starting
SUMC won't implicitly start or require MCP dependencies.
"""
