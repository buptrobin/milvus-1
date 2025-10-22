# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Milvus vector database project configured with Python dependencies using `uv` package manager.

## Development Environment

### Package Management
- **Tool**: `uv` (fast Python package manager)
- **Python Version**: >= 3.11
- **Virtual Environment**: `.venv` (managed by uv)

### Installation Commands
```bash
# Install all dependencies
uv sync

# Install with specific mirror (China)
uv sync --index-url https://mirrors.aliyun.com/pypi/simple/

# Run Python scripts
uv run python script.py

# Add new dependencies
uv add package_name

# Install optional dependencies
uv sync --all-extras  # Install all optional deps
uv sync --extra ml    # Install ML dependencies
uv sync --extra dev   # Install dev tools
```

## Dependencies

### Core Dependencies
- **pymilvus**: Official Milvus Python SDK
- **pymilvus-model**: Model utilities for Milvus
- **pandas, numpy**: Data processing
- **python-dotenv**: Environment variable management

### Optional Dependencies
- **ML**: transformers, sentence-transformers, torch, onnxruntime
- **Image**: pillow, opencv-python
- **Dev**: pytest, black, ruff, jupyter

### MCP (Model Context Protocol) Dependencies
- **context7-mcp-python**: Context7 MCP Server - Provides up-to-date library documentation and code examples
- **playwright-mcp**: Playwright wrapper for MCP - Browser automation capabilities
- **playwright**: Core browser automation library with Chromium, Firefox, and WebKit support

## Development Workflow

### Running Code
```bash
# Run with uv
uv run python your_script.py

# Test Milvus connection
uv run python test_milvus.py
```

### Code Quality
```bash
# Format code
uv run black .

# Lint code
uv run ruff check .

# Run tests
uv run pytest
```

## Milvus Connection

Default connection parameters:
- Host: localhost
- Port: 19530

Example usage:
```python
from pymilvus import connections
connections.connect(host='localhost', port='19530')
```

## MCP Server Usage

### Context7 MCP Server
```bash
# Run Context7 MCP server with stdio transport
uv run context7-mcp-python --transport stdio

# Run with SSE transport
uv run context7-mcp-python --transport sse --port 34504 --host localhost
```

### Playwright MCP
```python
from playwright.sync_api import sync_playwright
import playwright_mcp

# Browser automation example
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto("https://example.com")
    # Perform automation tasks
    browser.close()
```

## Notes

- Milvus is a vector database for similarity search and AI applications
- Use Aliyun mirror for faster package downloads in China
- The project uses `hatchling` as build backend
- MCP servers enable AI assistants to interact with external tools and services
- Playwright browsers are installed in `C:\Users\{username}\AppData\Local\ms-playwright\`