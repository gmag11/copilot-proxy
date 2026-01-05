"""FastAPI application exposing the Copilot proxy endpoints."""
from __future__ import annotations

import json
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from .config import (
    get_api_key as get_config_api_key,
    get_base_url as get_config_base_url,
    get_context_length,
    get_model_name as get_config_model_name,
    get_temperature,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.z.ai/api/coding/paas/v4"
DEFAULT_MODEL = "GLM-4.7"
API_KEY_ENV_VARS = ("ZAI_API_KEY", "ZAI_CODING_API_KEY", "GLM_API_KEY")
BASE_URL_ENV_VAR = "ZAI_API_BASE_URL"
CHAT_COMPLETION_PATH = "/chat/completions"
DEFAULT_MAX_OUTPUT_TOKENS = 4096
MAX_REQUEST_SIZE_BYTES = 100_000  # ~100KB limit for Z.AI API requests

def get_model_catalog():
    """Generate the model catalog dynamically based on config."""
    context_length = get_context_length()
    return [
        {
            "name": "GLM-4.7",
            "model": "GLM-4.7",
            "modified_at": "2025-12-21T00:00:00Z",
            "size": 0,
            "digest": "GLM-4.7",
            "details": {
                "format": "glm",
                "family": "glm",
                "families": ["glm"],
                "parameter_size": "cloud",
                "quantization_level": "cloud",
            },
        },
        {
            "name": "GLM-4-Plus",
            "model": "GLM-4-Plus",
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 0,
            "digest": "GLM-4-Plus",
            "details": {
                "format": "glm",
                "family": "glm",
                "families": ["glm"],
                "parameter_size": "cloud",
                "quantization_level": "cloud",
            },
        },
        {
            "name": "GLM-4.6",
            "model": "GLM-4.6",
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 0,
            "digest": "GLM-4.6",
            "details": {
                "format": "glm",
                "family": "glm",
                "families": ["glm"],
                "parameter_size": "cloud",
                "quantization_level": "cloud",
            },
        },
        {
            "name": "GLM-4.5",
            "model": "GLM-4.5",
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 0,
            "digest": "GLM-4.5",
            "details": {
                "format": "glm",
                "family": "glm",
                "families": ["glm"],
                "parameter_size": "cloud",
                "quantization_level": "cloud",
            },
        },
        {
            "name": "GLM-4.5-Air",
            "model": "GLM-4.5-Air",
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 0,
            "digest": "GLM-4.5-Air",
            "details": {
                "format": "glm",
                "family": "glm",
                "families": ["glm"],
                "parameter_size": "cloud",
                "quantization_level": "cloud",
            },
        },
        {
            "name": "GLM-4.5-AirX",
            "model": "GLM-4.5-AirX",
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 0,
            "digest": "GLM-4.5-AirX",
            "details": {
                "format": "glm",
                "family": "glm",
                "families": ["glm"],
                "parameter_size": "cloud",
                "quantization_level": "cloud",
            },
        },
        {
            "name": "GLM-4.5-Flash",
            "model": "GLM-4.5-Flash",
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 0,
            "digest": "GLM-4.5-Flash",
            "details": {
                "format": "glm",
                "family": "glm",
                "families": ["glm"],
                "parameter_size": "cloud",
                "quantization_level": "cloud",
            },
        },
        {
            "name": "GLM-4.6V",
            "model": "GLM-4.6V",
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 0,
            "digest": "GLM-4.6V",
            "details": {
                "format": "glm",
                "family": "glm",
                "families": ["glm"],
                "parameter_size": "cloud",
                "quantization_level": "cloud",
            },
        },
        {
            "name": "GLM-4.6V-Flash",
            "model": "GLM-4.6V-Flash",
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 0,
            "digest": "GLM-4.6V-Flash",
            "details": {
                "format": "glm",
                "family": "glm",
                "families": ["glm"],
                "parameter_size": "cloud",
                "quantization_level": "cloud",
            },
        },
        {
            "name": "GLM-4.6V-FlashX",
            "model": "GLM-4.6V-FlashX",
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 0,
            "digest": "GLM-4.6V-FlashX",
            "details": {
                "format": "glm",
                "family": "glm",
                "families": ["glm"],
                "parameter_size": "cloud",
                "quantization_level": "cloud",
            },
        },
        {
            "name": "GLM-4.5V",
            "model": "GLM-4.5V",
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 0,
            "digest": "GLM-4.5V",
            "details": {
                "format": "glm",
                "family": "glm",
                "families": ["glm"],
                "parameter_size": "cloud",
                "quantization_level": "cloud",
            },
        },
        {
            "name": "AutoGLM-Phone-Multilingual",
            "model": "AutoGLM-Phone-Multilingual",
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 0,
            "digest": "AutoGLM-Phone-Multilingual",
            "details": {
                "format": "glm",
                "family": "glm",
                "families": ["glm"],
                "parameter_size": "cloud",
                "quantization_level": "cloud",
            },
        },
        {
            "name": "GLM-4-32B-0414-128K",
            "model": "GLM-4-32B-0414-128K",
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 0,
            "digest": "GLM-4-32B-0414-128K",
            "details": {
                "format": "glm",
                "family": "glm",
                "families": ["glm"],
                "parameter_size": "cloud",
                "quantization_level": "cloud",
            },
        },
    ]


MODEL_CATALOG = get_model_catalog()


def _truncate_content(content: str, max_length: int = 500) -> str:
    """Truncate content if it's too long for logging."""
    if len(content) <= max_length:
        return content
    return content[:max_length] + f"... (truncated, total length: {len(content)})"


def _is_vision_model(model_name: str) -> bool:
    """Detect if a model supports vision based on its name.
    
    Vision models have version numbers directly followed by 'V'
    (e.g., GLM-4.6V, GLM-4.5V, GLM-4.6V-Flash).
    """
    if not model_name:
        return False
    # Pattern: version number (digit.digit+) followed by 'V'
    # Matches: "4.6V", "4.5V", etc. in model names
    return bool(re.search(r'\d+\.\d+V', model_name))


def _get_api_key() -> str:
    # First try to get API key from config file
    config_api_key = get_config_api_key()
    if config_api_key:
        return config_api_key

    # Fall back to environment variables
    for env_var in API_KEY_ENV_VARS:
        api_key = os.getenv(env_var)
        if api_key:
            return api_key.strip()

    raise RuntimeError(
        "Missing Z.AI API key. Please set it using 'copilot-proxy config set-api-key <key>' "
        f"or set one of the following environment variables: {', '.join(API_KEY_ENV_VARS)}"
    )


def _get_base_url() -> str:
    # First try to get base URL from config file
    config_base_url = get_config_base_url()
    if config_base_url:
        base_url = config_base_url
    else:
        # Fall back to environment variable or default
        base_url = os.getenv(BASE_URL_ENV_VAR, DEFAULT_BASE_URL).strip()
        if not base_url:
            base_url = DEFAULT_BASE_URL

    if not base_url.startswith("http://") and not base_url.startswith("https://"):
        base_url = f"https://{base_url}"
    return base_url.rstrip("/")


def _get_chat_completion_url() -> str:
    base_url = _get_base_url()
    if base_url.endswith(CHAT_COMPLETION_PATH):
        return base_url
    return f"{base_url}{CHAT_COMPLETION_PATH}"


@asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: D401 - FastAPI lifespan signature
    """Ensure configuration is ready before serving requests."""

    try:
        _ = _get_api_key()
        print("GLM Coding Plan proxy is ready.")
    except Exception as exc:  # pragma: no cover - startup logging
        print(f"Failed to initialise GLM Coding Plan proxy: {exc}")
    yield


def create_app() -> FastAPI:
    """Create and return a configured FastAPI application."""

    app = FastAPI(lifespan=_lifespan)

    @app.get("/")
    async def root():  # noqa: D401 - FastAPI route
        """Return a simple health message."""

        return {"message": "GLM Coding Plan proxy is running"}

    @app.get("/api/ps")
    async def list_running_models():  # noqa: D401 - FastAPI route
        """Return an empty list as we do not host local models."""

        return {"models": []}

    @app.get("/api/version")
    async def get_version():  # noqa: D401 - FastAPI route
        """Expose a version compatible with the Ollama API expectations."""

        return {"version": "0.6.4"}

    @app.get("/api/tags")
    @app.get("/api/list")
    async def list_models():  # noqa: D401 - FastAPI route
        """Return the static catalog of GLM models."""

        return {"models": get_model_catalog()}

    @app.post("/api/show")
    async def show_model(request: Request):  # noqa: D401 - FastAPI route
        """Handle Ollama-compatible model detail queries."""

        try:
            body = await request.json()
            model_name = body.get("model")
        except Exception:
            model_name = DEFAULT_MODEL

        if not model_name:
            model_name = get_config_model_name() or DEFAULT_MODEL

        context_length = get_context_length()
        
        # Detect if the model supports vision
        supports_vision = _is_vision_model(model_name)
        
        # Build capabilities array (legacy Ollama format)
        capabilities = ["tools"]
        if supports_vision:
            capabilities.append("vision")
        
        # Calculate token limits
        max_output_tokens = DEFAULT_MAX_OUTPUT_TOKENS
        max_prompt_tokens = max(0, context_length - max_output_tokens)

        return {
            "template": "{{ .System }}\n{{ .Prompt }}",
            "capabilities": capabilities,
            "details": {
                "family": "glm",
                "families": ["glm"],
                "format": "glm",
                "parameter_size": "cloud",
                "quantization_level": "cloud",
            },
            "model_info": {
                "general.basename": model_name,
                "general.architecture": "glm",
                "glm.context_length": context_length,
                # Add capability information in Ollama-compatible format
                "glm.type": "chat",
                "glm.supports.streaming": True,
                "glm.supports.vision": supports_vision,
                "glm.supports.tool_calls": True,
                "glm.limits.max_prompt_tokens": max_prompt_tokens,
                "glm.limits.max_output_tokens": max_output_tokens,
                "glm.limits.max_context_window_tokens": context_length,
            },
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):  # noqa: D401 - FastAPI route
        """Forward chat completion calls to the Z.AI backend."""

        body = await request.json()
        
        # Check request size before processing
        body_str = json.dumps(body, ensure_ascii=False)
        body_size = len(body_str.encode('utf-8'))
        
        if body_size > MAX_REQUEST_SIZE_BYTES:
            logger.warning(f"Request too large: {body_size} bytes (limit: {MAX_REQUEST_SIZE_BYTES})")
            logger.warning("This typically happens when VS Code sends very long system prompts")
            logger.warning("Consider truncating system messages or using a different model configuration")
            
            # Return error before sending to Z.AI
            return {
                "id": "error",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", DEFAULT_MODEL),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Request too large ({body_size} bytes). Z.AI API has a limit of approximately {MAX_REQUEST_SIZE_BYTES} bytes. Please reduce the system prompt size or message length."
                    },
                    "finish_reason": "length"
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
        
        # Log request structure
        logger.info(f"Chat completion request: model={body.get('model')}, stream={body.get('stream')}, messages_count={len(body.get('messages', []))}, size={body_size} bytes")
        
        # Log all top-level keys to identify unsupported parameters
        logger.info(f"Request keys: {list(body.keys())}")
        
        # Log message structure (without full content)
        if "messages" in body:
            for i, msg in enumerate(body["messages"]):
                content_preview = str(msg.get("content", ""))[:100] if msg.get("content") else "None"
                logger.debug(f"Message {i}: role={msg.get('role')}, content_length={len(str(msg.get('content', '')))}, content_preview={content_preview}...")
        
        logger.debug(f"Full request body (first 2000 chars): {body_str[:2000]}")
        if len(body_str) > 2000:
            logger.debug(f"Full request body (last 1000 chars): ...{body_str[-1000:]}")

        if not body.get("model"):
            body["model"] = get_config_model_name() or DEFAULT_MODEL

        # Apply temperature override if configured
        config_temp = get_temperature()
        if config_temp is not None:
            body["temperature"] = config_temp

        stream = body.get("stream", False)

        api_key = _get_api_key()
        chat_completion_url = _get_chat_completion_url()

        async def generate_chunks() -> AsyncGenerator[bytes, None]:
            async with httpx.AsyncClient(timeout=300.0) as client:
                try:
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    }
                    response = await client.post(
                        chat_completion_url,
                        headers=headers,
                        json=body,
                        timeout=None,
                    )
                    response.raise_for_status()

                    async for chunk in response.aiter_bytes():
                        yield chunk

                except httpx.HTTPStatusError as exc:
                    logger.error(f"HTTP {exc.response.status_code} error from Z.AI API")
                    logger.error(f"Request URL: {exc.request.url}")
                    logger.error(f"Request body: {_truncate_content(json.dumps(body, ensure_ascii=False), 2000)}")
                    try:
                        error_body = exc.response.text
                        logger.error(f"Response body: {_truncate_content(error_body, 1000)}")
                    except Exception:
                        logger.error("Could not read response body")
                    
                    if exc.response.status_code == 401:
                        raise RuntimeError("Unauthorized. Check your Z.AI API key.") from exc
                    raise

        if stream:
            return StreamingResponse(generate_chunks(), media_type="text/event-stream")

        async with httpx.AsyncClient(timeout=300.0) as client:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            try:
                response = await client.post(chat_completion_url, headers=headers, json=body)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                logger.error(f"HTTP {exc.response.status_code} error from Z.AI API (non-streaming)")
                logger.error(f"Request URL: {exc.request.url}")
                logger.error(f"Request body: {_truncate_content(json.dumps(body, ensure_ascii=False), 2000)}")
                try:
                    error_body = exc.response.text
                    logger.error(f"Response body: {_truncate_content(error_body, 1000)}")
                except Exception:
                    logger.error("Could not read response body")
                
                if exc.response.status_code == 401:
                    raise RuntimeError("Unauthorized. Check your Z.AI API key.") from exc
                raise

    return app


app = create_app()

__all__ = [
    "API_KEY_ENV_VARS",
    "BASE_URL_ENV_VAR",
    "CHAT_COMPLETION_PATH",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "MODEL_CATALOG",
    "app",
    "create_app",
]
