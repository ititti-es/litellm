"""Public-edge restrictions (X-Liminal-Edge: public): routes, keys, MCP tools, private models."""

from __future__ import annotations

from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse
from starlette.routing import Mount, Route
from starlette.testclient import TestClient

from litellm.proxy.public_edge import (
    PublicEdgeMiddleware,
    enforce_public_edge,
    is_public_edge,
    is_public_route,
    private_model,
)

EDGE: Final = {"X-Liminal-Edge": "public"}


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({}, False),
        ({"x-liminal-edge": "public"}, True),
        ({"x-liminal-edge": " PUBLIC "}, True),
        ({"x-liminal-edge": "private"}, False),
    ],
)
def test_is_public_edge(headers: dict[str, str], expected: bool) -> None:
    assert is_public_edge(headers) is expected


@pytest.mark.parametrize(
    "path",
    [
        "/v1/chat/completions",
        "/chat/completions",
        "/v1/responses",
        "/v1/responses/resp_123",
        "/v1/responses/resp_1/input_items",
        "/v1/messages",
        "/v1/models",
        "/v1/embeddings",
        "/v1/rerank",
        "/v1/audio/speech",
        "/v1/images/generations",
        "/images/generations",
        "/v1/images/edits",
        "/v1/images/variations",
        "/health",
    ],
)
def test_inference_routes_are_public(path: str) -> None:
    assert is_public_route(path)


@pytest.mark.parametrize(
    "path",
    [
        "/mcp",
        "/mcp/",
        "/mcp-rest/tools/list",
        "/v1/mcp/server",
        "/key/list",
        "/key/generate",
        "/user/info",
        "/model/new",
        "/ui/",
        "/sso/key/generate",
        "/login",
        "/v1/memory/search",
        "/redoc",
        "/docs",
        "/swaggerX",
        "/routes",
        "/health/readiness",
        "/health/services",
        "/health/test_connection",
        "/v1/images/generationsX",
        "/openai/deployments/x/images/generations",
        "/.well-known/oauth-authorization-server",
        "/v1/responsesX",
        "/config/yaml",
        "/spend/logs",
    ],
)
def test_everything_else_is_not_public(path: str) -> None:
    assert not is_public_route(path)


@pytest.mark.parametrize("path", ["/", "/openapi.json", "/swagger/swagger-ui.css", "/swagger/root.jpg"])
def test_docs_page_is_public(path: str) -> None:
    assert is_public_route(path)


def _app(root_path: str = "") -> Starlette:
    async def ok(_request: object) -> PlainTextResponse:
        return PlainTextResponse("reached")

    mcp_app: Final = Starlette(routes=[Route("/", ok)])
    app: Final = Starlette(
        routes=[Route("/v1/chat/completions", ok, methods=["POST"]), Route("/key/list", ok), Mount("/mcp", app=mcp_app)]
    )
    app.add_middleware(PublicEdgeMiddleware)
    return app


def test_middleware_refuses_admin_and_mounted_routes_on_the_edge() -> None:
    client: Final = TestClient(_app())
    assert client.get("/key/list", headers=EDGE).status_code == 403
    assert client.get("/mcp/", headers=EDGE).status_code == 403
    assert client.post("/v1/chat/completions", headers=EDGE).text == "reached"


def test_middleware_leaves_requests_without_the_header_alone() -> None:
    client: Final = TestClient(_app())
    assert client.get("/key/list").text == "reached"
    assert client.get("/mcp/").text == "reached"


def test_middleware_strips_root_path() -> None:
    client: Final = TestClient(_app(), root_path="/liminal")
    assert client.get("/liminal/key/list", headers=EDGE).status_code == 403
    assert client.post("/liminal/v1/chat/completions", headers=EDGE).text == "reached"


def _request(headers: dict[str, str]) -> MagicMock:
    request: Final = MagicMock()
    request.headers = Headers(headers)
    return request


PUBLIC_KEY: Final = {"public": True}
PUBLIC_DEPLOYMENT: Final = [
    {"litellm_params": {"model": "deepinfra/zai-org/GLM-5.3-Flash", "api_key": "os.environ/DEEPINFRA_API_KEY"}}
]


def test_no_header_is_a_no_op_even_for_admin_mcp_and_private_models() -> None:
    enforce_public_edge(
        _request({}),
        {"tools": [{"type": "mcp", "server_label": "cua_soraka"}]},
        key_metadata=None,
        is_admin=True,
        deployments_for_model=[{"litellm_params": {"model": "claude_cli/claude-sonnet-5"}}],
    )


def test_public_key_with_public_model_passes() -> None:
    enforce_public_edge(
        _request(EDGE),
        {"model": "glm-5.3-flash"},
        key_metadata=PUBLIC_KEY,
        is_admin=False,
        deployments_for_model=PUBLIC_DEPLOYMENT,
    )


@pytest.mark.parametrize(
    ("metadata", "is_admin"),
    [(None, False), ({}, False), ({"public": False}, False), ({"public": "true"}, False), (PUBLIC_KEY, True)],
)
def test_only_non_admin_keys_flagged_public_are_accepted(metadata: dict | None, is_admin: bool) -> None:
    with pytest.raises(HTTPException) as exc:
        enforce_public_edge(_request(EDGE), {}, key_metadata=metadata, is_admin=is_admin, deployments_for_model=None)
    assert exc.value.status_code == 401


def test_mcp_tools_are_refused() -> None:
    with pytest.raises(HTTPException) as exc:
        enforce_public_edge(
            _request(EDGE),
            {
                "model": "glm-5.3-flash",
                "tools": [
                    {"type": "function", "function": {"name": "f"}},
                    {"type": "mcp", "server_label": "cua_soraka"},
                ],
            },
            key_metadata=PUBLIC_KEY,
            is_admin=False,
            deployments_for_model=PUBLIC_DEPLOYMENT,
        )
    assert exc.value.status_code == 403


def test_function_tools_are_allowed() -> None:
    enforce_public_edge(
        _request(EDGE),
        {"model": "glm-5.3-flash", "tools": [{"type": "function", "function": {"name": "f"}}]},
        key_metadata=PUBLIC_KEY,
        is_admin=False,
        deployments_for_model=PUBLIC_DEPLOYMENT,
    )


@pytest.mark.parametrize(
    "params",
    [
        {"model": "claude_cli/claude-sonnet-5"},
        {"model": "chatgpt/gpt-6-luna"},
        {"model": "blitzcrank/ensemble"},
        {"model": "xai/grok-4.3", "use_xai_oauth": True},
        {"model": "anthropic/claude-sonnet-5", "use_anthropic_oauth": True},
        {"model": "ensemble", "custom_llm_provider": "blitzcrank"},
    ],
)
def test_host_executed_and_subscription_models_are_refused(params: dict) -> None:
    assert private_model([{"litellm_params": params}])
    with pytest.raises(HTTPException) as exc:
        enforce_public_edge(
            _request(EDGE),
            {"model": "x"},
            key_metadata=PUBLIC_KEY,
            is_admin=False,
            deployments_for_model=[{"litellm_params": params}],
        )
    assert exc.value.status_code == 403


def test_proxy_app_refuses_admin_and_mcp_routes_on_the_edge() -> None:
    from litellm.proxy.proxy_server import app

    assert isinstance(app, FastAPI)
    client: Final = TestClient(app)
    for path in ("/key/list", "/mcp/", "/v1/memory/search", "/routes"):
        assert client.get(path, headers=EDGE).status_code == 403, path
    for path in ("/", "/openapi.json"):
        assert client.get(path, headers=EDGE).status_code == 200, path
    # Without the header the middleware lets the request through to the route as before.
    assert client.get("/key/list").status_code != 403


def _starlette_request(headers: dict[str, str], body: bytes) -> object:
    from starlette.requests import Request

    async def receive() -> dict:
        return {"type": "http.request", "body": body, "more_body": False}

    scope: Final = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "root_path": "",
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": ("172.25.9.2", 1234),
        "server": ("172.25.9.1", 4000),
        "scheme": "http",
    }
    return Request(scope, receive)


@pytest.mark.asyncio
@pytest.mark.parametrize(("metadata", "expected"), [({}, 401), ({"public": True}, None)])
async def test_user_api_key_auth_runs_the_public_edge_check(metadata: dict, expected: int | None) -> None:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth import user_api_key_auth as auth_module

    key: Final = UserAPIKeyAuth(api_key="hashed", metadata=metadata)
    request: Final = _starlette_request({**EDGE, "content-type": "application/json"}, b'{"model": "glm-5.3-flash"}')
    with (
        patch.object(auth_module, "_user_api_key_auth_builder", AsyncMock(return_value=key)),
        patch.object(auth_module, "_authorize_authenticated_request", AsyncMock(return_value=key)),
        patch.object(auth_module, "_deployments_for_requested_model", return_value=PUBLIC_DEPLOYMENT),
    ):
        if expected is None:
            assert await auth_module.user_api_key_auth(request=request, api_key="Bearer sk-x") is key
        else:
            with pytest.raises(HTTPException) as exc:
                await auth_module.user_api_key_auth(request=request, api_key="Bearer sk-x")
            assert exc.value.status_code == expected


CHATGPT_KEY: Final = {"public": True, "allow_chatgpt_subscription": True}


def test_chatgpt_opt_in_key_may_use_chatgpt_models() -> None:
    enforce_public_edge(
        _request(EDGE),
        {"model": "gpt-6-luna", "tools": [{"type": "image_generation"}]},
        key_metadata=CHATGPT_KEY,
        is_admin=False,
        deployments_for_model=[{"litellm_params": {"model": "chatgpt/gpt-6-luna"}}],
    )


@pytest.mark.parametrize("opt_in", [False, "true", 1, None])
def test_chatgpt_needs_a_literal_true_opt_in(opt_in: object) -> None:
    with pytest.raises(HTTPException) as exc:
        enforce_public_edge(
            _request(EDGE),
            {"model": "gpt-6-luna"},
            key_metadata={"public": True, "allow_chatgpt_subscription": opt_in},
            is_admin=False,
            deployments_for_model=[{"litellm_params": {"model": "chatgpt/gpt-6-luna"}}],
        )
    assert exc.value.status_code == 403


@pytest.mark.parametrize(
    "params",
    [
        {"model": "claude_cli/claude-sonnet-5"},
        {"model": "blitzcrank/ensemble"},
        {"model": "xai/grok-4.3", "use_xai_oauth": True},
        {"model": "anthropic/claude-sonnet-5", "use_anthropic_oauth": True},
        {"model": "chatgpt/gpt-6-luna", "use_anthropic_oauth": True},
    ],
)
def test_chatgpt_opt_in_does_not_unlock_other_private_models(params: dict) -> None:
    with pytest.raises(HTTPException) as exc:
        enforce_public_edge(
            _request(EDGE),
            {"model": "m"},
            key_metadata=CHATGPT_KEY,
            is_admin=False,
            deployments_for_model=[{"litellm_params": params}],
        )
    assert exc.value.status_code == 403


def test_chatgpt_opt_in_still_refuses_mcp_tools() -> None:
    with pytest.raises(HTTPException) as exc:
        enforce_public_edge(
            _request(EDGE),
            {"model": "gpt-6-luna", "tools": [{"type": "mcp", "server_label": "cua_soraka"}]},
            key_metadata=CHATGPT_KEY,
            is_admin=False,
            deployments_for_model=[{"litellm_params": {"model": "chatgpt/gpt-6-luna"}}],
        )
    assert exc.value.status_code == 403
