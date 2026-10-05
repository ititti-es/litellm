"""Restrictions for requests that arrive through the public edge (``X-Liminal-Edge: public``).

The public reverse proxy (Traefik's liminal.ititti.es router) sets this header on every request it
forwards, overwriting any client-supplied value. Requests without it (localhost, tailnet) are not
affected. The header only ever adds restrictions, so a client that sends it on another path just
gets the public treatment; it never unlocks anything.

For public-edge requests liminal enforces, in addition to the edge's own path allowlist:

- ``PublicEdgeMiddleware``: only the inference routes in ``PUBLIC_ROUTES`` and ``/health``. This also covers the
  ASGI apps mounted beside the API router (``/mcp``) that never reach ``user_api_key_auth``.
- ``enforce_public_edge`` (called from ``user_api_key_auth`` once the key is known):
  - the key must be flagged public (``metadata.public`` is ``true``); the master key and other
    admin keys are refused even when flagged;
  - no MCP tools in the body (``tools`` entries of type ``mcp``);
  - no model whose deployments run on the proxy host or on a subscription login: the
    ``claude_cli``, ``chatgpt`` and ``blitzcrank`` providers and ``use_*_oauth`` deployments.
    A key whose metadata sets ``allow_chatgpt_subscription`` to ``true`` may use ``chatgpt``
    models; the other private providers stay refused for every key.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Final

from fastapi import HTTPException, Request, status
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

PUBLIC_EDGE_HEADER: Final = "x-liminal-edge"
PUBLIC_EDGE_VALUE: Final = "public"

# Exact paths (with and without the /v1 prefix) a public-edge request may reach.
PUBLIC_ROUTES: Final = frozenset(
    prefix + path
    for prefix in ("", "/v1")
    for path in (
        "/chat/completions",
        "/responses",
        "/messages",
        "/models",
        "/embeddings",
        "/rerank",
        "/audio/speech",
        "/images/generations",
        "/images/edits",
        "/images/variations",
    )
)
# Model health for the caller's own models; ``health_endpoint`` skips private deployments on the edge.
PUBLIC_EXACT_ROUTES: Final = frozenset({"/health"})
# A stored response can be read back by its id (``/v1/responses/{id}``, ``.../input_items``);
# ownership is still checked by the responses endpoints.
PUBLIC_ROUTE_PREFIXES: Final = ("/responses/", "/v1/responses/")

# Providers that execute on the proxy host or spend a subscription login stored by the proxy.
PRIVATE_PROVIDERS: Final = frozenset({"claude_cli", "chatgpt", "blitzcrank"})
SUBSCRIPTION_FLAGS: Final = ("use_anthropic_oauth", "use_xai_oauth")
# Key metadata opt-in for the ChatGPT subscription provider on the public edge.
CHATGPT_OPT_IN: Final = "allow_chatgpt_subscription"


def is_public_edge(headers: Mapping[str, str]) -> bool:
    return headers.get(PUBLIC_EDGE_HEADER, "").strip().lower() == PUBLIC_EDGE_VALUE


def is_public_route(path: str) -> bool:
    return path in PUBLIC_ROUTES or path in PUBLIC_EXACT_ROUTES or path.startswith(PUBLIC_ROUTE_PREFIXES)


def _route_path(scope: Scope) -> str:
    """The path Starlette routes on: ``path`` without the ASGI ``root_path``."""
    path: Final[str] = scope["path"]
    root_path: Final[str] = scope.get("root_path", "")
    if root_path and path.startswith(root_path + "/"):
        return path[len(root_path) :]
    return path


def _scope_headers(scope: Scope) -> dict[str, str]:
    return {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}


class PublicEdgeMiddleware:
    """Refuse every public-edge request outside ``PUBLIC_ROUTES`` before routing."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket") and is_public_edge(_scope_headers(scope)):
            if not is_public_route(_route_path(scope)):
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                    return
                response: Final = JSONResponse(
                    {"error": {"message": "Not available on the public endpoint.", "type": "forbidden"}},
                    status_code=status.HTTP_403_FORBIDDEN,
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def _deny(code: int, message: str) -> HTTPException:
    return HTTPException(status_code=code, detail={"error": message})


def _has_mcp_tools(request_data: Mapping[str, object]) -> bool:
    tools: Final = request_data.get("tools")
    if not isinstance(tools, list):
        return False
    return any(isinstance(tool, Mapping) and tool.get("type") == "mcp" for tool in tools)


def _is_private_deployment(litellm_params: Mapping[str, object], *, allow_chatgpt: bool) -> bool:
    model: Final = str(litellm_params.get("model") or "")
    provider: Final = str(litellm_params.get("custom_llm_provider") or model.split("/", 1)[0])
    if any(litellm_params.get(flag) for flag in SUBSCRIPTION_FLAGS):
        return True
    return provider in PRIVATE_PROVIDERS and not (allow_chatgpt and provider == "chatgpt")


def private_model(deployments: Iterable[Mapping[str, object]] | None, *, allow_chatgpt: bool = False) -> bool:
    """Whether any deployment of a model group is host-executed or subscription-backed.

    ``allow_chatgpt`` exempts the ``chatgpt`` provider only (see ``CHATGPT_OPT_IN``).
    """
    for deployment in deployments or ():
        params = deployment.get("litellm_params")
        if isinstance(params, Mapping) and _is_private_deployment(params, allow_chatgpt=allow_chatgpt):
            return True
    return False


def enforce_public_edge(
    request: Request,
    request_data: Mapping[str, object],
    *,
    key_metadata: Mapping[str, object] | None,
    is_admin: bool,
    deployments_for_model: Iterable[Mapping[str, object]] | None,
) -> None:
    """Raise 401/403 for a public-edge request the public endpoint must not serve. No-op otherwise."""
    if not is_public_edge(request.headers):
        return
    if is_admin or not key_metadata or key_metadata.get("public") is not True:
        raise _deny(status.HTTP_401_UNAUTHORIZED, "This key is not enabled for the public endpoint.")
    if _has_mcp_tools(request_data):
        raise _deny(status.HTTP_403_FORBIDDEN, "MCP tools are not available on the public endpoint.")
    if private_model(deployments_for_model, allow_chatgpt=key_metadata.get(CHATGPT_OPT_IN) is True):
        raise _deny(status.HTTP_403_FORBIDDEN, "This model is not available on the public endpoint.")
