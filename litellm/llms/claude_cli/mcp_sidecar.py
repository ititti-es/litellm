"""Low-level stdio MCP endpoint for the private tool relay."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, TypedDict

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from pydantic import TypeAdapter
from typing_extensions import NotRequired, ReadOnly

from .tool_bridge import (
    _MAX_RELAY_LINE,  # pyright: ignore[reportPrivateUsage]  # shared private relay protocol limit
    ToolDefinition,
)


class _RelayRequest(TypedDict):
    capability: ReadOnly[str]
    id: ReadOnly[str]
    name: ReadOnly[str]
    arguments: ReadOnly[Mapping[str, object]]


class _RelayResponse(TypedDict):
    content: ReadOnly[str]
    is_error: NotRequired[ReadOnly[bool]]


class _Args(argparse.Namespace):
    socket: str
    directory: str


_RESPONSE_ADAPTER: Final = TypeAdapter(_RelayResponse)
_DEFINITIONS_ADAPTER: Final = TypeAdapter(tuple[ToolDefinition, ...])


async def relay(
    socket: str, capability: str, name: str, arguments: Mapping[str, object], call_id: str
) -> _RelayResponse:
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    reader, writer = await asyncio.open_unix_connection(socket, limit=_MAX_RELAY_LINE)
    try:
        request: Final[_RelayRequest] = {
            "capability": capability,
            "id": call_id,
            "name": name,
            "arguments": arguments,
        }
        writer.write((json.dumps(request) + "\n").encode())
        await writer.drain()
        payload: Final = await reader.readline()
        if not payload or len(payload) > _MAX_RELAY_LINE:
            raise ValueError("external tool relay disconnected")
        return _RESPONSE_ADAPTER.validate_json(payload)
    finally:
        writer.close()
        await writer.wait_closed()


async def serve(socket: str, capability: str, definitions: Sequence[ToolDefinition]) -> None:
    server: Final = Server("litellm_bridge")
    enabled: Final = frozenset(item["name"] for item in definitions)

    @server.list_tools()
    async def list_tools(  # pyright: ignore[reportUnusedFunction]  # registered by decorator
    ) -> list[types.Tool]:  # mutable-ok: MCP SDK fixes the handler return type
        return [  # mutable-ok: MCP SDK requires a list from this handler
            types.Tool(
                name=item["name"],
                description=item["description"],
                inputSchema=item["parameters"],  # pyright: ignore[reportArgumentType]  # SDK requires dict
            )
            for item in definitions
        ]

    @server.call_tool()
    async def call_tool(  # pyright: ignore[reportUnusedFunction]  # registered by decorator
        name: str,
        arguments: dict[str, object],  # mutable-ok: MCP SDK fixes the handler signature
    ) -> types.CallToolResult:
        if name not in enabled:
            raise ValueError("tool is not enabled")
        meta: Final = server.request_context.meta
        call_id: Final = meta.model_extra.get("claudecode/toolUseId") if meta and meta.model_extra else None
        if not isinstance(call_id, str) or not call_id.startswith("toolu_"):
            raise ValueError("Claude tool use ID is missing")
        answer: Final = await relay(socket, capability, name, arguments, call_id)
        return types.CallToolResult(
            content=[  # mutable-ok: MCP SDK requires a list at its response boundary
                types.TextContent(type="text", text=answer["content"])
            ],
            isError=answer.get("is_error", False),
        )

    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())


def main() -> None:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--directory", required=True)
    args: Final = parser.parse_args(namespace=_Args())
    directory: Final = Path(args.directory)
    asyncio.run(
        serve(
            args.socket,
            (directory / "capability").read_text(),
            _DEFINITIONS_ADAPTER.validate_json((directory / "tools.json").read_text()),
        )
    )


if __name__ == "__main__":
    main()
