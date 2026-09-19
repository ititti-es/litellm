"""Manage the session manager's compaction policies

    python -m litellm.proxy.compaction_policies list
    python -m litellm.proxy.compaction_policies show NAME            > policy.yaml
    python -m litellm.proxy.compaction_policies set policy.yaml      # create or replace (YAML or JSON)
    python -m litellm.proxy.compaction_policies seed [NAME ...] [--force]
    python -m litellm.proxy.compaction_policies delete NAME

Built-in presets (default, opencode, openclaw, hermes) work without being stored. A stored policy with the same
name replaces the preset, so `seed opencode` then `show`/`set` gives an editable copy. Select a policy with
`general_settings.session_manager.compaction.policy`, or per request with `metadata: {"compaction_policy": NAME}`.
See litellm/sessions/compaction_policy.py for every field.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Final

import yaml

from litellm.proxy.session_import import repository_from_url
from litellm.repositories.conversation_session_repository import ConversationSessionRepository
from litellm.sessions.compaction_policy import CompactionPolicy, policy_from, policy_from_document
from litellm.sessions.compaction_presets import PRESETS

_ACTOR: Final = "compaction_policies_cli"


class _BlockDumper(yaml.SafeDumper):
    """Writes multi-line text as literal blocks, so prompts read and edit as they are written"""


def _represent_text(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|" if "\n" in value else None)


_BlockDumper.add_representer(str, _represent_text)


def to_yaml(policy: CompactionPolicy) -> str:
    return yaml.dump(dict(policy.document()), Dumper=_BlockDumper, sort_keys=False, allow_unicode=True, width=120)


def from_text(text: str) -> CompactionPolicy:
    """A policy from its YAML or JSON form (JSON is YAML too)"""
    return policy_from_document(yaml.safe_load(text))


async def listing(repository: ConversationSessionRepository) -> tuple[str, ...]:
    stored: Final = MappingProxyType({policy.name: policy for policy in await repository.list_policies()})
    names: Final = sorted(frozenset(PRESETS) | frozenset(stored))
    return tuple(
        f"{name:<20} {'stored' if name in stored else 'built-in':<9} "
        f"{'(overrides preset) ' if name in stored and name in PRESETS else ''}"
        f"{stored[name].source if name in stored else PRESETS[name].source}"
        for name in names
    )


async def resolve(repository: ConversationSessionRepository, name: str) -> CompactionPolicy | None:
    stored: Final = await repository.find_policy(name)
    if stored is not None:
        return policy_from(stored.name, stored.source, stored.instructions, stored.settings)
    return PRESETS.get(name)


async def store(repository: ConversationSessionRepository, policy: CompactionPolicy) -> None:
    await repository.upsert_policy(
        policy.name,
        policy.source,
        policy.instructions,
        policy.settings.model_dump(exclude_defaults=True),
        _ACTOR,
    )


async def seed(repository: ConversationSessionRepository, names: tuple[str, ...], force: bool) -> tuple[str, ...]:
    unknown: Final = tuple(name for name in names if name not in PRESETS)
    if unknown:
        raise ValueError(f"no built-in preset named {', '.join(unknown)}; presets are {', '.join(PRESETS)}")
    lines: Final[list[str]] = []  # mutable-ok: one report line per preset
    for name in names or tuple(PRESETS):
        if not force and await repository.find_policy(name) is not None:
            lines.append(f"{name}: already stored (use --force to overwrite)")
            continue
        await store(repository, PRESETS[name])
        lines.append(f"{name}: stored")
    return tuple(lines)


async def _main(args: argparse.Namespace) -> int:
    url: Final = args.database_url or os.environ.get("DATABASE_URL") or None
    if args.command == "show" and url is None and args.name in PRESETS:
        sys.stdout.write(to_yaml(PRESETS[args.name]))
        return 0
    if url is None:
        sys.stderr.write("DATABASE_URL (or --database-url) is required\n")
        return 2
    async with repository_from_url(url) as repository:
        if repository is None:
            return 2
        if args.command == "list":
            sys.stdout.write("".join(line + "\n" for line in await listing(repository)))
            return 0
        if args.command == "show":
            found: Final = await resolve(repository, args.name)
            if found is None:
                sys.stderr.write(f"no policy named {args.name}\n")
                return 1
            sys.stdout.write(to_yaml(found))
            return 0
        if args.command == "set":
            policy: Final = from_text(Path(args.file).expanduser().read_text())
            await store(repository, policy)
            sys.stdout.write(f"{policy.name}: stored\n")
            return 0
        if args.command == "seed":
            sys.stdout.write("".join(line + "\n" for line in await seed(repository, tuple(args.names), args.force)))
            return 0
        deleted: Final = await repository.delete_policy(args.name)
        sys.stdout.write(f"{args.name}: {'deleted' if deleted else 'not stored'}\n")
        return 0 if deleted else 1


def _parser() -> argparse.ArgumentParser:
    parser: Final = argparse.ArgumentParser(
        prog="python -m litellm.proxy.compaction_policies", description=(__doc__ or "").split("\n")[0]
    )
    parser.add_argument("--database-url", help="defaults to $DATABASE_URL")
    commands: Final = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="stored policies and built-in presets")
    show: Final = commands.add_parser("show", help="print a policy as editable YAML")
    show.add_argument("name")
    replace: Final = commands.add_parser("set", help="create or replace a policy from its YAML or JSON form")
    replace.add_argument("file")
    seed_command: Final = commands.add_parser("seed", help="store built-in presets so they can be edited")
    seed_command.add_argument("names", nargs="*")
    seed_command.add_argument("--force", action="store_true", help="overwrite stored policies of the same name")
    delete: Final = commands.add_parser("delete", help="remove a stored policy (a preset of that name returns)")
    delete.add_argument("name")
    return parser


def main() -> int:
    return asyncio.run(_main(_parser().parse_args()))


if __name__ == "__main__":
    sys.exit(main())
