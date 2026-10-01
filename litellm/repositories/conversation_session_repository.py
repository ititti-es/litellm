"""
Repository for gateway-owned conversation transcripts and compaction policies.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Final, Protocol

from pydantic import JsonValue, TypeAdapter
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.repositories.prisma_protocols import DatabaseClient, TableActions
from litellm.sessions.session import search_text

if TYPE_CHECKING:
    from prisma import models as prisma_models


def jsonb_text(value: object) -> str:
    """JSON for a Postgres jsonb column, which cannot hold NUL; it becomes U+FFFD, the replacement character"""
    return json.dumps(_without_nul(value))


def _without_nul(value: object) -> object:
    if isinstance(value, str):
        return value.replace("\x00", "\ufffd")
    if isinstance(value, Mapping):
        return {_without_nul(key): _without_nul(item) for key, item in value.items()}  # mutable-ok: JSON payload
    if isinstance(value, list | tuple):
        return [_without_nul(item) for item in value]  # mutable-ok: JSON payload
    return value


@dataclass(frozen=True, slots=True)
class ItemRow:
    seq: int
    content: str
    is_output: bool
    meta: str | None = None


@dataclass(frozen=True, slots=True)
class ImportWrite:
    session_id: str
    owner: str
    version: int
    source: str
    source_id: str
    title: str | None
    parent_session_id: str | None
    metadata: str
    created_at: datetime
    last_upstream_at: datetime
    items: tuple[ItemRow, ...]
    responses: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class StoredTranscript:
    version: int
    contents: tuple[object, ...]
    owner: str | None = None
    """Who holds it now; an import must not take a session away from the key that is using it"""


@dataclass(frozen=True, slots=True)
class TurnWrite:
    session_id: str
    owner: str
    version: int
    last_upstream_at: datetime
    expires_at: datetime | None
    items: tuple[ItemRow, ...]
    response_id: str
    item_count: int
    metadata: str | None = None


class _TurnSessionFields(TypedDict):
    owner: ReadOnly[str]
    version: ReadOnly[int]
    last_upstream_at: ReadOnly[datetime]
    expires_at: ReadOnly[datetime | None]
    metadata: NotRequired[ReadOnly[str]]


@dataclass(frozen=True, slots=True)
class StoredSession:
    session_id: str
    owner: str
    version: int
    last_upstream_at: datetime
    items: tuple["prisma_models.LiteLLM_ConversationItem", ...]
    responses: tuple["prisma_models.LiteLLM_ConversationResponse", ...]
    metadata: object = None


class _BatchWriteTable(Protocol):
    def create_many(self, data: Sequence[Mapping[str, object]], *, skip_duplicates: bool | None = None) -> None: ...

    def upsert(self, where: Mapping[str, object], data: Mapping[str, object]) -> None: ...


class _ConversationBatch(Protocol):
    @property
    def litellm_conversationsession(self) -> _BatchWriteTable: ...

    @property
    def litellm_conversationitem(self) -> _BatchWriteTable: ...

    @property
    def litellm_conversationresponse(self) -> _BatchWriteTable: ...

    async def commit(self) -> None: ...


class _ConversationDb(Protocol):
    @property
    def litellm_conversationsession(self) -> TableActions["prisma_models.LiteLLM_ConversationSession"]: ...

    @property
    def litellm_conversationitem(self) -> TableActions["prisma_models.LiteLLM_ConversationItem"]: ...

    @property
    def litellm_conversationresponse(self) -> TableActions["prisma_models.LiteLLM_ConversationResponse"]: ...

    @property
    def litellm_compactionpolicy(self) -> TableActions["prisma_models.LiteLLM_CompactionPolicy"]: ...

    def batch_(self) -> _ConversationBatch: ...


_ITEM: Final = TypeAdapter(dict[str, JsonValue])


class TranscriptConflict(Exception):
    pass


class ConversationSessionRepository:
    def __init__(self, prisma_client: DatabaseClient) -> None:
        self._prisma_client: Final = prisma_client

    @property
    def _db(self) -> _ConversationDb:
        return self._prisma_client.db  # pyright: ignore[reportReturnType]  # generated client satisfies the protocol

    async def save_turn(self, write: TurnWrite) -> None:
        """One committed turn, atomically: the session row, its new items, and the response that addresses them

        Another worker that moved the session on first wins: a newer stored version is refused here, and an item
        seq it already wrote fails the batch, so a response never addresses items it did not produce
        """
        stored: Final = await self._db.litellm_conversationsession.find_unique(
            where={"session_id": write.session_id}  # mutable-ok: Prisma where filter must be a dict
        )
        if stored is not None and stored.version > write.version:
            raise TranscriptConflict(
                f"session {write.session_id} is at version {stored.version}, this write is for {write.version}"
            )
        batch: Final = self._db.batch_()
        fields: Final[_TurnSessionFields] = {
            "owner": write.owner,
            "version": write.version,
            "last_upstream_at": write.last_upstream_at,
            "expires_at": write.expires_at,
        }
        session_fields: Final = _turn_session_fields(fields, write.metadata)
        batch.litellm_conversationsession.upsert(
            where={"session_id": write.session_id},  # mutable-ok: Prisma where filter must be a dict
            data={  # mutable-ok: Prisma upsert payload must be a dict
                "create": {"session_id": write.session_id, **session_fields},  # mutable-ok: Prisma create payload
                "update": session_fields,
            },
        )
        batch.litellm_conversationitem.create_many(
            data=[  # mutable-ok: Prisma create_many takes a list of dicts
                _item_row(write.session_id, write.version, item) for item in write.items
            ],
        )
        response_fields: Final = {  # mutable-ok: Prisma payload must be a dict
            "session_id": write.session_id,
            "version": write.version,
            "item_count": write.item_count,
        }
        batch.litellm_conversationresponse.upsert(
            where={"response_id": write.response_id},  # mutable-ok: Prisma where filter must be a dict
            data={  # mutable-ok: Prisma upsert payload must be a dict
                "create": {"response_id": write.response_id, **response_fields},  # mutable-ok: Prisma create payload
                "update": response_fields,
            },
        )
        await batch.commit()

    async def import_session(self, write: ImportWrite) -> None:
        """An imported transcript, atomically; imports are pinned, so retention never removes them"""
        batch: Final = self._db.batch_()
        fields: Final = {  # mutable-ok: Prisma payload must be a dict
            "owner": write.owner,
            "version": write.version,
            "last_upstream_at": write.last_upstream_at,
            "expires_at": None,
            "pinned": True,
            "title": write.title,
            "parent_session_id": write.parent_session_id,
            "metadata": write.metadata,
        }
        batch.litellm_conversationsession.upsert(
            where={"session_id": write.session_id},  # mutable-ok: Prisma where filter must be a dict
            data={  # mutable-ok: Prisma upsert payload must be a dict
                "create": {  # mutable-ok: Prisma create payload
                    "session_id": write.session_id,
                    "source": write.source,
                    "source_id": write.source_id,
                    "created_at": write.created_at,
                    **fields,
                },
                "update": fields,
            },
        )
        batch.litellm_conversationitem.create_many(
            data=[_item_row(write.session_id, write.version, item) for item in write.items],  # mutable-ok: Prisma rows
            skip_duplicates=True,
        )
        for response_id, item_count in write.responses:
            batch.litellm_conversationresponse.upsert(
                where={"response_id": response_id},  # mutable-ok: Prisma where filter must be a dict
                data=_response_upsert(response_id, write.session_id, write.version, item_count),
            )
        await batch.commit()

    async def current_transcript(self, session_id: str) -> StoredTranscript | None:
        session: Final = await self._db.litellm_conversationsession.find_unique(
            where={"session_id": session_id}  # mutable-ok: Prisma where filter must be a dict
        )
        if session is None:
            return None
        items: Final = await self._db.litellm_conversationitem.find_many(
            where={"session_id": session_id, "version": session.version},  # mutable-ok: Prisma where filter
            order={"seq": "asc"},  # mutable-ok: Prisma order clause must be a dict
        )
        return StoredTranscript(
            version=session.version, contents=tuple(item.content for item in items), owner=session.owner
        )

    async def load_by_response(self, response_id: str) -> StoredSession | None:
        response: Final = await self._db.litellm_conversationresponse.find_unique(
            where={"response_id": response_id}  # mutable-ok: Prisma where filter must be a dict
        )
        if response is None:
            return None
        session: Final = await self._db.litellm_conversationsession.find_unique(
            where={"session_id": response.session_id}  # mutable-ok: Prisma where filter must be a dict
        )
        if session is None or session.version != response.version:
            return None
        return await self._transcript(session)

    async def load_session(self, session_id: str) -> StoredSession | None:
        """A session by its own id, for a client that names the conversation rather than the turn it follows"""
        session: Final = await self._db.litellm_conversationsession.find_unique(
            where={"session_id": session_id}  # mutable-ok: Prisma where filter must be a dict
        )
        return await self._transcript(session) if session is not None else None

    async def load_session_history(self, session_id: str) -> StoredSession | None:
        """A session with the items of every stored version, oldest first

        Compaction rebases a session onto a summary as a new version and leaves the earlier versions' rows in
        place, so this is the whole conversation as it was had, not just what the next turn replays.
        """
        session: Final = await self._db.litellm_conversationsession.find_unique(
            where={"session_id": session_id}  # mutable-ok: Prisma where filter must be a dict
        )
        if session is None:
            return None
        items: Final = await self._db.litellm_conversationitem.find_many(
            where={"session_id": session_id},  # mutable-ok: Prisma where filter must be a dict
            order=[{"version": "asc"}, {"seq": "asc"}],  # mutable-ok: Prisma order clause must be a list of dicts
        )
        responses: Final = await self._db.litellm_conversationresponse.find_many(
            where={"session_id": session_id}  # mutable-ok: Prisma where filter must be a dict
        )
        return StoredSession(
            session_id=session.session_id,
            owner=session.owner,
            version=session.version,
            last_upstream_at=session.last_upstream_at,
            metadata=session.metadata,
            items=tuple(items),
            responses=tuple(responses),
        )

    async def _transcript(self, session: "prisma_models.LiteLLM_ConversationSession") -> StoredSession:
        scope: Final = {  # mutable-ok: Prisma where filter must be a dict
            "session_id": session.session_id,
            "version": session.version,
        }
        items: Final = await self._db.litellm_conversationitem.find_many(
            where=scope,
            order={"seq": "asc"},  # mutable-ok: Prisma order clause must be a dict
        )
        responses: Final = await self._db.litellm_conversationresponse.find_many(where=scope)
        return StoredSession(
            session_id=session.session_id,
            owner=session.owner,
            version=session.version,
            last_upstream_at=session.last_upstream_at,
            metadata=session.metadata,
            items=tuple(items),
            responses=tuple(responses),
        )

    async def delete_session(self, session_id: str) -> None:
        await self._db.litellm_conversationsession.delete_many(
            where={"session_id": session_id}  # mutable-ok: Prisma where filter must be a dict
        )

    async def delete_expired(self, now: datetime) -> int:
        """Pinned sessions are permanent whatever their expiry says"""
        return await self._db.litellm_conversationsession.delete_many(
            where={"expires_at": {"lt": now}, "pinned": False}  # mutable-ok: Prisma where filter must be a dict
        )

    async def set_pinned(self, session_id: str, pinned: bool) -> bool:
        data: Final = _pin_payload(pinned)
        updated: Final = await self._db.litellm_conversationsession.update_many(
            where={"session_id": session_id},  # mutable-ok: Prisma where filter must be a dict
            data=data,
        )
        return updated > 0

    async def upsert_policy(
        self, name: str, source: str, instructions: str, settings: Mapping[str, object] | None, actor: str | None
    ) -> "prisma_models.LiteLLM_CompactionPolicy":
        fields: Final = {  # mutable-ok: Prisma payload must be a dict
            "source": source,
            "instructions": instructions,
            "settings": json.dumps(settings) if settings is not None else None,
            "updated_by": actor,
        }
        return await self._db.litellm_compactionpolicy.upsert(
            where={"name": name},  # mutable-ok: Prisma where filter must be a dict
            data={  # mutable-ok: Prisma upsert payload must be a dict
                "create": {"name": name, "created_by": actor, **fields},  # mutable-ok: Prisma create payload
                "update": fields,
            },
        )

    async def find_policy(self, name: str) -> "prisma_models.LiteLLM_CompactionPolicy | None":
        return await self._db.litellm_compactionpolicy.find_unique(
            where={"name": name}  # mutable-ok: Prisma where filter must be a dict
        )

    async def list_policies(self) -> Sequence["prisma_models.LiteLLM_CompactionPolicy"]:
        return await self._db.litellm_compactionpolicy.find_many(
            order={"name": "asc"}  # mutable-ok: Prisma order clause must be a dict
        )

    async def delete_policy(self, name: str) -> bool:
        deleted: Final = await self._db.litellm_compactionpolicy.delete_many(
            where={"name": name}  # mutable-ok: Prisma where filter must be a dict
        )
        return deleted > 0


def _turn_session_fields(fields: _TurnSessionFields, metadata: str | None) -> _TurnSessionFields:
    if metadata is None:
        return fields
    with_metadata: Final[_TurnSessionFields] = {**fields, "metadata": metadata}
    return with_metadata


def _item_row(session_id: str, version: int, item: ItemRow) -> Mapping[str, object]:
    row: Final = {  # mutable-ok: Prisma row payload
        "session_id": session_id,
        "version": version,
        "seq": item.seq,
        "content": item.content,
        "is_output": item.is_output,
        "search_text": search_text(_ITEM.validate_json(item.content)) or "",
    }
    return {**row, "meta": item.meta} if item.meta is not None else row  # mutable-ok: Prisma row payload


def _response_upsert(response_id: str, session_id: str, version: int, item_count: int) -> Mapping[str, object]:
    fields: Final = {"session_id": session_id, "version": version, "item_count": item_count}  # mutable-ok: Prisma
    return {"create": {"response_id": response_id, **fields}, "update": fields}  # mutable-ok: Prisma upsert payload


def _pin_payload(pinned: bool) -> Mapping[str, object]:
    return {"pinned": True, "expires_at": None} if pinned else {"pinned": False}  # mutable-ok: Prisma update payload
