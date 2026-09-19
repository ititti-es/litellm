"""
Repository for gateway-owned conversation transcripts and compaction policies.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Final, Protocol

from litellm.repositories.prisma_protocols import DatabaseClient, TableActions

if TYPE_CHECKING:
    from prisma import models as prisma_models


@dataclass(frozen=True, slots=True)
class ItemRow:
    seq: int
    content: str
    is_output: bool


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


@dataclass(frozen=True, slots=True)
class StoredSession:
    session_id: str
    owner: str
    version: int
    last_upstream_at: datetime
    items: tuple["prisma_models.LiteLLM_ConversationItem", ...]
    responses: tuple["prisma_models.LiteLLM_ConversationResponse", ...]


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


class ConversationSessionRepository:
    def __init__(self, prisma_client: DatabaseClient) -> None:
        self._prisma_client: Final = prisma_client

    @property
    def _db(self) -> _ConversationDb:
        return self._prisma_client.db  # pyright: ignore[reportReturnType]  # generated client satisfies the protocol

    async def save_turn(self, write: TurnWrite) -> None:
        """One committed turn, atomically: the session row, its new items, and the response that addresses them"""
        batch: Final = self._db.batch_()
        session_fields: Final = {  # mutable-ok: Prisma payload must be a dict
            "owner": write.owner,
            "version": write.version,
            "last_upstream_at": write.last_upstream_at,
            "expires_at": write.expires_at,
        }
        batch.litellm_conversationsession.upsert(
            where={"session_id": write.session_id},  # mutable-ok: Prisma where filter must be a dict
            data={  # mutable-ok: Prisma upsert payload must be a dict
                "create": {"session_id": write.session_id, **session_fields},  # mutable-ok: Prisma create payload
                "update": session_fields,
            },
        )
        batch.litellm_conversationitem.create_many(
            data=[  # mutable-ok: Prisma create_many takes a list of dicts
                {  # mutable-ok: one Prisma row payload per item
                    "session_id": write.session_id,
                    "version": write.version,
                    "seq": item.seq,
                    "content": item.content,
                    "is_output": item.is_output,
                }
                for item in write.items
            ],
            skip_duplicates=True,
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
            items=tuple(items),
            responses=tuple(responses),
        )

    async def delete_session(self, session_id: str) -> None:
        await self._db.litellm_conversationsession.delete_many(
            where={"session_id": session_id}  # mutable-ok: Prisma where filter must be a dict
        )

    async def delete_expired(self, now: datetime) -> int:
        return await self._db.litellm_conversationsession.delete_many(
            where={"expires_at": {"lt": now}}  # mutable-ok: Prisma where filter must be a dict
        )

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
