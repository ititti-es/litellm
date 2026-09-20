"""Provider OAuth credentials kept in the proxy's database rather than on one host's disk.

A file under `~/.config/litellm` is right for a workstation and wrong for a gateway: it does not survive a
container without a mounted volume, and replicas each have their own, so each has to log in separately and
each refreshes independently.

Records live in `LiteLLM_CredentialsTable`, the table the proxy already uses for provider credentials, under
the name `oauth/<provider>`. Values are encrypted with LiteLLM's own helper, so they are protected by
`LITELLM_SALT_KEY` (or the master key) exactly like every other stored credential.

Reads happen on the request path, which is synchronous, so this uses a synchronous connection rather than
the proxy's async Prisma client.
"""

import hashlib
import json
import time
from collections.abc import Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from litellm._logging import verbose_logger
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper

CREDENTIAL_NAME_PREFIX: Final = "oauth/"
CREDENTIAL_KIND: Final = "provider_oauth"

# Postgres advisory locks are keyed on a bigint. Sharing one namespace across providers would serialise
# unrelated refreshes, so the provider name is folded into the key.
_LOCK_NAMESPACE: Final = 0x4C4C4D4F  # "LLMO"

_TABLE: Final = '"LiteLLM_CredentialsTable"'
_CACHE_SECONDS: Final = 5.0

# Prisma accepts pool settings on the URL that libpq knows nothing about, and the proxy appends several to
# DATABASE_URL before startup; psycopg rejects the whole URL if one is present. Allowing the parameters
# libpq documents, rather than excluding the ones Prisma adds, means a new Prisma setting cannot break
# this again -- only a genuinely new libpq parameter would need adding.
_LIBPQ_PARAMS: Final = frozenset(
    {
        "application_name",
        "channel_binding",
        "client_encoding",
        "connect_timeout",
        "fallback_application_name",
        "gssencmode",
        "hostaddr",
        "keepalives",
        "keepalives_count",
        "keepalives_idle",
        "keepalives_interval",
        "options",
        "passfile",
        "require_auth",
        "service",
        "sslcert",
        "sslcrl",
        "sslkey",
        "sslmode",
        "sslpassword",
        "sslrootcert",
        "sslsni",
        "target_session_attrs",
    }
)


def libpq_url(database_url: str) -> str:
    """Return `database_url` carrying only the query parameters libpq accepts."""
    parsed: Final = urlsplit(database_url)
    kept: Final = tuple((k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k in _LIBPQ_PARAMS)
    return urlunsplit(parsed._replace(query=urlencode(kept)))


def _credential_info(provider: str) -> dict[str, str]:
    return {"kind": CREDENTIAL_KIND, "provider": provider}  # mutable-ok: serialised immediately


def credential_name(provider: str) -> str:
    return f"{CREDENTIAL_NAME_PREFIX}{provider}"


def _lock_key(provider: str) -> int:
    """A 63-bit advisory-lock key for one provider.

    Derived from a cryptographic digest rather than `hash()`, whose string hashing is randomised per process:
    two replicas would otherwise compute different keys and coordinate with nothing.
    """
    digest: Final = int.from_bytes(hashlib.sha256(provider.encode()).digest()[:4], "big") & 0x7FFFFFFF
    return (_LOCK_NAMESPACE << 31) | digest


def _encrypted(record: Mapping[str, object]) -> dict[str, str]:
    encrypted = {}  # mutable-ok: built in one pass and handed straight to the database
    for key, value in record.items():
        encrypted[key] = encrypt_value_helper(json.dumps(value))
    return encrypted


def _decrypted(stored: Mapping[str, object]) -> dict[str, object] | None:
    decoded = {}  # mutable-ok: the decrypted record returned to the caller
    for key, value in stored.items():
        if not isinstance(value, str):
            return None
        plain = decrypt_value_helper(value, key)
        if plain is None:
            verbose_logger.warning("Could not decrypt stored OAuth field %r; is LITELLM_SALT_KEY set?", key)
            return None
        try:
            decoded[key] = json.loads(plain)
        except json.JSONDecodeError:
            decoded[key] = plain
    return decoded


class DatabaseCredentialStore:
    """Reads and writes one provider's OAuth record in the proxy database."""

    def __init__(self, provider: str, database_url: str, fallback_path: str | None = None) -> None:
        self.provider = provider
        self.database_url = database_url
        self.fallback_path = fallback_path
        self.name = credential_name(provider)
        self._cached: object | None = None
        self._cached_at = 0.0

    def _connect(self):
        import psycopg

        return psycopg.connect(libpq_url(self.database_url))

    def read(self) -> object | None:
        """Return the stored record, briefly cached so a burst of requests is one query."""
        now: Final = time.time()
        if self._cached is not None and now - self._cached_at < _CACHE_SECONDS:
            return self._cached
        record: Final = self._read_uncached()
        self._cached, self._cached_at = record, now
        return record

    def _read_uncached(self) -> object | None:
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    f"SELECT credential_values FROM {_TABLE} WHERE credential_name = %s",
                    (self.name,),
                )
                row = cur.fetchone()
        except Exception as exc:  # noqa: BLE001  # any database failure degrades to the local file
            verbose_logger.warning("Could not read %s credentials from the database: %s", self.provider, exc)
            return self._from_fallback()
        if not row or not isinstance(row[0], dict):
            return None
        return _decrypted(row[0])

    def _from_fallback(self) -> object | None:
        """Fall back to the local file when the database is unreachable, so a blip is not an outage."""
        if not self.fallback_path:
            return None
        from litellm.llms.base_llm.oauth import TokenFile

        return TokenFile(self.fallback_path).read()

    def write(self, data: object) -> None:
        if not isinstance(data, Mapping):
            raise TypeError("An OAuth credential record must be a mapping")
        values: Final = json.dumps(_encrypted(data))
        info: Final = json.dumps(_credential_info(self.provider))
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {_TABLE}
                    (credential_id, credential_name, credential_values, credential_info,
                     created_at, created_by, updated_at, updated_by)
                VALUES (gen_random_uuid()::text, %s, %s::jsonb, %s::jsonb, now(), %s, now(), %s)
                ON CONFLICT (credential_name) DO UPDATE
                    SET credential_values = EXCLUDED.credential_values,
                        credential_info = EXCLUDED.credential_info,
                        updated_at = now(),
                        updated_by = EXCLUDED.updated_by
                """,
                (self.name, values, info, CREDENTIAL_KIND, CREDENTIAL_KIND),
            )
            conn.commit()
        self._cached, self._cached_at = dict(data), time.time()  # mutable-ok: a snapshot, not the caller's

    def delete(self) -> bool:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {_TABLE} WHERE credential_name = %s",
                (self.name,),
            )
            removed = cur.rowcount > 0
            conn.commit()
        self._cached, self._cached_at = None, 0.0
        return removed

    def guard(self) -> AbstractContextManager[None]:
        return self._guard()

    @contextmanager
    def _guard(self) -> Generator[None]:
        """Hold a Postgres advisory lock for the length of a refresh.

        Refresh tokens rotate, so two replicas refreshing from the same record would each invalidate the
        other's and leave one of them unable to refresh again. The lock is released when the transaction
        ends, including if this process dies.
        """
        try:
            conn = self._connect()
        except Exception as exc:  # noqa: BLE001  # without the database there is nothing to coordinate with
            # The caller's own in-process lock still applies.
            verbose_logger.warning("Proceeding without a %s refresh lock: %s", self.provider, exc)
            yield
            return
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_xact_lock(%s)", (_lock_key(self.provider),))
                self._cached = None  # whoever held the lock may have written a newer record
                yield
        finally:
            conn.rollback()  # ends the transaction, releasing the lock
            conn.close()


def database_store_factory(database_url: str):
    """Return a factory the OAuth modules can use to build a store per provider."""

    def factory(provider: str, path: str) -> DatabaseCredentialStore:
        return DatabaseCredentialStore(provider, database_url, fallback_path=path)

    return factory


def use_database_oauth_credentials(setting: object) -> bool:
    """Point the OAuth modules at the database when `general_settings` asks for it.

    Returns whether the database store was installed. Left unset it defaults to on wherever a DATABASE_URL
    exists, since a proxy with a database is the case this store is for; `false` keeps the files.
    """
    import os

    from litellm.llms.base_llm.oauth import set_credential_store_factory

    if setting is False:
        set_credential_store_factory(None)
        return False
    database_url: Final = os.getenv("DATABASE_URL") or ""
    if not database_url:
        if setting is True:
            verbose_logger.warning("store_oauth_credentials_in_db is set but DATABASE_URL is not; using files")
        set_credential_store_factory(None)
        return False
    set_credential_store_factory(database_store_factory(database_url))
    verbose_logger.info("Provider OAuth credentials will be read from and written to the database")
    return True
