import json
from types import SimpleNamespace

import pytest

from litellm.proxy.compaction_policies import from_text, listing, resolve, seed, store, to_yaml
from litellm.sessions.compaction_presets import OPENCODE, PRESETS


class Policies:
    def __init__(self):
        self.rows = {}

    async def upsert_policy(self, name, source, instructions, settings, actor):
        self.rows[name] = SimpleNamespace(
            name=name, source=source, instructions=instructions, settings=json.dumps(settings)
        )

    async def find_policy(self, name):
        return self.rows.get(name)

    async def list_policies(self):
        return [self.rows[name] for name in sorted(self.rows)]


async def test_seeding_stores_editable_copies_and_never_overwrites_without_force():
    repository = Policies()
    assert await seed(repository, ("opencode",), force=False) == ("opencode: stored",)
    assert await resolve(repository, "opencode") == OPENCODE

    edited = from_text(to_yaml(OPENCODE).replace("What did we do so far?", "Where were we?"))
    await store(repository, edited)
    assert await seed(repository, ("opencode",), force=False) == (
        "opencode: already stored (use --force to overwrite)",
    )
    assert (await resolve(repository, "opencode")).settings.summary_lead_in == "Where were we?"
    await seed(repository, ("opencode",), force=True)
    assert await resolve(repository, "opencode") == OPENCODE

    with pytest.raises(ValueError):
        await seed(repository, ("nope",), force=False)


async def test_listing_shows_presets_stored_policies_and_overrides():
    repository = Policies()
    await seed(repository, ("hermes",), force=False)
    await store(repository, from_text('{"name": "terse", "instructions": "Summarize in five bullets."}'))
    lines = await listing(repository)
    assert len(lines) == len(PRESETS) + 1
    assert any(line.startswith("hermes") and "overrides preset" in line for line in lines)
    assert any(line.startswith("terse") and "stored" in line for line in lines)
    assert any(line.startswith("openclaw") and "built-in" in line for line in lines)
    assert await resolve(repository, "missing") is None


def test_the_editable_form_keeps_prompts_as_readable_blocks_and_round_trips():
    for policy in PRESETS.values():
        text = to_yaml(policy)
        assert from_text(text) == policy
    assert "instructions: |" in to_yaml(OPENCODE)
    assert "## Objective\n" in to_yaml(OPENCODE)
