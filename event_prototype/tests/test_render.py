"""The triage prompt has to show the model every event it is accountable for."""

from __future__ import annotations

from event_prototype.fixtures import seed
from event_prototype.queue import EventQueue
from event_prototype.render import PromptRenderer
from event_prototype.store import IN_MEMORY


async def test_triage_prompt_covers_every_active_event() -> None:
    async with await EventQueue.open(IN_MEMORY) as queue:
        ids = await seed(queue)
        rows = await queue.list_events()
        prompt = PromptRenderer().triage(rows)

    assert len(ids) == len(rows)
    for row in rows:
        assert f'<event id="{row.id}"' in prompt
        assert row.description in prompt

    # The tool names are the model's whole vocabulary — they must be named.
    for tool in ("handle_one_event", "handle_event_sequence", "defer_event"):
        assert tool in prompt


async def test_subagent_prompt_carries_the_brief() -> None:
    async with await EventQueue.open(IN_MEMORY) as queue:
        await seed(queue)
        rows = await queue.list_events()
        prompt = PromptRenderer().subagent(rows[:2], "Answer Mira, then post the devlog.")

    assert "Answer Mira, then post the devlog." in prompt
    assert "in the order given above" in prompt  # sequence framing for >1 event
