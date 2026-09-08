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


async def test_the_subagent_conversation_splits_into_stable_and_per_turn_parts() -> None:
    renderer = PromptRenderer()
    async with await EventQueue.open(IN_MEMORY) as queue:
        await seed(queue)
        row = (await queue.list_events())[0]
        event = renderer.event(row)

    # The system prompt carries no event, so it reads the same on every call...
    system = renderer.subagent_system()
    assert row.description not in system
    assert system == renderer.subagent_system()
    # ...each event is one turn...
    assert f'<event id="{row.id}"' in event
    assert row.description in event
    # ...and the brief follows them, framed for a sequence only when it is one.
    brief = renderer.brief("Answer Mira, then post the devlog.", sequence=True)
    assert "Answer Mira, then post the devlog." in brief
    assert "in the order given" in brief
    assert "in the order given" not in renderer.brief("Answer Mira.", sequence=False)
