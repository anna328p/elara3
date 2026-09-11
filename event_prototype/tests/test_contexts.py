"""Contexts: a stream's conversation, stored as turns and replayed as a call."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from anthropic import Omit
from anthropic.types import MessageParam

from event_prototype.agents import AgentRole
from event_prototype.config import Config
from event_prototype.contexts import Block, PendingToolCalls, Role, Turn, to_api
from event_prototype.events import Priority
from event_prototype.queue import EventQueue
from event_prototype.render import PromptRenderer
from event_prototype.store import Action, Status
from event_prototype.tools import Dispatcher

from conftest import FakeMessages, completion, counting_selects, message

# -- the transform, no database needed --------------------------------------

TOOL_USE: dict[str, Any] = {
    "type": "tool_use",
    "id": "toolu_01",
    "name": "send_message",
    "input": {"venue": "discord", "body": "on it"},
}
TOOL_RESULT: dict[str, Any] = {"type": "tool_result", "tool_use_id": "toolu_01", "content": "sent"}
THINKING: dict[str, Any] = {"type": "thinking", "thinking": "hm", "signature": "sig=="}


def blocks(message: MessageParam) -> list[Block]:
    """The content of a sent message, in the shape the tests read it."""
    return cast(list[Block], message["content"])


def test_consecutive_same_role_turns_become_one_message() -> None:
    turns = [
        Turn.user("first ping", event_id=1),
        Turn.user("second ping", event_id=2),
        Turn.user("the brief"),
        Turn(Role.ASSISTANT, ({"type": "text", "text": "done"},)),
    ]

    messages = to_api(turns)

    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert [b["text"] for b in blocks(messages[0])] == ["first ping", "second ping", "the brief"]


def test_tool_results_lead_the_user_message_they_land_in() -> None:
    turns = [
        Turn.user("do the thing"),
        Turn(Role.ASSISTANT, (THINKING, TOOL_USE)),
        Turn(Role.USER, (TOOL_RESULT,)),
        Turn.user("another ping arrived meanwhile", event_id=7),
    ]

    messages = to_api(turns)

    assert [b["type"] for b in blocks(messages[2])] == ["tool_result", "text"]
    # The assistant turn is passed through untouched, signature and all.
    assert blocks(messages[1]) == [THINKING, TOOL_USE]


def test_a_transcript_awaiting_tool_results_is_not_sendable() -> None:
    turns = [Turn.user("do the thing"), Turn(Role.ASSISTANT, (TOOL_USE,))]

    with pytest.raises(PendingToolCalls) as caught:
        to_api(turns)
    assert caught.value.tool_use_ids == ("toolu_01",)

    # Answering it makes the same transcript sendable again.
    to_api([*turns, Turn(Role.USER, (TOOL_RESULT,))])


def test_the_sent_content_is_the_stored_content() -> None:
    """Caching is a request-level decision; nothing about it is written into a turn."""
    reply = Turn.of(completion("done"))

    sent = to_api([Turn.user("hi"), reply])

    assert reply.content == ({"type": "text", "text": "done"},)
    assert "cache_control" not in str(sent)


# -- persistence ----------------------------------------------------------


async def test_a_turn_round_trips_verbatim(queue: EventQueue) -> None:
    event_id = await queue.submit(message(), Priority.NORMAL)
    stream_id = (await queue.get(event_id)).stream_id
    assert stream_id is not None
    context = await queue.context_for(stream_id)

    turns = [
        Turn.user("hello", event_id=event_id),
        Turn(Role.ASSISTANT, (THINKING, TOOL_USE)),
        Turn(Role.USER, (TOOL_RESULT,)),
    ]
    await queue.append_turns(context.id, turns)

    with counting_selects() as selects:
        rows = await queue.transcript(context.id)
    assert len(selects) == 1, selects
    assert [row.to_turn() for row in rows] == turns
    assert rows[0].event_id == event_id and rows[1].event_id is None
    assert [row.id for row in rows] == sorted(row.id for row in rows)


async def test_a_stream_opens_one_context_and_keeps_it(queue: EventQueue) -> None:
    first = await queue.submit(message(sender="mira"), Priority.NORMAL)
    second = await queue.submit(message(sender="hal"), Priority.NORMAL)
    stream_id = (await queue.get(first)).stream_id
    assert stream_id is not None

    opened = await queue.context_for(stream_id)
    again = await queue.context_for(stream_id)

    assert opened.id == again.id
    assert opened.agent == again.agent
    assert opened.agent.role is AgentRole.SUBAGENT
    assert (await queue.get(second)).stream_id == stream_id
    (summary,) = await queue.list_streams()
    assert summary.stream.context_id == opened.id
    assert summary.turns == 0


async def test_the_stream_listing_counts_turns_without_multiplying_events(
    queue: EventQueue,
) -> None:
    ids = [await queue.submit(message(f"m{i}"), Priority.LOW) for i in range(3)]
    stream_id = (await queue.get(ids[0])).stream_id
    assert stream_id is not None
    context = await queue.context_for(stream_id)
    await queue.append_turns(context.id, [Turn.user("a"), Turn.user("b")])
    # A second stream with no context still lists.
    await queue.submit(message(conversation="#general"), Priority.LOW)

    with counting_selects() as selects:
        summaries = {s.stream.id: s for s in await queue.list_streams()}

    assert len(selects) == 1, selects
    assert summaries[stream_id].active == 3
    assert summaries[stream_id].turns == 2
    assert [s.turns for s in summaries.values() if s.stream.id != stream_id] == [0]


# -- dispatch -------------------------------------------------------------


async def dispatcher(queue: EventQueue, messages: FakeMessages) -> Dispatcher:
    return Dispatcher(
        queue,
        client=SimpleNamespace(messages=messages),  # type: ignore[arg-type]
        config=Config(),
        renderer=PromptRenderer(),
        agent=await queue.spawn(AgentRole.TRIAGE),
    )


async def test_work_in_one_stream_goes_to_its_context(queue: EventQueue) -> None:
    first = await queue.submit(message("did the render finish?"), Priority.HIGH)
    messages = FakeMessages("Told her it finished.")
    triage = await dispatcher(queue, messages)

    subagent = await triage.assign([first], "answer her", Action.HANDLE_ONE_EVENT)
    await triage.drain()

    # The subagent is the context's agent, and the same one next time round.
    stream_id = (await queue.get(first)).stream_id
    assert stream_id is not None
    context = await queue.context_for(stream_id)
    assert subagent == context.agent

    second = await queue.submit(message("and the alpha channel?"), Priority.HIGH)
    messages.text = "Yes, preserved, as I said it finished."
    with counting_selects() as selects:
        assert await triage.assign([second], "follow up", Action.HANDLE_ONE_EVENT) == subagent
    # Routed and recorded in one transaction: the rows (their streams ride along),
    # the routing pairs, the context — nothing fetched again between steps.
    assert len(selects) == 3, selects
    await triage.drain()

    # One transcript that grew: event, brief, reply, event, brief, reply.
    turns = [row.to_turn() for row in await queue.transcript(context.id)]
    assert [t.role for t in turns] == [Role.USER] * 2 + [Role.ASSISTANT] + [Role.USER] * 2 + [Role.ASSISTANT]
    assert [t.event_id for t in turns] == [first, None, None, second, None, None]
    assert turns[2].text == "Told her it finished."

    # The second call sent the first call's turns ahead of its own...
    sent = messages.calls[1]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert "did the render finish?" in str(sent[0]["content"])
    # ...under a system prompt, asking for the prefix to be cached.
    assert messages.calls[1]["system"] == PromptRenderer().subagent_system()
    assert messages.calls[1]["cache_control"] == {"type": "ephemeral"}

    # And the actions show one subagent handling the stream twice.
    history = await queue.history_for([first, second])
    assigned = {e.assigned_agent_id for entries in history.values() for e in entries if e.assigned_agent_id}
    assert assigned == {subagent.id}
    assert all(row.status is Status.COMPLETED for row in await queue.get_many([first, second]))
    (first_outcome, second_outcome) = triage.dispositions
    assert first_outcome.context_id == second_outcome.context_id == context.id
    assert second_outcome.usage is not None


async def test_work_across_streams_goes_to_a_one_shot_subagent(queue: EventQueue) -> None:
    here = await queue.submit(message(conversation="#workshop"), Priority.HIGH)
    there = await queue.submit(message(conversation="#general"), Priority.HIGH)
    messages = FakeMessages("Handled both.")
    triage = await dispatcher(queue, messages)

    subagent = await triage.assign([here, there], "together", Action.HANDLE_EVENT_SEQUENCE)
    await triage.drain()

    # No stream was routed, nothing was written, and nothing asked to be cached.
    assert all(s.stream.context_id is None for s in await queue.list_streams())
    (call,) = messages.calls
    assert isinstance(call["cache_control"], Omit)
    assert [m["role"] for m in call["messages"]] == ["user"]
    assert "in the order given" in str(call["messages"][0]["content"])
    (outcome,) = triage.dispositions
    assert outcome.context_id is None
    assert outcome.report == "Handled both."
    # A later assignment in one of those streams opens a context with a new agent.
    later = await queue.submit(message(conversation="#workshop"), Priority.HIGH)
    assert await triage.assign([later], "again", Action.HANDLE_ONE_EVENT) != subagent
    await triage.drain()


async def test_streams_that_share_a_context_route_to_it(queue: EventQueue) -> None:
    first = await queue.submit(message(conversation="#workshop"), Priority.HIGH)
    messages = FakeMessages("On it.")
    triage = await dispatcher(queue, messages)
    subagent = await triage.assign([first], "answer her", Action.HANDLE_ONE_EVENT)
    await triage.drain()
    workshop = (await queue.get(first)).stream_id
    assert workshop is not None
    context = await queue.context_for(workshop)

    # #general moves into the workshop's conversation; work spanning both now lands there.
    there = await queue.submit(message(conversation="#general"), Priority.HIGH)
    general = (await queue.get(there)).stream_id
    assert general is not None
    await queue.join_context(general, context.id)
    here = await queue.submit(message(conversation="#workshop"), Priority.HIGH)

    messages.text = "Both handled."
    assert await triage.assign([there, here], "together", Action.HANDLE_EVENT_SEQUENCE) == subagent
    await triage.drain()

    # The one transcript grew by the two events, the brief, and the reply.
    turns = [row.to_turn() for row in await queue.transcript(context.id)]
    assert [t.event_id for t in turns] == [first, None, None, there, here, None, None]
    assert turns[-1].text == "Both handled."
    (_, outcome) = triage.dispositions
    assert outcome.context_id == context.id
    assert len(outcome.streams) == 2  # crossed streams, and still went to the context


async def test_a_failed_call_leaves_the_events_in_the_transcript(queue: EventQueue) -> None:
    event_id = await queue.submit(message(), Priority.HIGH)
    triage = Dispatcher(
        queue,
        client=None,  # type: ignore[arg-type]
        config=Config(),
        renderer=PromptRenderer(),
        agent=await queue.spawn(AgentRole.TRIAGE),
    )

    await triage.assign([event_id], "answer her", Action.HANDLE_ONE_EVENT)
    await triage.drain()

    stream_id = (await queue.get(event_id)).stream_id
    assert stream_id is not None
    context = await queue.context_for(stream_id)
    turns = await queue.transcript(context.id)
    assert [t.role for t in turns] == [Role.USER, Role.USER]  # shown, never answered
    assert (await queue.get(event_id)).status is Status.PENDING
    (outcome,) = triage.dispositions
    assert outcome.error is not None
