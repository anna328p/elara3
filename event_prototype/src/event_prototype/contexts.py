"""Contexts: the conversations agents have, as a series of turns.

A context is what a stream routes to. It is stored as turns, one per Messages
API message, so that replaying it is a matter of listing the rows in order and
sending them. Content is kept as the API's own content blocks, verbatim: a tool
call is a `tool_use` block in an assistant turn answered by a `tool_result`
block in the next user turn, thinking blocks carry signatures the API checks on
replay, and neither survives being reshaped into anything of our own.

`Turn` is the in-memory value; `TurnRow` in `store.py` is its persistence.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

from anthropic.types import Message, MessageParam
from anthropic.types.beta import BetaMessage, BetaMessageParam

#: One Anthropic content block, as JSON.
type Block = dict[str, Any]


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class PendingToolCalls(ValueError):
    """The transcript ends on tool calls nobody has answered yet.

    That is a real state — the agent is waiting on results — but not one that
    can be sent: the API wants every `tool_use` matched before it will continue.
    """

    def __init__(self, tool_use_ids: Sequence[str]) -> None:
        super().__init__(f"unanswered tool calls: {', '.join(tool_use_ids)}")
        self.tool_use_ids = tuple(tool_use_ids)


@dataclass(frozen=True, slots=True)
class Turn:
    """One message in a context. Exactly one Messages API message."""

    role: Role
    content: tuple[Block, ...]
    #: The event that occasioned this turn, when one did. A brief from triage
    #: or the agent's own reply has none.
    event_id: int | None = None

    @classmethod
    def user(cls, text: str, *, event_id: int | None = None) -> Turn:
        return cls(Role.USER, ({"type": "text", "text": text},), event_id)

    @classmethod
    def of(cls, response: Message | BetaMessage) -> Turn:
        """The assistant turn a completion is, blocks dumped verbatim.

        `exclude_none` drops the SDK's optional fields (`citations`, say) that
        were never set, and nothing else: the content the API sent back is the
        content it will accept on replay.
        """
        blocks = tuple(
            block.model_dump(mode="json", exclude_none=True) for block in response.content
        )
        return cls(Role.ASSISTANT, blocks)

    @classmethod
    def results(cls, message: BetaMessageParam) -> Turn:
        """The user turn the tool runner answers a call with, blocks kept as sent.

        The runner builds it as `tool_result` blocks in a user message. A bare
        string is the API's shorthand for one text block and is stored as that,
        so there is one shape to read back.
        """
        content = message["content"]
        if isinstance(content, str):
            return cls(Role.USER, ({"type": "text", "text": content},))
        return cls(Role.USER, tuple(dict(block) for block in content))

    @property
    def text(self) -> str:
        """The visible text, ignoring thinking and tool blocks."""
        return "\n".join(
            block["text"] for block in self.content if block["type"] == "text"
        ).strip()

    @property
    def tool_use_ids(self) -> tuple[str, ...]:
        return tuple(b["id"] for b in self.content if b["type"] == "tool_use")

    @property
    def tool_result_ids(self) -> tuple[str, ...]:
        return tuple(b["tool_use_id"] for b in self.content if b["type"] == "tool_result")


def to_api(turns: Iterable[Turn]) -> list[MessageParam]:
    """The message list this context sends.

    Turns are stored one per message, but the API wants roles to alternate, so
    consecutive same-role turns — three pings arriving before a reply — are
    merged into one message here. Within a merged user message the tool results
    lead, since the API wants them answered before anything else is said.

    Nothing about caching is decided here. The request marks its own breakpoint
    (see `subagent.py`), so the sent content is byte-for-byte the stored content.
    """
    messages: list[tuple[Role, list[Block]]] = []
    for turn in turns:
        if messages and messages[-1][0] is turn.role:
            messages[-1][1].extend(turn.content)
        else:
            messages.append((turn.role, list(turn.content)))
    for role, blocks in messages:
        if role is Role.USER:
            blocks.sort(key=lambda block: block["type"] != "tool_result")
    _require_answered(messages)
    # The blocks came from the API or are going to it in its own shape; the
    # SDK's TypedDicts cannot see that through a JSON column, hence the cast.
    return [
        cast(MessageParam, {"role": role.value, "content": blocks})
        for role, blocks in messages
    ]


def _require_answered(messages: Sequence[tuple[Role, list[Block]]]) -> None:
    """Every tool call must have its result in the message that follows."""
    for i, (role, blocks) in enumerate(messages):
        if role is not Role.ASSISTANT:
            continue
        called = [b["id"] for b in blocks if b["type"] == "tool_use"]
        if not called:
            continue
        answered: set[str] = set()
        if i + 1 < len(messages):
            following = messages[i + 1][1]
            answered = {b["tool_use_id"] for b in following if b["type"] == "tool_result"}
        if unanswered := [c for c in called if c not in answered]:
            raise PendingToolCalls(unanswered)
