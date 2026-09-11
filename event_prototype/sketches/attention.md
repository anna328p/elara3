# Attention: a live context, and the triage tool that steers it

A sketch. It describes the finished shape rather than the road to it, in the
vocabulary of the prototype (`../README.md`): events, streams, contexts, the
two passes, and the wake-ups `watch` runs on.

## Context

Realtime priority exists today only as a sort key. A message from someone in an
active conversation still waits for a triage pass, a model call whose job is
sorting, and then for a subagent call, and the reply lands a minute later in a
conversation that moves in seconds. The architecture note says such events are
addressed *now*, with a queue per channel as a maybe.

The sketch gives the character one place where it is present in real time. At
most one context is *live*: events in the streams that route to it go straight
to that context, without a triage call, and are answered by a thread that does
nothing else. Which context is live is triage's decision, made with one more
tool in the pass it was already running, since triage is the model that sees
every arrival and ranks it. Once a context is live, triage keeps its hands off
it, and the first message of a conversation is the only one that waits for a
pass.

A whole context is live, not a stream. Two streams that already share a context
are attended together, the thread inherits everything that context's agent has
said and been shown, and when attention moves on the context stays where it
was, cold, with triage assigning into it as before. Making a context live
changes who writes to it, not what it is.

## The conversational agent

The prototype mints a subagent for every context, and its framing is a task
handler's: triage hands you events, report back. That is the wrong voice for a
conversation even when nobody is waiting on it, and it would be the identity
the live replies are attributed to. So the role of a context's agent follows
what the context is for.

```python
class AgentRole(StrEnum):
    TRIAGE = "triage"
    SWEEP = "sweep"
    #: A presence in one ongoing exchange: the agent of a channel or direct
    #: stream's context, in batch and live alike.
    CONVERSATIONAL = "conversational"
    #: A task handler: a job stream's context, or one-shot work.
    SUBAGENT = "subagent"
```

`resolve_context` mints a conversational agent for a channel or direct stream
and a subagent for a job stream; one-shot assignments still get a subagent. The
role picks the system prompt, `conversation_system.md.j2` or the existing
`subagent_system.md.j2`, and since a context's role never changes, neither does
its cached prefix. The conversational prompt says what the agent is: the
character's presence in this one exchange, given each new message with a brief
from triage or, when live, with the note that the conversation is happening
now. Its report is the text it would send, in the character's voice. In the
real framework this is the agent that calls the personality model's tools,
which is where the logistics and personality boundary sits.

The thread serves conversational agents and nothing else. That is why a job
stream cannot be made live, rather than a check written into the tool.

## The live context

Focus is a pointer at a context, recorded as an append-only history of shifts;
the current focus is the last row.

```python
class Shift(StrEnum):
    ACQUIRE = "acquire"    # nothing was live; this context now is
    PREEMPT = "preempt"    # another context was live; this one takes over
    RELEASE = "release"    # nothing is live now

class AttentionRow(Base):
    __tablename__ = "attention"
    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(default=utcnow)
    shift: Mapped[Shift]
    #: The context made live, or None on a release.
    context_id: Mapped[int | None] = mapped_column(ForeignKey("contexts.id"))
    #: Who decided: triage on a divert, the conversational agent on a
    #: release, whether it yielded or the thread timed out on its behalf.
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id"))
    reason: Mapped[str]
```

`EventQueue.focus()` returns the current row or `None`; `shift()` appends one
and, after its commit, wakes subscribers with a `Shifted`, the third kind of
wake next to `Arrived` and `Heartbeat`:

```python
@dataclass(frozen=True, slots=True)
class Shifted:
    """The focus moved. The thread wants to know; `watch` does not care."""

    shift: Shift
    context_id: int | None
    at: datetime

type Wake = Arrived | Heartbeat | Shifted
```

`due` in the scheduler already only counts arrivals and heartbeats, so `watch`
is unchanged by it. Nothing else is cached: how long the context has been live,
how many events the thread has taken and when the last one came are reads over
`attention`, `event_actions` and `events`, and the triage prompt that needs
them makes one query per number.

A context has one writer at a time. While it is live the thread is that writer:
`EventQueue.assign` refuses an assignment that would route into the live
context, with an error the triage model can read ("stream 4 is live; the
realtime thread has it"), and `triage_view` drops pending events whose stream
routes there, so triage neither sees nor touches them. The other direction is
handled by waiting: when a context becomes live with a batch assignment still
in flight, the thread makes no call until that assignment has reported.

That refusal needs the queue to know what is in flight, which the prototype
does not record: an assigned event stays `PENDING` until the assignee reports.
So `assign` sets a fourth status, `ASSIGNED`, and `record_failure` returns the
events to `PENDING` rather than leaving them as they were. `Action` gains
`ATTEND` for the thread's claims and `RETURNED` for claims given back. Two
concurrent passes were already unsafe without this; the thread just makes it
unavoidable.

## Diverting attention

Triage's vocabulary grows by one tool:

`divert_attention(stream_id, reason)` makes the context that stream routes to
live, opening one if the stream has none. It is an `ACQUIRE` when nothing was
live and a `PREEMPT` otherwise, in which case the previous context is finished
with as the thread section describes. It is refused for a stream whose context
belongs to a subagent, that is, a job stream. Every pending event in the
diverted stream is the thread's from that moment: the dispatcher claims them
for it, so a later `handle_one_event` on one of them in the same pass is
refused the way a second disposition already is, and they no longer count
against "every event id above must appear in exactly one call".

There is no tool to hold and none to release. Holding is what a pass does when
it sees the live context and assigns elsewhere anyway; the assignment's own
reason records it. Releasing belongs to the conversation: the conversational
agent yields when it judges the exchange over, and the thread times out on its
behalf when nothing has arrived for `focus_idle_seconds` and no reply is owed.
Triage takes the thread away only by giving it to someone else.

The triage prompt shows the live context in a `<live>` block above its events:
its streams' titles, how long it has been live, events attended, seconds since
the last one, and whether a reply is owed, meaning the transcript ends on a
user turn. The block is there for two reasons. A pass deciding whether to
divert needs to know what it would be taking the thread from. And a brief to a
conversational agent replying elsewhere can say "she is talking to Alice right
now", which changes what a good reply looks like.

The guidance in the template is short. Divert when a person is in an exchange
that moves in seconds, which a direct stream at realtime priority nearly always
is and a room mention rarely is. Do not divert for a single message when a
reply is owed where the thread already is; a switch costs the new context a
cold read and the person left behind a slower answer, and their conversation
is not dropped, only assigned in batch like everything else. When the
candidate is the same person writing from a second place, assign it into the
live context's stream instead, and note that this is the message between
contexts the prototype does not yet have.

## The realtime thread

The thread is a loop over the live context, in `attention.py`, a subscriber on
the queue next to `watch`. Each turn of it:

1. Wait on its subscription. A `Shifted` wake means the focus has moved or
   gone: finish with the old context (below) and start on the new one. An
   `Arrived` wake with nothing live is nothing to do, and the loop waits again
   without a read.
2. Claim every pending event whose stream routes to the live context:
   `ASSIGNED`, with an `ATTEND` action attributed to the conversational agent,
   which is taking them for itself. Everything pending is taken at once, so
   three messages that arrived during the last call become one user message,
   which `contexts.to_api` already does by merging consecutive user turns.
3. Append the events as turns plus one brief, `live.md.j2`: this is
   happening now; answer what wants answering, and say nothing if nothing does.
   The brief is a turn rather than a change of system prompt, so the cached
   prefix survives the change of mode.
4. Call the conversational agent with one tool, `yield_focus(reason)`.
   Append the reply, complete the events with the reply as their report, and if
   the tool was called, append its result and record a `RELEASE` shift
   attributed to that agent.
5. Go round again. The wait carries a timeout of `focus_idle_seconds` measured
   from the live context's last event, and a `Heartbeat` with nothing pending
   there and no reply owed is a `RELEASE` too, attributed to the same agent
   with the reason saying how long it sat idle. Nothing here polls: the queue
   wakes the thread after each commit that could concern it, its reads are
   idempotent, and a wake that turns out to be someone else's arrival costs
   one query and nothing else.

Finishing with a context returns anything the thread claimed but never
completed to `PENDING` with a `RETURNED` action, so a crash mid-call strands
nothing past the next start. On startup, a focus left over from a previous run
is released the same way; on shutdown, the loop releases in a `finally`. The
real framework wants a lease here, renewed each turn, so that a dead thread
cannot hold focus from beyond the grave; the prototype has one process and a
`finally`.

The thread's calls use `realtime_model` and `realtime_effort`, both set low:
the point is to answer in seconds, and the context is the same one the same
agent will continue at higher effort when triage next assigns into it.

## Config

```toml
realtime_model = "claude-sonnet-5"
realtime_effort = "low"
focus_idle_seconds = 180
```

## Commands

```sh
uv run python -m event_prototype attend                 # watch plus the thread, until ^C
uv run python -m event_prototype attend --play evening  # the same, fed a scripted evening of events
uv run python -m event_prototype focus                  # the current focus and every shift so far
```

`attend` runs `watch` and the thread in one process and prints each shift as
it happens and each of the thread's turns with its cache reads, so a
preemption is visible as the cold read it is. The queue wakes only its own
process, so an event written from a second terminal would wait for the
heartbeat; `--play` names a scenario in `fixtures.py`, a timed list of events
submitted from inside the process while the loops run, which is how the flow
below is driven by hand.

## Execution flow

One evening, as the process sees it. `attend` opens the queue, releases any
focus a previous run left behind, and starts two subscribers: `watch`, which
runs triage on an urgent arrival and on the heartbeat, and the thread. The
sweep stays a command.

A nightly backup reports. Its `JobEvent` is committed at `LOW`, and the queue
wakes both with `Arrived`. The thread has nothing live and goes back to
waiting without a read. `watch` sees it is below `urgent_priority` and waits
for its heartbeat, on which triage runs, assigns the report to the backup job's
context, whose agent is a subagent, and that subagent reports.

Alice sends a direct message, submitted at `REALTIME`. `watch` starts a triage
pass, since realtime is urgent. The pass sees one event, a direct stream,
nothing live, and calls `divert_attention`; `resolve_context` opens a context
with a conversational agent, the `ACQUIRE` row commits, the dispatcher claims
Alice's message for the thread, and the queue wakes with `Shifted`. The thread
reads the focus, claims the message with an `ATTEND`, appends the event turn
and the live brief, calls the conversational agent at low effort with
`yield_focus` on offer, appends the reply, and completes the event with the
reply as its report. Nobody else was deciding anything about that message, so
there is no race to explain: triage is the one place routing is decided, and
the pass that saw the message first is the pass that gave it away.

Alice sends two more while that call is in flight. Each commit wakes the
thread; its next wait hands back both wakes as one list; it claims both, and
`to_api` folds them into one user message. One call, reading the prefix the
last one wrote. `watch` woke too, but `triage_view` drops events in the live
context, so the pass it would run sees nothing pending and is skipped.

Bob mentions the character in a room, at `HIGH`. `watch` runs triage, which
sees Bob's event and, in the `<live>` block, that Alice's context has taken
four events, the last twenty seconds ago, with no reply owed. It does not
divert. It hands the event to the room's context, opened now with a
conversational agent of its own, briefed that the character is talking to
Alice at the moment. That is the hold, recorded as the assignment's reason.
Bob's reply comes in a minute rather than seconds, in the same voice, from an
agent that will remember it next time. The thread was woken by the arrival,
found nothing pending in its context, and waited.

Alice writes goodnight. The thread replies and calls `yield_focus`; the tool
result is appended, a `RELEASE` attributed to the conversational agent
commits, `Shifted` goes out, and the thread finishes with the context, finding
nothing to return. Had she just stopped writing, the thread's wait would have
run out at `focus_idle_seconds` and released on the agent's behalf instead.
Either way the context is now cold, and a message from Alice an hour later
goes through the same triage pass and the same divert, with the thread's first
call a cold read of the whole transcript, which the report prints as such.

Ctrl-C cancels the main task. The thread's `finally` releases and returns any
claim it never completed; `watch` unwinds as its own documentation says.

## Verification

Tests against the in-memory database, no API: a claimed event is `ASSIGNED` and
absent from `triage_view`; `assign` into the live context raises;
`divert_attention` on a job stream is refused because its agent is a subagent;
a divert claims the stream's pending events within the pass, so a second
disposition on one of them is refused; a `PREEMPT` returns the old context's
unfinished claims to `PENDING` and leaves its completed ones alone; a release
with everything completed returns nothing; `shift` wakes subscribers with
`Shifted` after commit and not on rollback; the idle heartbeat releases only
with nothing pending and no reply owed; a stale focus is released on startup.
The thread's model call is faked the way `test_dispatch` fakes subagents, and
the evening above is a test that drives `watch` and the thread with the
fixture scenario and asserts the shift history it leaves.

## Left out

Budget: the architecture reserves realtime quota the rest of the system cannot
draw on, and the thread should meter against it, with triage told how much is
left when it decides whether to divert. Liveness: the lease above. A message
between contexts, which is what the same-person-elsewhere case needs. A
release tool for triage, if a live context ever turns out to be wasting the
thread with nobody worth answering; the idle timeout covers that for now. And
whether the thread should ever run tools of its own; here a conversational
agent is the same single call as a subagent, and describes what it would send.
