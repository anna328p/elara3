# Attention: a live context, and the triage tool that chooses it

A sketch. It describes the finished design, not the decisions behind it, using
the prototype's terms (`../README.md`): events, streams, contexts, the two
passes, and the wake-ups that `watch` runs on.

## Context

Realtime priority is currently only a sort key. A message from someone in an
active conversation waits for a triage pass and then for a subagent call, so
the reply arrives a minute or more after a message that expected one in
seconds. The architecture note says realtime events are handled immediately and
suggests a queue per channel as one option.

In this design, at most one context is *live*. Events in the streams that route
to a live context are handled by a dedicated loop, the thread, without a triage
call. Triage decides which context is live, using one additional tool in the
pass it is already running, because triage is the model that sees every urgent
arrival. While a context is live, triage does not assign into it. Only the
first message of a conversation waits for a triage pass.

Focus names a context. Two streams that share a context are attended together,
the thread continues with the context's full transcript, and when focus moves
elsewhere the context remains and triage assigns into it as before. Making a
context live changes only which loop writes to it.

## The conversational agent

The prototype mints a subagent for every context. Its system prompt describes a
task handler: triage hands you events, report back. That framing is wrong for a
conversation, in batch as well as live, and it would be the identity that live
replies are attributed to. So the role of a context's agent depends on what the
context is for.

```python
class AgentRole(StrEnum):
    TRIAGE = "triage"
    SWEEP = "sweep"
    #: The agent of a channel or direct stream's context. It represents the
    #: character in that one exchange, in batch and live alike.
    CONVERSATIONAL = "conversational"
    #: A task handler: a job stream's context, or one-shot work.
    SUBAGENT = "subagent"
```

`resolve_context` mints a conversational agent for a channel or direct stream
and a subagent for a job stream. One-shot assignments still get a subagent. The
role selects the system prompt, `conversation_system.md.j2` or the existing
`subagent_system.md.j2`. A context's role never changes, so its cached prefix
does not change either. The conversational prompt states that the agent
represents the character in this one exchange, that each new message comes
with a brief from triage or, when live, with a note that the conversation is
happening now, and that its report is the text it would send, in the
character's voice. In the real framework this is the agent that calls the
personality model's tools.

The thread serves conversational agents only. A job stream therefore cannot be
made live, and the tool needs no separate check for it.

## The live context

Focus is a pointer to a context, stored as an append-only history of shifts.
The current focus is the last row.

```python
class Shift(StrEnum):
    ACQUIRE = "acquire"    # nothing was live; this context now is
    PREEMPT = "preempt"    # another context was live; this one replaces it
    RELEASE = "release"    # nothing is live now

class AttentionRow(Base):
    __tablename__ = "attention"
    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(default=utcnow)
    shift: Mapped[Shift]
    #: The context made live, or None on a release.
    context_id: Mapped[int | None] = mapped_column(ForeignKey("contexts.id"))
    #: Who decided: the triage agent on a divert; the conversational agent on
    #: a release, whether it yielded or the thread timed out on its behalf.
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id"))
    reason: Mapped[str]
```

`EventQueue.focus()` returns the current row or `None`. `shift()` appends a row
and, after the commit, wakes subscribers with a `Shifted`, a third kind of wake
alongside `Arrived` and `Heartbeat`:

```python
@dataclass(frozen=True, slots=True)
class Shifted:
    """The focus moved. The thread acts on it; `watch` ignores it."""

    shift: Shift
    context_id: int | None
    at: datetime

type Wake = Arrived | Heartbeat | Shifted
```

`due` in the scheduler already counts only arrivals and heartbeats, so `watch`
needs no change. Nothing else is cached: how long the context has been live,
how many events the thread has taken and when the last one arrived are queries
over `attention`, `event_actions` and `events`, one per number, made when the
triage prompt is rendered.

A context has one writer at a time. While a context is live the thread is its
writer. `EventQueue.assign` refuses an assignment that would route into the
live context, with an error the triage model can read ("stream 4 is live; the
realtime thread has it"), and `triage_view` omits pending events whose stream
routes there. In the other direction, if a context becomes live while a batch
assignment into it is still running, the thread makes no call until that
assignment has reported.

The refusal requires the queue to know which events are being handled, which
the prototype does not record: an assigned event stays `PENDING` until the
assignee reports. So `assign` sets a fourth status, `ASSIGNED`, and
`record_failure` sets the events back to `PENDING` instead of leaving them
unchanged. `Action` gains `ATTEND` for the thread's claims and `RETURNED` for
claims given back. Two concurrent passes were already unsafe without this
status; the thread makes the problem certain to occur.

## Diverting attention

Triage gets one more tool.

`divert_attention(stream_id, reason)` makes the context that the stream routes
to live, opening a context if the stream has none. It records an `ACQUIRE` if
nothing was live and a `PREEMPT` otherwise, in which case the thread finishes
with the previous context as described in the next section. It is refused for
a stream whose context belongs to a subagent, which means a job stream. Every
pending event in the diverted stream now belongs to the thread: the dispatcher
claims them, so a later `handle_one_event` on one of them in the same pass is
refused the same way a second disposition is, and they are exempt from the
rule that every event id must appear in exactly one call.

There is no tool to hold and none to release. A pass holds by seeing the live
context and assigning elsewhere anyway; the assignment's reason records the
decision. Release is decided from inside the conversation: the conversational
agent yields when it judges the exchange over, and the thread times out on its
behalf when nothing has arrived for `focus_idle_seconds` and no reply is owed.
Triage can move the thread to another context but cannot stop it.

The triage prompt shows the live context in a `<live>` block above its events:
the titles of its streams, how long it has been live, how many events the
thread has attended, seconds since the last one, and whether a reply is owed,
which means the transcript ends on a user turn. The block serves two purposes.
A pass deciding whether to divert needs to know what the thread is currently
doing. And a brief to a conversational agent replying elsewhere can state that
the character is talking to Alice right now, which changes what a good reply
looks like.

The template's guidance is short. Divert when a person is replying within
seconds, which is usual for a direct stream at realtime priority and unusual
for a room mention. Do not divert for a single message when a reply is owed in
the live context; a switch means an uncached first call in the new context and
a slower answer for the person in the previous one, whose messages are then
assigned in batch like everything else. When the candidate is the same person
writing from a second place, assign it into the live context's stream instead.
That case needs the message between contexts that the prototype does not have.

## The realtime thread

The thread is a loop over the live context, in `attention.py`, subscribed to
the queue alongside `watch`. Each iteration:

1. Wait on the subscription. A `Shifted` wake means the focus has moved or
   gone: finish with the old context (below) and start on the new one. An
   `Arrived` wake with nothing live requires no action, and the loop waits
   again without a read.
2. Claim every pending event whose stream routes to the live context: set it
   to `ASSIGNED` with an `ATTEND` action attributed to the conversational
   agent, which is taking the events for itself. All pending events are taken
   at once, so three messages that arrived during the previous call become one
   user message; `contexts.to_api` already merges consecutive user turns.
3. Append the events as turns plus one brief, `live.md.j2`: this conversation
   is happening now; answer what needs answering, and say nothing if nothing
   does. The brief is a turn and not a change of system prompt, so the cached
   prefix survives the change of mode.
4. Call the conversational agent with one tool, `yield_focus(reason)`. Append
   the reply, complete the events with the reply as their report, and if the
   tool was called, append its result and record a `RELEASE` shift attributed
   to that agent.
5. Repeat. The wait has a timeout of `focus_idle_seconds`, measured from the
   live context's last event. A `Heartbeat` with nothing pending in the
   context and no reply owed is also a `RELEASE`, attributed to the same agent,
   with a reason stating how long the context was idle. Nothing polls: the
   queue wakes the thread after each commit that could concern it, its reads
   are idempotent, and a wake caused by an arrival in some other stream costs
   one query.

Finishing with a context sets every event the thread claimed but did not
complete back to `PENDING` with a `RETURNED` action, so a crash during a call
leaves nothing stranded past the next start. On startup, a focus left by a
previous run is released the same way. On shutdown, the loop releases in a
`finally`. The real framework needs a lease here, renewed each iteration, so
that a dead thread does not keep holding focus; the prototype has one process
and a `finally`.

The thread's calls use `realtime_model` and `realtime_effort`, both set low.
The goal is a reply within seconds, and the same agent will continue the same
context at higher effort when triage next assigns into it.

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

`attend` runs `watch` and the thread in one process. It prints each shift as
it happens and each of the thread's calls with its cache reads, so a
preemption shows up as a call with no cache reads. The queue wakes only its
own process, so an event written from a second terminal would wait for the
heartbeat. `--play` names a scenario in `fixtures.py`, a timed list of events
submitted from inside the process while the loops run. That is how the flow
below is run by hand.

## Execution flow

A scenario, followed through the process. `attend` opens the queue, releases
any focus left by a previous run, and starts two subscribers: `watch`, which
runs triage on an urgent arrival and on the heartbeat, and the thread. The
sweep remains a separate command.

A nightly backup reports. Its `JobEvent` is committed at `LOW`, and the queue
wakes both subscribers with `Arrived`. The thread has nothing live and waits
again without a read. `watch` sees a priority below `urgent_priority` and
waits for its heartbeat. On the heartbeat, triage assigns the report to the
backup job's context, whose agent is a subagent, and that subagent reports.

Alice sends a direct message at `REALTIME`. `watch` starts a triage pass,
because realtime is urgent. The pass sees one event in a direct stream and
nothing live, and calls `divert_attention`. `resolve_context` opens a context
with a conversational agent, the `ACQUIRE` row commits, the dispatcher claims
Alice's message for the thread, and the queue wakes with `Shifted`. The thread
reads the focus, claims the message with an `ATTEND`, appends the event turn
and the live brief, calls the conversational agent at low effort with
`yield_focus` available, appends the reply, and completes the event with the
reply as its report. No other loop makes routing decisions, so there is no
race between them.

Alice sends two more messages while that call is running. Each commit wakes
the thread. Its next wait returns both wakes in one list, it claims both
events, and `to_api` merges them into one user message. That is one call, and
it reads the prefix the previous call cached. `watch` was woken as well, but
`triage_view` omits events in the live context, so the pass finds nothing
pending and is skipped.

Bob mentions the character in a room, at `HIGH`. `watch` runs triage. The pass
sees Bob's event and, in the `<live>` block, that Alice's context has taken
four events, the last one twenty seconds ago, with no reply owed. It does not
divert. It assigns the event to the room's context, which is opened now with
a conversational agent of its own, with a brief stating that the character is
currently talking to Alice. That is the hold, and the assignment's reason
records it. Bob gets a reply in about a minute instead of seconds, in the
character's voice, and the room's agent keeps the exchange in its context.
The thread was woken by the arrival, found nothing pending in its context,
and waited.

Alice writes goodnight. The thread replies and calls `yield_focus`. The tool
result is appended, a `RELEASE` attributed to the conversational agent
commits, `Shifted` is sent, and the thread finishes with the context, which
has nothing to return. If she had stopped writing without saying so, the
thread's wait would have timed out after `focus_idle_seconds` and released on
the agent's behalf. In either case the context is no longer live. A message
from Alice an hour later goes through the same triage pass and the same
divert, and the thread's first call reads the whole transcript uncached, which
the report shows.

Ctrl-C cancels the main task. The thread's `finally` releases and returns any
claim it did not complete. `watch` shuts down as its own documentation
describes.

## Verification

Tests against the in-memory database, without the API: a claimed event is
`ASSIGNED` and absent from `triage_view`; `assign` into the live context
raises; `divert_attention` on a job stream is refused because its agent is a
subagent; a divert claims the stream's pending events within the pass, so a
second disposition on one of them is refused; a `PREEMPT` sets the old
context's unfinished claims back to `PENDING` and leaves its completed ones
alone; a release with everything completed returns nothing; `shift` wakes
subscribers with `Shifted` after commit and not on rollback; the idle
heartbeat releases only with nothing pending and no reply owed; a stale focus
is released on startup. The thread's model call is faked the way
`test_dispatch` fakes subagents. The scenario above is a test that drives
`watch` and the thread with the fixture events and asserts the resulting shift
history.

## Left out

Budget: the architecture reserves realtime quota that the rest of the system
cannot use. The thread should meter against it, and triage should be told how
much is left when it decides whether to divert. Liveness: the lease described
above. A message between contexts, which the same-person-elsewhere case needs.
A release tool for triage, for a live context that is holding the thread on a
conversation not worth answering; the idle timeout covers this for now. And
whether the thread should run tools of its own; here a conversational agent is
the same single call as a subagent, and describes what it would send.
