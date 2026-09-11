# Attention: a live context and the control plane that steers it

A sketch. It describes the finished shape rather than the road to it, in the
vocabulary of the prototype (`../README.md`): events, streams, contexts, the
two passes.

## Context

Realtime priority exists today only as a sort key. A message from someone in an
active conversation still waits for a triage pass, a model call whose job is
sorting, and then for a subagent call, and the reply lands a minute later in a
conversation that moves in seconds. The architecture note says such events are
addressed *now*, with a queue per channel as a maybe.

The sketch gives the character one place where it is present in real time. At
most one context is *live*: events in the streams that route to it go straight
to that context, without a triage call, and are answered by a thread that does
nothing else. Which context is live is decided by an attention agent, a small
model that sees streams rather than events and runs only when something might
change its mind. Triage keeps sorting everything else and learns to keep its
hands off the live context.

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
    ATTENTION = "attention"
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
stream cannot be focused, rather than a check written into the tool.

## The live context

Focus is a pointer at a context, recorded as an append-only history of shifts;
the current focus is the last row.

```python
class Shift(StrEnum):
    ACQUIRE = "acquire"    # nothing was live; this context now is
    PREEMPT = "preempt"    # another context was live; this one takes over
    HOLD = "hold"          # looked, and left the live context as it was
    RELEASE = "release"    # nothing is live now

class AttentionRow(Base):
    __tablename__ = "attention"
    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(default=utcnow)
    shift: Mapped[Shift]
    #: The context made live, or None on a release.
    context_id: Mapped[int | None] = mapped_column(ForeignKey("contexts.id"))
    #: Who decided: the attention agent, or the thread on a yield.
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id"))
    reason: Mapped[str]
```

`EventQueue.focus()` returns the current row or `None`; `shift()` appends one.
Nothing else is cached. How long the context has been live, how many events the
thread has taken and when the last one came are all reads over `attention`,
`event_actions` and `events`, and the prompt that needs them (below) makes one
query per number.

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

## The realtime thread

The thread is a loop over the live context, in `attention.py`. Each turn of it:

1. Wait on its subscription. A `Shifted` wake means the focus has moved or
   gone: finish with the old context (below) and start on the new one. An
   `Arrived` wake with nothing live is nothing to do, and the loop waits again
   without a read.
2. Claim every pending event whose stream routes to the live context:
   `ASSIGNED`, with an `ATTEND` action attributed to the attention agent and
   the context's conversational agent as assignee. Everything pending is taken
   at once, so three messages that arrived during the last call become one user
   message, which `contexts.to_api` already does by merging consecutive user
   turns.
3. Append the events as turns plus one brief, `live.md.j2`: this is
   happening now; answer what wants answering, and say nothing if nothing does.
   The brief is a turn rather than a change of system prompt, so the cached
   prefix survives the change of mode.
4. Call the conversational agent with one tool, `yield_focus(reason)`.
   Append the reply, complete the events with the reply as their report, and if
   the tool was called, append its result and record a `RELEASE` shift
   attributed to that agent.
5. Go round again. Nothing here polls: the thread is woken by the queue after
   each commit that could concern it, and its reads are idempotent, so a wake
   that turns out to be someone else's arrival costs one query and nothing
   else.

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

## The control plane

The attention agent decides where the thread is. It is a separate agent role,
`AgentRole.ATTENTION`, one row minted when `attend` starts, and every shift
except a yield is attributed to it. It is small (Haiku-class by default,
`attention_model`) because it is shown very little.

It is a subscriber on the queue like `watch`, and decides from the wake's own
metadata what to do. An `Arrived` at `HIGH` or above is worth a look: with
nothing live it is an offer, with something live it is contention, and either
way the agent reads its view and makes one call. An `Arrived` below that is
ignored. A `Heartbeat` is the idle case: the control plane waits with a
timeout of `focus_idle_seconds`, measured from the live context's last event,
and when the wait runs out with nothing pending there and no reply owed it
releases by reflex, with no call, since the thread's `yield_focus` is where
judgment about a conversation's end lives. A `Shifted` from the thread's yield
is a reason to look at whatever else is waiting.

`Shifted` is the third kind of wake, next to `Arrived` and `Heartbeat`:

```python
@dataclass(frozen=True, slots=True)
class Shifted:
    """The focus moved. The thread and the control plane both want to know."""

    shift: Shift
    context_id: int | None
    at: datetime

type Wake = Arrived | Heartbeat | Shifted
```

`EventQueue.shift()` wakes with it after its commit, the same way `submit`
wakes with `Arrived`. `watch` ignores it, since `due` only cares about arrivals
and heartbeats. Because a wake arrives once per commit, a `hold` is honoured
until the next arrival by construction; nothing has to remember what was
already considered.

What it sees is `attention.md.j2`: the live context, if any, as its streams'
titles, how long it has been live, events attended, seconds since the last one,
and whether a reply is owed (the transcript ends on a user turn); then each
candidate stream with its kind, the age and priority of its pending events, and
their digest lines or descriptions. It sees no event bodies and no backlog. Its
tools:

`focus(stream_id, reason)` makes the context that stream routes to live,
opening one if the stream has none: an `ACQUIRE` when nothing was live, a
`PREEMPT` otherwise, and the previous context is finished with as above. It is
refused for a context whose agent is not conversational. `pull(stream_id,
reason)` routes another stream into the live context through `join_context`, so
the same person writing from a second place is answered in one conversation; it
is refused when that stream already has a different context, since that is the
message between contexts the prototype does not have, and re-pointing would
hide history from the thread. `release(reason)` says nothing deserves the
thread right now. `hold(reason)` keeps things as they are, and is recorded as a
`HOLD` row naming the live context, so the history shows the agent looked and
chose not to move.

The guidance in the template is short. A person typing in a direct stream
outweighs a busy room. Do not preempt for a single message when a reply is owed
where you are; a switch costs the new context a cold read and the person you
leave a slower answer through triage, and the thread will come to them after it
yields. Prefer releasing to holding an idle context, since an unattended stream
is not ignored, only batched. Pull rather than switch when the candidate is the
same person.

Triage is shown the live context too, read-only, in a `<live>` block above its
events, so that its briefs can say "she is talking to Alice right now" to a
conversational agent replying elsewhere. The control plane and triage never
share a decision: triage sorts events, attention places the thread, and the
only place they meet is the refusal in `assign`.

## Config

```toml
realtime_model = "claude-sonnet-5"
realtime_effort = "low"
attention_model = "claude-haiku-4-5"
focus_idle_seconds = 180
```

## Commands

```sh
uv run python -m event_prototype attend                 # watch, the control plane and the thread, until ^C
uv run python -m event_prototype attend --play evening  # the same, fed a scripted evening of events
uv run python -m event_prototype focus                  # the current focus and every shift so far
```

`attend` runs `watch` and the two new subscribers in one process and prints
each shift as it happens and each of the thread's turns with its cache reads,
so a preemption is visible as the cold read it is. The queue wakes only its own
process, so an event written from a second terminal would wait for the
heartbeat; `--play` names a scenario in `fixtures.py`, a timed list of events
submitted from inside the process while the loops run, which is how the flow
below is driven by hand.

## Execution flow

One evening, as the process sees it. `attend` opens the queue, mints the
attention agent, releases any focus a previous run left behind, and starts
three subscribers: `watch`, which runs triage on an urgent arrival and on the
heartbeat; the control plane; and the thread. The sweep stays a command.

A nightly backup reports. Its `JobEvent` is committed at `LOW`, and the queue
wakes all three with `Arrived`. The thread has nothing live and goes back to
waiting without a read. The control plane sees a priority below `HIGH` and does
the same. `watch` sees it is below `urgent_priority` and waits for its
heartbeat, on which triage runs, assigns the report to the backup job's
context, whose agent is a subagent, and that subagent reports.

Alice sends a direct message, submitted at `REALTIME`. Three wakes again.
`watch` starts a triage pass, since realtime is urgent. The control plane reads
its view, nothing live and one candidate, a direct stream with one realtime
event a second old, makes one small call, and gets `focus`; `resolve_context`
opens a context with a conversational agent, the `ACQUIRE` row commits, and
the queue wakes everyone with `Shifted`. The thread reads the focus, claims
Alice's message with an `ATTEND`, appends the event turn and the live brief,
calls the conversational agent at low effort with `yield_focus` on offer,
appends the reply, and completes the event with the reply as its report. The
triage pass, running the while, either read the queue before the acquire and
is refused when it calls `handle_one_event`, an error it reads and moves past,
or reached `assign` first, in which case the thread found the event `ASSIGNED`
and waited for the report before making its own first call. Either way the
context has one writer.

Alice sends two more while that call is in flight. Each commit wakes the
thread; its next wait hands back both wakes as one list; it claims both, and
`to_api` folds them into one user message. One call, reading the prefix the
last one wrote.

Bob mentions the character in a room, at `HIGH`. The control plane reads: the
live context has taken four events, the last twenty seconds ago, no reply
owed; the candidate is a channel stream with one high event and its
description. The call returns `hold`, with a reason, and a `HOLD` row is
written; nothing shifts. The thread was woken too, finds nothing pending in
its context, and waits. `watch` runs triage, which sees Bob's event, sees the
`<live>` block naming Alice, and hands the event to the room's context, opened
now with a conversational agent of its own, briefed that the character is
talking to Alice at the moment. Bob's reply comes in a minute rather than
seconds, in the same voice, from an agent that will remember it next time.

Alice writes goodnight. The thread replies and calls `yield_focus`; the tool
result is appended, a `RELEASE` attributed to the conversational agent
commits, `Shifted` goes out, and the thread finishes with the context, finding
nothing to return. Had she just stopped writing, the control plane's wait would
have run out at `focus_idle_seconds` and released by reflex instead. Either way
the context is now cold, and a message from Alice an hour later goes through
the same acquire, with the thread's first call a cold read of the whole
transcript, which the report prints as such.

Ctrl-C cancels the main task. The thread's `finally` releases and returns any
claim it never completed; `watch` unwinds as its own documentation says.

## Verification

Tests against the in-memory database, no API: a claimed event is `ASSIGNED` and
absent from `triage_view`; `assign` into the live context raises; a `PREEMPT`
returns the old context's unfinished claims to `PENDING` and leaves its
completed ones alone; a release with everything completed returns nothing;
`focus` on a job stream is refused because its agent is a subagent; `pull` of a
stream with its own context is refused; `shift` wakes subscribers with
`Shifted` after commit and not on rollback; an `Arrived` below `HIGH` makes the
control plane read nothing; the idle heartbeat releases only with nothing
pending and no reply owed; a stale focus is released on startup. The thread's
and the control plane's model calls are faked the way `test_dispatch` fakes
subagents, and the evening above is a test that drives the three loops with
the fixture scenario and asserts the shift history it leaves.

## Left out

Budget: the architecture reserves realtime quota the rest of the system cannot
draw on, and the thread should meter against it, with the attention agent told
how much is left. Liveness: the lease above. A message between contexts, which
would let `pull` take a stream that has history elsewhere. And whether the
thread should ever run tools of its own; here a conversational agent is the
same single call as a subagent, and describes what it would send.
