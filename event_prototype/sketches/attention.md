# Attention: a live context, and the triage tool that chooses it

A sketch. It describes the finished design, not the decisions behind it, using
the prototype's terms (`../README.md`): events, streams, contexts, the two
passes, the priorities, the heartbeats and the wake-ups that `watch` runs on,
and the memory and people stores.

## Context

The priorities on `main` are two facts about delivery: whether an event goes to
triage or to its stream's context, and whether anyone is woken for it.
`background` and `nudge` go to triage; `async` and `active` go to the context;
`nudge` and `active` wake their reader now. The README says that until
contexts have heartbeats of their own, the context-bound events pass through
triage anyway and are assigned into their context from there.

This design gives contexts their execution, so that stops. A context-bound
event is delivered to its context at `submit` and never enters the pending
set. A runner serves contexts: one of them, the *live* context, on every
arrival, and the rest on a heartbeat of their own, a row in `heartbeats` like
triage's and the sweep's. Triage still gives events to contexts: its tools act
on delivered events as well as pending ones. It also decides which context is
live, using one additional tool, and is shown the contexts that are waiting so
it can decide. Attention can rest on any context, a conversation or a task.

## Delivery

`submit` already opens the stream on first sight in the same transaction as
the event. For a context-bound priority it also resolves the stream's context,
opening one if needed, sets the event to `ASSIGNED`, and writes a `DELIVERED`
action with the context's agent as both actor and assignee, since the agent
received it. The event is never `PENDING`, so it is not in triage's pending
list, not counted by triage's workload, and never the sweep's. Triage sees it
under its context, below, and can act on it there. Its turn is not appended
yet: turns are the record of what the agent was shown, and an agent whose
call is in flight has not been shown this event. The runner appends the turn
when it next runs the context.

`ASSIGNED` is a fourth status, set here and by `assign`. `record_failure` sets
events back to `PENDING` instead of leaving them unchanged, and `Action` gains
`DELIVERED`. Two concurrent passes were already unsafe without the status.

A context *owes a call* when it has delivered events without a turn, or when
its last turn is a user turn. The first is a conversation with messages
unanswered. The second is a tool loop that stopped between iterations: the
subagent loop on `main` stores each tool result before the next request, so a
context left there ends on a user turn and `to_api` can send it again. Both
are one query over `events` and `turns`.

Assignment is delivery too. When triage hands events to a context, `assign`
sets them to `ASSIGNED` with the brief on the action, as now, and wakes the
runner instead of spawning a task. Whether the runner runs the context at
once is the priority's second fact, as at `submit`: an immediate event, a
`nudge` or an `active`, is run now, and a `background` or `async` one waits
for the context heartbeat with its brief, so many assignments batch into one
run. The live context is run now regardless, since it is served on every
arrival; a brief handed into a live conversation is how triage tells it that
the job it was asked about has reported. The dispatcher runs only one-shot
subagents, the events that belong to no context. So every context has one
executor, the runner, which holds an in-process `asyncio.Lock` per context
for the runs it starts in parallel on a heartbeat. A pass ends when its
decisions are recorded; the reports of context runs arrive later, and `watch`
prints them as they land.

## The conversational agent

The prototype mints a subagent for every context. Its system prompt describes a
task handler: triage hands you events, report back. That framing is wrong for a
conversation, and it would be the identity that live replies are attributed
to. So the role of a context's agent depends on what the context is for.

```python
class AgentRole(StrEnum):
    TRIAGE = "triage"
    SWEEP = "sweep"
    #: The agent of a channel or direct stream's context. It represents the
    #: character in that one exchange, on a heartbeat and live alike.
    CONVERSATIONAL = "conversational"
    #: A task handler: a job stream's context, or one-shot work.
    SUBAGENT = "subagent"

#: Conversational joins: its pass is the runner serving every conversational
#: context that owes a call. A subagent still runs when handed work, or when
#: its context is live, never on a clock.
LOOP_ROLES = (AgentRole.TRIAGE, AgentRole.SWEEP, AgentRole.CONVERSATIONAL)
```

`resolve_context` mints a conversational agent for a channel or direct stream
and a subagent for a job stream. One-shot assignments still get a subagent. The
role selects the system prompt, `conversation_system.md.j2` or the existing
`subagent_system.md.j2`. A context's role never changes, so its cached prefix
does not change either. The conversational prompt states that the agent
represents the character in this one exchange, that messages arrive as turns,
sometimes with a brief from triage and, when live, with a note that the
conversation is happening now, and that its report is the text it would send,
in the character's voice. It has the subagent's tools, `memory`,
`register_person` and `link_person`, and one more, `schedule_check_in`, which
on `main` only the passes have; here it asks for a run of this one context in
so many minutes with a note, for "see if they wrote back". In the real
framework this is the agent that calls the personality model's tools.

## Who is speaking

`main` gives the character known people: a name, a root page at
`/memories/people/<slug>.md`, and handles, one `(venue, username)` per row,
each belonging to one person. It looks up the senders of a batch in one query,
and the event turn of a known sender carries a `<sender>` element with the
person's name, the path of their page, and the page's current text, on every
event.

In a conversational agent's context, a person's profile is presented once: with
the first event from them in that context, and not with later ones. The
context keeps every turn, so the agent has already read the profile, and
repeating it adds tokens to a prefix that would otherwise be cached. The
runner applies the rule whenever it appends event turns. A one-shot subagent,
which the dispatcher runs, has no context to compare against, so it is shown
every known sender's profile, as now.

The turn's content stays verbatim: the profile text is pasted into it, because
the content is the record of what the API was sent, and a placeholder filled at
replay would either pin the version, which gains nothing over the copy, or
substitute the page's head, which changes the cached prefix and loses what the
agent saw. What the copy lacks is provenance, and a join table supplies it:

```python
class MemoryTransclusionRow(Base):
    """A memory version whose text a turn included, pasted in full."""

    __tablename__ = "memory_transclusions"

    #: Composite key, `turn_id` leading, so this is also the index on
    #: `turn_id` that every read needs: the reads start from a context's
    #: turns and ask what each included. Nothing looks up by version.
    turn_id: Mapped[int] = mapped_column(ForeignKey("turns.id"), primary_key=True)
    version_id: Mapped[int] = mapped_column(
        ForeignKey("memory_versions.id"), primary_key=True
    )
```

It references versions rather than entries: the version determines the entry
and records exactly which text was shown. A turn can include several pages,
as a merged batch with two senders does, and the table is not specific to
profiles; any page the harness pastes into a turn is recorded the same way,
so implicit recall, when it arrives, records what it surfaced in the same
place, and "already shown in this context" is one query for both. That query
joins the context's turns to their transclusions, the versions to their
entries, and the entries to the people whose root page they are.

When event turns are appended to a context, the batch's senders are looked up
as now, the people already presented in the context are subtracted, and
within the batch only the first event from each remaining person gets the
profile. The turn opens with the `<sender>` element and the `<event>` follows
it, so the template moves the element ahead of the event. The rule concerns
profiles presented, not senders seen: a sender who was unknown when they first
wrote, and whom the agent later registers or links, is presented with their
next event.

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
    #: Who decided: the triage agent on a divert; the context's agent on a
    #: release, whether it yielded, finished, or the runner timed out on its
    #: behalf.
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id"))
    reason: Mapped[str]
```

The context a `PREEMPT` took the runner from is the context of the row before
it. No column marks it; the triage prompt derives it when it lists that
context, below.

`EventQueue.focus()` returns the current row or `None`. `shift()` appends a row
and, after the commit, wakes subscribers with a `Shifted`. `assign` wakes with
an `Assigned` the same way. Both are new kinds of wake alongside `Arrived` and
`Elapsed`, and `Arrived` gains the context the event was delivered to, or
`None` when it went to triage, since that is part of why the wake fired.

```python
@dataclass(frozen=True, slots=True)
class Shifted:
    """The focus moved."""

    shift: Shift
    context_id: int | None
    at: datetime

@dataclass(frozen=True, slots=True)
class Assigned:
    """Triage handed events to a context, with a brief."""

    context_id: int
    event_ids: tuple[int, ...]
    #: The events' priorities, so the runner can tell whether any is immediate
    #: without a read.
    priorities: tuple[Priority, ...]
    at: datetime

type Wake = Arrived | Elapsed | Shifted | Assigned
```

## Context heartbeats

The heartbeat a context-bound event waits for is a `heartbeats` row for the
conversational role, declared by `watch` from `[heartbeat_seconds]` like the
other two, and served by `beat` the same way: every pass for the role fires
what is due and re-times what recurs, and `next_heartbeat_due` covers the
three roles at once. The pass for this role is not a model pass. It is the
runner running every conversational context that owes a call and is not live,
as parallel tasks, and `workload` for the role counts those contexts. An
`async` arrival waits for it, and so does an `active` one that triage neither
diverted to nor handed on with a brief. A subagent's context has no heartbeat:
it runs when handed work or when live, and a preempted one waits for triage
to divert back to it.

A check-in can name a context. `heartbeats` gains a nullable `context_id`,
`None` for the role-wide rows, and a conversational agent's
`schedule_check_in` writes a one-shot row for its own context, attributed to
it. When the tick comes due, the beat for the role returns it, and the runner
runs that context with the note as the brief, whether or not it owes a call.
The prompt shows a conversational agent when its context's next run is, as
the passes are shown theirs, so it can judge whether a check-in is worth it.

## The runner

The runner is in `attention.py`. Running a context means: take its lock;
append a turn for each delivered or assigned event without one, with profiles
as above, followed by triage's brief if the events carry one, a check-in's
note if that is why the run is happening, and otherwise `live.md.j2` if the
context is live and `heartbeat.md.j2` if not; run the agent's tool loop,
storing every message; complete the events with the final reply as their
report; release the lock. A live run uses `live_effort`; any other run uses
`subagent_effort`. A live run of a conversational agent also offers
`yield_focus(reason)`.

The runner is invoked from two places. `watch` calls it as the conversational
role's pass, on that role's heartbeat, for every waiting context at once. And
the runner's own subscription on the queue, alongside `watch`'s, serves the
live context and immediate assignments:

An `Arrived` delivered to the live context: run it now. Arrivals during a run
are turned at the next run, so a burst becomes one call, and `to_api` merges
them into one user message.

An `Assigned`: run the context now if it is live or any of the events is
immediate; otherwise nothing, and the events wait for the heartbeat.

A `Shifted`: if the focus moved to a context that owes a call, run it now. The
previous context is left as it is; anything it owes is served at the heartbeat
or when attention returns.

After a live run, the runner decides whether the context still holds it. A
subagent's context is released as soon as its loop ends owing nothing, since
a task that has reported is done. A conversational context is released when
its agent called `yield_focus`, or when the runner's wait, whose timeout is
`focus_idle_seconds` from the live context's last event, elapses with nothing
owed. Each release is a `RELEASE` shift attributed to the context's agent,
with the reason saying which.

After any run ends, the runner asks `digest_model` in the background for one
line saying what the context is doing now, from its last few turns, and stores
it in `contexts.summary`, following the precedent of `events.digest`: derived,
absent until written, and the last reply's first line stands in until then.
The line is written for the triage prompt, below. A failure leaves the old
line.

On startup, a focus left by a previous run is released. On shutdown, the loop
releases in a `finally`. The real framework needs a lease here, renewed each
iteration, so that a dead runner does not keep holding focus; the prototype
has one process and a `finally`.

## Triage under attention

Triage's prompt gains a `<contexts>` block above its events, listing the live
context and every context that owes a call. Each entry gives the context id,
its streams' titles, its agent's role, its state, its summary line, and for
each delivered event without a turn, the sender and the description. The
state is one of: live since a time, with events attended and seconds since the
last one, and whether a reply is owed; preempted at a time with the reason of
the shift that took the runner, derived from the attention history; or waiting
since its oldest unturned delivery, with when its heartbeat next runs it. A
subagent context that is paused between iterations of a loop shows how many
iterations it has run.

Triage acts on both. Its event tools take delivered events as well as
pending ones, by the ids the block lists. `handle_one_event` on a delivered
event has the runner run its context now with the brief instead of leaving it
for the heartbeat, and the live context is no exception. `handle_event_sequence`
routes as before, to the one context every event in it belongs to or, when
their streams share none, to a one-shot subagent; a delivered event already
belongs to its context, so a sequence containing one must route there, which
is the case of a stream carrying both a `background` event and an `active`
one. Both tools take an optional `context_id` that delivers the events into
that context instead of the one their streams route to. That is how a job's
pending result is handed into the conversation that asked for it, and it is
the message between contexts in its cheapest form: an event has one context,
so the job's own context does not see the result. `defer_event` takes pending
events only: a delivered event that triage does not act on is answered at its
context's heartbeat, which is what deferring it would mean. The rule that
every event id must appear in exactly one call covers the pending list;
delivered events are optional to act on.

Triage also gets one more tool. `divert_attention(context_id, reason)` makes
that context live: an `ACQUIRE` if nothing was live and a `PREEMPT` otherwise.
It is refused for a context that is already live. There is no tool to hold
and none to release. A pass holds by seeing the block and not diverting; a
context is released by its own agent or the runner, as above. The schedule
tools are unchanged, and `set_heartbeat_interval` with `everyone` now covers
the conversational role too, so a pass can make every waiting conversation
run more often for an hour.

The guidance in the template is short. Divert to a conversation when a person
is replying within seconds, which is usual for a direct stream at `active` and
unusual for a room mention. Do not divert for a single message when a reply is
owed in the live context; hand the other person's message to their context
with a brief instead, or leave it for the heartbeat, since a switch means an
uncached first call in the new context. Divert back to a preempted task when
nothing else needs the runner, so the task finishes. A task and a
conversation are both contexts; the block says which is which.

`early` on `main` says whether a batch of wakes calls for a triage pass ahead
of its heartbeat, and answers yes for a `nudge` or an `active` arrival. Here
it answers yes for a `nudge`, for an `active` arrival delivered to a context
that is not live, and for a `RELEASE`. An `active` arrival in the live context
needs no decision, and the runner serves it; `watch` reads the focus once per
batch of wakes to tell the two apart. A release runs a pass so that a paused
task, or the next conversation waiting, gets the runner without waiting for
the heartbeat. The pass is skipped when its workload is empty and no check-in
is due, as on `main`; for triage the workload now counts contexts that owe a
call and are not live as well as pending events, since a release with a
paused task is a decision the pass exists for.

## Config

```toml
live_effort = "low"
focus_idle_seconds = 180

[heartbeat_seconds]
triage = 300
sweep = 3600
conversational = 60
```

The runner uses `subagent_model` for both kinds of run; the live run differs
only in effort.

## Commands

```sh
uv run python -m event_prototype watch                  # the passes and the runner, until ^C
uv run python -m event_prototype watch --play evening   # the same, fed a scripted evening of events
uv run python -m event_prototype focus                  # the current focus and every shift so far
uv run python -m event_prototype contexts               # every context owing a call, with its state and summary
```

`watch` starts the runner's subscription beside its own and runs the
conversational role's heartbeat pass through it. It prints each shift as it
happens and each run with its cache reads, so a preemption shows up as a call
with no cache reads. The queue wakes only its own process, so an event written
from a second terminal would wait for a heartbeat. `--play` names a scenario
in `fixtures.py`, a timed list of events submitted from inside the process
while the loops run. That is how the flow below is run by hand.

## Execution flow

A scenario, followed through the process. `watch` opens the queue, declares
the three standing schedules from the config, releases any focus left by a
previous run, and starts the runner's subscription beside its own. The sweep
runs on its own schedule and is not part of this.

A research job's context is live. Triage diverted to it at an earlier pass,
and the runner is partway through the subagent's loop, storing each message
as it goes. Its summary line says what it is reading.

Alice sends a direct message: "hey could i get you to look at something real
quick?" The ingestion layer submits it at `active`, because a direct message
is a person in a live exchange. `submit` opens a context for her stream with a
conversational agent, sets the event to `ASSIGNED` with a `DELIVERED` action,
and wakes with an `Arrived` naming that context. The runner sees an arrival
outside the live context and does nothing. `watch` sees an active arrival
outside the live context, beats triage, which re-times its schedule, and runs
a pass. The pass has no pending events. Its `<contexts>` block shows the
research context, live for twelve minutes and mid-loop, and Alice's context,
waiting with one delivered event, her handle and her message, next due at the
conversational heartbeat in forty seconds. It calls `divert_attention` on her
context. A `PREEMPT` commits and a `Shifted` is sent.

The runner finishes the research iteration it is on, which leaves that
context ending on a user turn of tool results, and reads the new focus. It
takes Alice's lock, appends her event turn, opening with her profile and a
transclusion row since this context has presented nobody, appends the live
brief, and runs her agent at low effort with `yield_focus` available. The
reply is appended and the event completed.

Alice sends two more messages while that run is in flight. Each is delivered
to her context at `submit` and wakes both subscribers. `watch` reads the focus,
finds the arrivals are in the live context, and does not run a pass. The
runner turns both at its next run, merged into one user message, reading the
prefix the previous run cached. Nothing about these messages reaches triage.

The conversational heartbeat comes due. `watch` beats the role and runs the
runner's pass, which finds no conversational context owing a call other than
Alice's, which is live and skipped, and re-times the schedule.

Bob mentions the character in a room, at `active`. The event is delivered to
the room's context, opened now with a conversational agent. `watch` runs a
triage pass. The block shows Alice's context live with no reply owed and the
last event twenty seconds ago, and the room's context waiting with Bob's
message. The pass does not divert. It calls `handle_one_event` on Bob's event
with a brief saying the character is talking to Alice at the moment. The
event is `active`, so the `Assigned` wake has the runner run the room's
context now, with Bob's profile opening the turn if his handle is linked, and
his reply arrives within a minute. Had the pass left the event alone, or had
the event been `async`, the room's context would have run at the next
conversational heartbeat instead.

Alice's agent, replying to something she said she would send later, calls
`schedule_check_in` for fifteen minutes with the note "she said she'd send
the file". A one-shot `heartbeats` row for the conversational role names her
context. If she has gone quiet by then, the tick fires at the role's beat and
the runner runs her context with the note as the brief.

Alice writes goodnight. Her agent replies and calls `yield_focus`. A
`RELEASE` attributed to her agent commits, and the `Shifted` makes `watch` run
a triage pass. The block shows the research context preempted twenty minutes
ago, with the reason of that shift and its summary line, and nothing else
owing a call. The pass diverts to it. The runner reads the focus, takes the
lock, and sends the transcript again; the loop continues from the stored tool
results, uncached, which the report shows. When the loop ends with a report,
the runner releases the context, and the pass that follows finds nothing to
divert to.

Ctrl-C cancels the main task. The runner's `finally` releases; `watch` shuts
down as its own documentation describes.

## Verification

Tests against the in-memory database, without the API: an `active` or `async`
submit sets the event to `ASSIGNED` with a `DELIVERED` action and leaves it
out of `triage_view` and triage's workload; a context with a delivered event
owes a call, and so does one whose last turn is a user turn; running a context
appends one turn per delivered event and completes them; `assign` accepts a
delivered event, wakes the runner with `Assigned`, and spawns nothing for a
context, including the live one; the runner runs on an `Assigned` only for an
immediate event or the live context, and a `background` assignment waits for
the heartbeat; `assign` with a `context_id` delivers into that context; a
sequence holding a delivered event routes to its context or is refused;
`defer` refuses a delivered event; `divert_attention` on the live context is
refused; a `PREEMPT` leaves the previous context's turns and delivered events
as they are; a subagent context is released when its loop ends owing nothing;
a conversational context is released on yield and on the idle timeout, and
not before; `shift` wakes with `Shifted` after commit and not on rollback;
`early` is true for an active arrival outside the live context and a release,
and false for an active arrival inside it; the conversational role's beat
fires and re-times like the others, its workload counts waiting contexts and
skips the live one, and a context check-in runs only its context; the
preempted context is derived from the attention history; the first event
from a known sender in a context carries their profile with a transclusion
row, a second does not, a batch with two events from one person presents it
once, a one-shot subagent is shown it on every event, and a person registered
or linked after their first event is presented with their next. The runner's
model loop is faked the way `test_dispatch` fakes subagents. The scenario
above is a test that drives `watch` and the runner with the fixture events
and asserts the resulting shift history and ticks.

## Left out

Budget: the architecture reserves quota for live exchanges that the rest of
the system cannot use. The runner should meter against it, and triage should
be told how much is left when it decides whether to divert. Liveness: the
lease described above. A message between contexts that both contexts see;
the `context_id` on the event tools moves one event into one context, and the
same person writing from a second place needs more than that. Re-presenting
a profile whose page has changed since it was shown; the transclusion row
makes that a comparison against the page's head, but nothing does it yet. A
heartbeat for subagent contexts, so a preempted task could resume on a clock
rather than only by a divert. And the summary line's quality: it follows the
digest precedent, and whether one line from the last few turns describes a
task well enough for triage to choose between contexts is something the
scenario will show.
