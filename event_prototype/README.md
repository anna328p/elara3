# event_prototype

A working sketch of the elara3 event queue (see `../architecture.md`, §"The event
queue"): events accumulate in a priority queue, a triage model sorts them, and
each piece of work goes to a subagent.

## Running it

```sh
nix develop          # python 3.12 + uv + sqlite
uv sync --extra dev
uv run pytest

uv run python -m event_prototype seed --fresh   # a dozen synthetic events
uv run python -m event_prototype list
uv run python -m event_prototype streams        # the ongoing contexts they belong to
uv run python -m event_prototype triage --dry-run   # the prompt, no API call
uv run python -m event_prototype triage             # the real thing
uv run python -m event_prototype sweep              # reconsider the backlog
```

The live passes need `ANTHROPIC_API_KEY`, read from the `.env` at the repo root.
Models, efforts and the database path are set in `config.toml`.

## Two passes

An event is in one of three states, and each state is somebody's business.

**Pending** is triage's. `triage` renders every pending event in full
(`templates/triage.md.j2`) and hands it to a model whose only way to act is three
MCP tools:

- `handle_one_event` — spawn a subagent for an event that stands alone
- `handle_event_sequence` — one subagent for several events, in order
- `defer_event` — this does not need attention now; say why

Triage is on the critical path, so it is shown as little as possible: the
deferred backlog appears only as one line per event, enough to notice that a
deferred event bears on a live one and pull it back in, and cheap enough that a
long backlog cannot bury the message someone is waiting on.

That line is written for the job. When an event is set aside, a small model
(`digest_model`, Haiku by default) is asked for one line naming the concrete
subject — who is involved, what it concerns, what is outstanding — so the next
triage can match it against whatever has just arrived. The event's own
`description` says how it was filed rather than what it is about, and makes a
poor substitute; it stands in only until a line has been written. Generation is
spawned in the background and drained with the subagents, so no pass waits on
it, and a failure just leaves the previous line standing.

**Deferred** is the sweep's. `sweep` runs on a schedule or in idle time with
nobody waiting, sees the backlog in full with each event's complete history, and
has the dispositions that triage should not be making in a hurry:

- `escalate_event` — this does need doing; set its real priority and return it to triage
- `handle_event` — needs work now, and is self-contained enough to dispatch directly
- `archive_event` — this will never need action
- `keep_deferred` — still not worth acting on, with a fresh account of why

**Archived** is nobody's. The event drops out of both passes and survives in the
log and `list --all`.

## Streams

A stream is a durable locus of activity: a room, a correspondent, a recurring
job. Events derive their own — `MessageEvent` from its venue and conversation,
`JobEvent` and `ScheduledEvent` from the series a run belongs to — so every
message in a room necessarily lands on the same stream rather than being sorted
onto one. Belonging is optional: a one-off alarm or a single job run says so by
naming no stream at all.

Venue and stream are orthogonal. The venue is the medium — Discord, email — and
nothing is ever "the email stream"; it only qualifies a conversation's id, so
that `#general` on two platforms stays two streams. What a two-party exchange
needs beyond that is `direct`, because an id alone cannot say whether a place is
a room or a relationship, and the two are not the same kind of thing.

The prompts show the stream as an attribute on each event, which sharpens
`handle_event_sequence` rather than replacing it. A stream is where an event
happened, not what it is about, so sharing one is a reason to look for a
connection and not evidence of one — and the clearest reason to group events
runs the other way, across streams, as when a job's result answers a question
someone asked somewhere else. That case is why the tool keeps its shape, and it
is where persistent contexts will land: once a subagent is a stream's long-lived
cached context, a same-stream sequence means handing work to that context and a
cross-stream one means a message between two. The pass report prints the streams
each assignment spanned, so a crossing is visible when it happens.

Streams capture the locus, not the topic. A `#general` mention and a later DM
from the same person are different streams, so it is still the digest line that
links them; the two signals are orthogonal and both are needed.

The handling tools *schedule* rather than execute: each call mints a subagent,
records the assignment, spawns an asyncio task and returns immediately, so
subagents run in parallel while the pass keeps working. A pass ends when every
spawned task has finished, and prints what each subagent reported back.

Within one pass an event gets exactly one disposition; a second call for the same
event is refused with an error the model can read. Across passes it can be
revisited freely — that is the point.

The MCP servers run in-process, connected over an in-memory transport: the same
protocol as a remote server, minus the subprocess.

## Shape of the code

| module | what it holds |
| --- | --- |
| `events.py` | the `Event` protocol, `Priority`, and the concrete kinds |
| `agents.py` | agent identity: a role and a UUID, minted when an agent starts |
| `streams.py` | stream identity: the kind of context, its key and its label |
| `store.py` | the three SQLAlchemy rows and engine setup |
| `queue.py` | `EventQueue`: submit, edit, the dispositions, and the two views |
| `render.py` + `templates/` | rows → prompts |
| `tools.py` | the two tool sets and the dispatcher they act on |
| `runner.py` | driving one model pass against one tool set |
| `triage.py` / `sweep.py` | the two passes, tying the above together |
| `subagent.py` | the model call that handles assigned events |
| `digest.py` | the model call that writes an event's backlog line |

Nothing here holds global state: the queue owns its engine, the tools close over
a dispatcher, and both are passed in explicitly.

## State and record

`events` holds current state, one row per event. `event_log` is the append-only
record of what was decided about each one: a timestamp, the action, the reason or
instructions, and the agent it is attributed to. Assignments also name the
subagent the work went to, so a dispatch and the report that follows it can be
tied together. State changes and their log rows are written in the same
transaction, so status and reasoning can never disagree.

`streams` holds identity only. It caches neither a last-activity stamp nor an
event count: both are aggregates over `events`, and the listing needs a GROUP BY
anyway, so a cached copy would be one more thing that can be wrong — and would
be, since events do not arrive in the order they happened.

Reads are indexed on `(event_id, timestamp)` and never go per-event. Triage is a
single query, since the digest needs no history at all; the sweep is two, one for
the events and one for the whole slice of log they point at, however large the
backlog. An event's stream rides along in those same statements — the join is
declared on the relationship rather than requested at each call site, so it
cannot be forgotten on a path that later renders a prompt, and the counts above
are the ones the tests assert.

Events are never deleted. Archiving stamps `archived_at` and drops the event from
both passes, but `list --all` still shows it, log and all.

## What it doesn't do

Subagents are a single model call with no tools, so they describe how they would
handle an event rather than doing it. There is no heartbeat, so the sweep is a
command you run rather than something that fires on idle time or a backlog
threshold, and nothing yet decides when a pass should happen. No token budgets,
and no subagent-of-a-subagent — those come with the real framework.
