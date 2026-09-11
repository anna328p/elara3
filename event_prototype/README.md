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
uv run python -m event_prototype streams        # the ongoing loci they belong to
uv run python -m event_prototype triage --dry-run   # the prompt, no API call
uv run python -m event_prototype triage             # the real thing
uv run python -m event_prototype sweep              # reconsider the backlog
uv run python -m event_prototype context 1      # the conversation stream 1 routes to
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

**Archived** is nobody's. The event drops out of both passes and survives in
`event_actions` and `list --all`.

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
someone asked somewhere else. That case is why the tool keeps its shape: a
same-stream assignment hands work to the stream's context (next section), and a
cross-stream one goes there too when the streams already share a context;
otherwise it will become a message between two contexts. The pass report
prints the streams each assignment spanned, so a crossing is visible when it
happens.

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

## Contexts

A context is the conversation an agent is having, stored as a series of turns
and replayed as a Messages API call. Each stream routes to at most one live
context, opened on the first assignment in that stream and kept from then on;
several streams may share one, and re-pointing a stream is how it moves to a
fresh context when the old one is collapsed. The context owns the subagent's
identity, so an assignment in a stream that already has one goes to the same
subagent that handled it before, with everything it was shown and everything it
said still in front of it. `context <stream-id>` prints that transcript.

A turn is exactly one API message: a role and a list of the API's own content
blocks, kept verbatim. That is what lets tool calls live in the same record
without a side table. A call is a `tool_use` block in an assistant turn and its
result a `tool_result` block in the user turn that follows, so storing messages
whole keeps every pair intact, along with the signatures on thinking blocks that
the API checks on replay. Tool definitions are not stored; like the system
prompt, they are invariants supplied at call time. A turn may carry the id of the
event that occasioned it, and the events and the brief that triage sends are
each a turn of their own with that link, while the subagent's reply has none.

Replay (`contexts.to_api`) is where the API's rules are met rather than the
store's: consecutive same-role turns merge into one message because roles must
alternate, tool results go first within a merged user message, and a transcript
whose last turn holds unanswered tool calls is refused, since that is the agent
waiting on results and not something the API will continue. Nothing about
caching is written into a turn; the request asks for it, so the bytes sent are
the bytes stored.

Caching works on the prefix. The subagent's framing is the system prompt and
never changes, each event and brief is a new turn at the end, and the request
carries a top-level cache marker that lands on the last block sent. The API
looks back from there for the prefix the previous call wrote, so each call in a
stream reads the last one and caches through its own; the pass report prints
the cached and newly cached input tokens for every subagent call. Two limits are
worth knowing. A prefix under 1024 tokens (2048 on Haiku) is not cached at all,
so a stream's first exchange or two silently pay full price, and the default
entry lives five minutes, so the saving lands within a burst of activity rather
than across the hours between bursts; the one-hour option exists and is a
decision to make with numbers. Assignments that cross streams without sharing
a context, or belong to none, still go to a one-shot subagent with a single-turn
conversation and ask for no caching, since nothing will read it back.

## Shape of the code

| module | what it holds |
| --- | --- |
| `events.py` | the `Event` protocol, `Priority`, and the concrete kinds |
| `agents.py` | agent identity: the role and UUID of a row the store mints when an agent starts |
| `streams.py` | stream identity: the kind of locus, its key and its label |
| `contexts.py` | a turn, and the transform from a transcript to an API message list |
| `store.py` | the six SQLAlchemy rows, the resolvers and the spawner, and engine setup |
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

`events` holds current state, one row per event. `event_actions` is the append-only
record of what was decided about each one: a timestamp, the action, the reason or
instructions, and the agent it is attributed to. Assignments also name the
subagent the work went to, so a dispatch and the report that follows it can be
tied together. State changes and their action rows are written in the same
transaction, so status and reasoning can never disagree.

`agents` is what those attributions point at: one row per agent, holding its
role, minted by the store when the agent starts work. `event_actions` and `contexts`
reference it by foreign key, and SQLite is told to enforce them, so nothing can
be attributed to an agent that was never minted and the role is written once.

`streams` holds identity and routing: which context, if any, work in the stream
goes to. It caches neither a last-activity stamp nor an event count: both are
aggregates over `events`, and the listing needs a GROUP BY anyway, so a cached
copy would be one more thing that can be wrong — and would be, since events do
not arrive in the order they happened.

`contexts` holds the agent whose conversation it is, and `turns` the
conversation, ordered by id within a context in the same way `event_actions` is
ordered within an event. Turns are appended before the subagent is called and
after it replies, so a call that fails leaves the events it was shown in the
transcript, which is what happened; the failure itself goes to `event_actions`.

Reads are indexed on `(event_id, timestamp)` and never go per-event. Triage is a
single query, since the digest needs no history at all; the sweep is two, one for
the events and one for the whole slice of actions they point at, however large the
backlog. An event's stream rides along in those same statements — the join is
declared on the relationship rather than requested at each call site, so it
cannot be forgotten on a path that later renders a prompt, and the counts above
are the ones the tests assert.

Events are never deleted. Archiving stamps `archived_at` and drops the event from
both passes, but `list --all` still shows it, actions and all.

## What it doesn't do

Subagents are a single model call with no tools, so they describe how they would
handle an event rather than doing it; the transcript can hold a tool loop, but
nothing runs one. Contexts only grow: there is no collapse or compaction, and no
message from one context to another, so a cross-stream assignment still goes to
a subagent that remembers nothing. There is no heartbeat, so the sweep is a
command you run rather than something that fires on idle time or a backlog
threshold, and nothing yet decides when a pass should happen. No token budgets,
and no subagent-of-a-subagent — those come with the real framework.
