# event_prototype

A working sketch of the elara3 event queue (see `../architecture.md`, §"The event
queue"): events accumulate in a priority queue, a triage model sorts them, and
each piece of work goes to a subagent.

## Running it

```sh
nix develop          # python 3.12 + uv + sqlite
uv sync --extra dev
uv run pytest

uv run python -m event_prototype seed --fresh   # ten synthetic events
uv run python -m event_prototype list
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
| `store.py` | the two SQLAlchemy rows and engine setup |
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

Reads are indexed on `(event_id, timestamp)` and never go per-event. Triage is a
single query, since the digest needs no history at all; the sweep is two, one for
the events and one for the whole slice of log they point at, however large the
backlog.

Events are never deleted. Archiving stamps `archived_at` and drops the event from
both passes, but `list --all` still shows it, log and all.

## What it doesn't do

Subagents are a single model call with no tools, so they describe how they would
handle an event rather than doing it. There is no heartbeat, so the sweep is a
command you run rather than something that fires on idle time or a backlog
threshold, and nothing yet decides when a pass should happen. No token budgets,
and no subagent-of-a-subagent — those come with the real framework.
