"""Synthetic events, so the pipeline has something to chew on without a live world.

The set is deliberately mixed: things that clearly need answering now, things
that clearly do not, a pair that belongs together, and a job result that changes
how an earlier message should be answered. Every priority appears at least once,
placed the way an ingestion layer would place it: a person addressing the
character is active, a room's chatter and an unhurried question are async, a
job worth interrupting for is a nudge, and the rest is background.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from .events import BaseEvent, JobEvent, MessageEvent, Priority, ScheduledEvent
from .queue import EventQueue
from .store import utcnow


def synthetic_events() -> list[tuple[BaseEvent, Priority]]:
    """A dozen events, timed relative to now so their ages read sensibly.

    Submitted out of chronological order on purpose: a stream's last activity
    is the newest event on it, not the one that happened to be queued last.
    """
    now = utcnow()

    def ago(**kwargs: float) -> datetime:
        return now - timedelta(**kwargs)

    return [
        (
            MessageEvent(
                timestamp=ago(seconds=40),
                description="Mira is mid-conversation in #workshop and just asked a direct question",
                sender="mira",
                venue="discord",
                conversation="#workshop",
                body="wait, so did the render finish or did it choke on the alpha channel again?",
            ),
            Priority.ACTIVE,
        ),
        (
            JobEvent(
                timestamp=ago(minutes=2),
                description="Overnight render job finished",
                # A one-off render, so it belongs to no stream, and so it can
                # only be a nudge or background; a nudge, because Mira is asking.
                job_id="render-0412",
                outcome="succeeded",
                summary=(
                    "Rendered 240 frames at 1440p in 6h11m. Alpha channel preserved; "
                    "two frames (117, 118) show banding in the gradient."
                ),
            ),
            Priority.NUDGE,
        ),
        (
            ScheduledEvent(
                timestamp=ago(minutes=3),
                description="Reminder fired: post the weekly devlog",
                fires_at=ago(minutes=3),
                note="Post the devlog before Friday evening. Draft is in workspace notes.",
                schedule="weekly-devlog",
            ),
            Priority.NUDGE,
        ),
        (
            MessageEvent(
                timestamp=ago(minutes=12),
                description="Direct message from Tobias asking about collaborating",
                sender="tobias",
                venue="discord",
                conversation="dm:tobias",
                body=(
                    "hey — liked the piece you posted last week. any interest in doing "
                    "something together for the winter showcase? deadline is the 30th."
                ),
                direct=True,
            ),
            Priority.ACTIVE,
        ),
        (
            MessageEvent(
                timestamp=ago(minutes=9),
                description="Follow-up from Tobias with the details he forgot",
                sender="tobias",
                venue="discord",
                conversation="dm:tobias",
                body="forgot to say: submissions are max 3 minutes, and they want a rough cut by the 20th.",
                direct=True,
            ),
            Priority.ACTIVE,
        ),
        (
            # Same stream as Mira's question, and nothing to do with it: sharing
            # a stream is not on its own a reason to handle two events together.
            MessageEvent(
                timestamp=ago(minutes=20),
                description="Passing remark in #workshop, no question in it",
                sender="hal",
                venue="discord",
                conversation="#workshop",
                body="the new brush engine update broke my pressure curves btw",
            ),
            Priority.ASYNC,
        ),
        (
            ScheduledEvent(
                timestamp=ago(hours=1),
                description="Weekly memory consolidation timer fired",
                fires_at=ago(hours=1),
                note="Look over the week's memories, merge duplicates, prune anything stale.",
                schedule="memory-consolidation",
            ),
            Priority.BACKGROUND,
        ),
        (
            JobEvent(
                timestamp=ago(hours=6),
                description="Archive backup job failed",
                job_id="backup-1183",
                outcome="failed",
                summary="rsync exited 23 after 4.2GB — permission denied on /workspace/old/.cache",
                job="backup",
            ),
            Priority.BACKGROUND,
        ),
        (
            # The same job failing the same way a day earlier. Two events on one
            # stream saying the same thing is what the sweep exists to notice.
            JobEvent(
                timestamp=ago(days=1, hours=6),
                description="Archive backup job failed",
                job_id="backup-1182",
                outcome="failed",
                summary="rsync exited 23 after 4.2GB — permission denied on /workspace/old/.cache",
                job="backup",
            ),
            Priority.BACKGROUND,
        ),
        (
            MessageEvent(
                timestamp=ago(hours=5),
                description="Mentioned in a busy group chat, no question directed at the character",
                sender="quill",
                venue="discord",
                conversation="#general",
                body="ha, elara's been doing this for months, ask them about the gradient thing",
            ),
            Priority.ASYNC,
        ),
        (
            MessageEvent(
                timestamp=ago(days=3),
                description="Unanswered question from Ines, going stale",
                sender="ines",
                venue="discord",
                conversation="dm:ines",
                body="do you still have the brush pack you were using in august? no rush",
                direct=True,
            ),
            Priority.ASYNC,
        ),
        (
            MessageEvent(
                timestamp=ago(days=2, hours=4),
                description="Automated platform notification, no action implied",
                sender="noreply@gallery.example",
                venue="email",
                conversation="monthly-stats",
                body="Your monthly stats: 3 new followers, 412 views. View your dashboard.",
                direct=True,
            ),
            Priority.BACKGROUND,
        ),
    ]


async def seed(queue: EventQueue) -> list[int]:
    """Put the synthetic events on the queue; returns their ids."""
    return [await queue.submit(event, priority) for event, priority in synthetic_events()]
