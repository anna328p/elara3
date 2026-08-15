"""Synthetic events, so the pipeline has something to chew on without a live world.

The set is deliberately mixed: things that clearly need answering now, things
that clearly do not, a pair that belongs together, and a job result that changes
how an earlier message should be answered.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from .events import BaseEvent, JobEvent, MessageEvent, Priority, ScheduledEvent
from .queue import EventQueue
from .store import utcnow


def synthetic_events() -> list[tuple[BaseEvent, Priority]]:
    """Ten events, timed relative to now so their ages read sensibly."""
    now = utcnow()

    def ago(**kwargs: float) -> datetime:
        return now - timedelta(**kwargs)

    return [
        (
            MessageEvent(
                timestamp=ago(seconds=40),
                description="Mira is mid-conversation in #workshop and just asked a direct question",
                sender="mira",
                channel="#workshop",
                body="wait, so did the render finish or did it choke on the alpha channel again?",
            ),
            Priority.REALTIME,
        ),
        (
            JobEvent(
                timestamp=ago(minutes=2),
                description="Overnight render job finished",
                job_id="render-0412",
                outcome="succeeded",
                summary=(
                    "Rendered 240 frames at 1440p in 6h11m. Alpha channel preserved; "
                    "two frames (117, 118) show banding in the gradient."
                ),
            ),
            Priority.NORMAL,
        ),
        (
            ScheduledEvent(
                timestamp=ago(minutes=3),
                description="Reminder fired: post the weekly devlog",
                fires_at=ago(minutes=3),
                note="Post the devlog before Friday evening. Draft is in workspace notes.",
            ),
            Priority.HIGH,
        ),
        (
            MessageEvent(
                timestamp=ago(minutes=12),
                description="Direct message from Tobias asking about collaborating",
                sender="tobias",
                channel="dm",
                body=(
                    "hey — liked the piece you posted last week. any interest in doing "
                    "something together for the winter showcase? deadline is the 30th."
                ),
            ),
            Priority.HIGH,
        ),
        (
            MessageEvent(
                timestamp=ago(minutes=9),
                description="Follow-up from Tobias with the details he forgot",
                sender="tobias",
                channel="dm",
                body="forgot to say: submissions are max 3 minutes, and they want a rough cut by the 20th.",
            ),
            Priority.NORMAL,
        ),
        (
            ScheduledEvent(
                timestamp=ago(hours=1),
                description="Weekly memory consolidation timer fired",
                fires_at=ago(hours=1),
                note="Look over the week's memories, merge duplicates, prune anything stale.",
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
            ),
            Priority.LOW,
        ),
        (
            MessageEvent(
                timestamp=ago(hours=5),
                description="Mentioned in a busy group chat, no question directed at the character",
                sender="quill",
                channel="#general",
                body="ha, elara's been doing this for months, ask them about the gradient thing",
            ),
            Priority.LOW,
        ),
        (
            MessageEvent(
                timestamp=ago(days=3),
                description="Unanswered question from Ines, going stale",
                sender="ines",
                channel="dm",
                body="do you still have the brush pack you were using in august? no rush",
            ),
            Priority.NORMAL,
        ),
        (
            MessageEvent(
                timestamp=ago(days=2, hours=4),
                description="Automated platform notification, no action implied",
                sender="noreply@gallery.example",
                channel="email",
                body="Your monthly stats: 3 new followers, 412 views. View your dashboard.",
            ),
            Priority.BACKGROUND,
        ),
    ]


async def seed(queue: EventQueue) -> list[int]:
    """Put the synthetic events on the queue; returns their ids."""
    return [await queue.submit(event, priority) for event, priority in synthetic_events()]
