"""Feedback pipeline: event ingestion, consumer loop, and embedding updates.

Public API
----------
FeedbackHandler   -- processes a single event as a Qdrant profile update
FeedbackProducer  -- enqueues events (Redis stream or in-memory fallback)
FeedbackConsumer  -- async worker that drains the queue and calls FeedbackHandler
_dwell_alpha      -- maps raw dwell seconds to an embedding learning rate
"""

from .event_handlers import FeedbackHandler, _dwell_alpha
from .producer import FeedbackProducer
from .consumer import FeedbackConsumer

__all__ = [
    "FeedbackHandler",
    "FeedbackProducer",
    "FeedbackConsumer",
    "_dwell_alpha",
]
