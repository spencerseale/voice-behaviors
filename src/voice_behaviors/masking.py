"""PII masking on the OTel export path."""

from typing import Any


def noop_masking_function(attributes: dict[str, Any]) -> dict[str, Any]:
    """No-op stand-in for the production PII redactor (proprietary).

    In deployment this scrubs sensitive fields out of span content before it
    leaves the process; here it passes the attribute mapping through unchanged so
    no redaction logic is shared -- the *registration* is what matters. A real
    redactor would return a copy with values for keys like the gen_ai message
    content / `lk.*` transcript fields replaced by "[REDACTED]".
    """
    return attributes


class MaskingSpanProcessor:
    """Apply a masking function to span content before the Braintrust exporter.

    The OTel path has NO equivalent of `braintrust.set_masking_function` -- that
    hook only scrubs native-SDK spans, and `BraintrustSpanProcessor` exports
    spans verbatim. To redact on this path we wrap the BraintrustSpanProcessor
    and rewrite each span's attributes (and per-event attributes, where the voice
    transcripts live) on end, before the inner processor serializes them.

    `ReadableSpan.attributes` returns `MappingProxyType(self._attributes)` fresh
    on each read, so reassigning `_attributes` here is reflected at export time.
    Reaching into `_attributes`/`_events` is private SDK API -- the only place
    redaction can happen on the OTel path -- and is wrapped in try/except so a
    masking error can never drop a span.
    """

    def __init__(self, inner: Any, masking_function: Any) -> None:
        self._inner = inner
        self._mask = masking_function

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        self._inner.on_start(span, parent_context)

    def on_end(self, span: Any) -> None:
        self._apply(span)
        self._inner.on_end(span)

    def _on_ending(self, span: Any) -> None:
        # Forward the optional pre-end hook (newer OTel SDKs) to the inner
        # processor; masking itself happens in on_end, before the inner queues.
        on_ending = getattr(self._inner, "_on_ending", None)
        if callable(on_ending):
            on_ending(span)

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)

    def _apply(self, span: Any) -> None:
        try:
            attrs = getattr(span, "_attributes", None)
            if attrs:
                span._attributes = dict(self._mask(dict(attrs)))
            for event in getattr(span, "_events", None) or ():
                ev_attrs = getattr(event, "attributes", None)
                if ev_attrs and hasattr(event, "_attributes"):
                    event._attributes = dict(self._mask(dict(ev_attrs)))
        except Exception:  # never let masking break export
            pass
