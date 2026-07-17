"""asr-gate: certified conformal transcription-triage for Mandarin ASR.

A thin wrapper over ``relmetrics`` (the portfolio's shared reliability-
metrics toolkit) plus one new-math module, :mod:`asr_gate.ltt`, which
implements the Learn-then-Test selective-risk certificate that
``relmetrics.conformal`` does not (yet) provide. See
``apps-design/03-APP-aishell-asr-audit.md`` for the full design spec this
package implements.
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
