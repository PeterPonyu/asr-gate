"""Per-utterance Character Error Rate (CER) after a pinned Mandarin normalizer.

Normalization pipeline (versioned; the version string travels in every
output JSON so cal/test are provably byte-identical, per design §6.3 / K1):

1. Unicode NFKC normalization.
2. Strip whitespace (all Unicode whitespace).
3. Strip punctuation: any character that is neither a CJK ideograph, a
   Latin/CJK alphanumeric, nor whitespace-already-stripped is dropped. This
   covers both ASCII and full-width Mandarin punctuation (，。！？、「」
   etc.) without a hand-maintained punctuation blocklist.
4. Numeral policy (configurable, default ``"keep"``): a DIGIT-SUBSTITUTION
   mapping between the ten Arabic digits and the ten single-character CJK
   numerals (〇一二三四五六七八九). This is a deliberate, documented scope
   limitation -- it does NOT parse multi-digit Chinese number READINGS
   (e.g. "二百三十四" for 234); it only equates digit-for-digit spellings
   (e.g. "2023" <-> "二〇二三"), which is the common ASR CER-normalization
   convention and the concrete failure mode named in the design doc
   (Whisper zero-shot emitting Arabic numerals against a Chinese-numeral
   reference). Full Chinese-number-reading normalization is out of scope
   for the MVP.

CER definition
--------------
``CER = char_edit_distance(norm(hyp), norm(ref)) / len(norm(ref))``,
clipped to ``[0, 1]`` (insertions can push the raw ratio above 1); the
number of per-utterance clips is tracked and reported (expected << 1% on
Aishell per the design doc).
"""

from __future__ import annotations

import unicodedata
from typing import Any, Dict, List, Literal, Optional

__all__ = [
    "NORMALIZER_VERSION",
    "NumeralPolicy",
    "normalize_text",
    "char_edit_distance",
    "compute_cer",
    "compute_cer_batch",
    "micro_cer",
    "macro_cer",
]

NORMALIZER_VERSION = "asr-gate-cer-normalizer-v1"

NumeralPolicy = Literal["keep", "digits-to-cjk", "cjk-to-digits"]

_DIGIT_TO_CJK = {
    "0": "〇", "1": "一", "2": "二", "3": "三", "4": "四",
    "5": "五", "6": "六", "7": "七", "8": "八", "9": "九",
}
_CJK_TO_DIGIT = {v: k for k, v in _DIGIT_TO_CJK.items()}


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= cp <= 0x4DBF  # CJK Extension A
        or 0xF900 <= cp <= 0xFAFF  # CJK Compatibility Ideographs
        or 0x3040 <= cp <= 0x30FF  # kana (Whisper occasionally emits)
    )


def normalize_text(text: str, numeral_policy: NumeralPolicy = "keep") -> str:
    """Apply the pinned Mandarin CER normalizer. See module docstring."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text)

    if numeral_policy == "digits-to-cjk":
        text = "".join(_DIGIT_TO_CJK.get(ch, ch) for ch in text)
    elif numeral_policy == "cjk-to-digits":
        text = "".join(_CJK_TO_DIGIT.get(ch, ch) for ch in text)
    elif numeral_policy != "keep":
        raise ValueError(f"unknown numeral_policy {numeral_policy!r}")

    kept = []
    for ch in text:
        if ch.isspace():
            continue
        if _is_cjk(ch) or ch.isalnum():
            kept.append(ch)
        # else: punctuation/symbols -- dropped.
    return "".join(kept)


def char_edit_distance(a: str, b: str) -> int:
    """Levenshtein (character-level) edit distance between two strings."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev
    return prev[m]


def compute_cer(
    hyp_text: str,
    ref_text: str,
    numeral_policy: NumeralPolicy = "keep",
) -> Dict[str, Any]:
    """Compute one utterance's CER after normalization, clipped to [0, 1].

    Returns
    -------
    dict
        ``cer`` (clipped), ``cer_raw`` (unclipped), ``clipped`` (bool),
        ``edits`` (int), ``ref_len`` (int, normalized-reference char
        count), ``hyp_norm``, ``ref_norm``.
    """
    hyp_norm = normalize_text(hyp_text, numeral_policy)
    ref_norm = normalize_text(ref_text, numeral_policy)
    ref_len = len(ref_norm)
    if ref_len == 0:
        raise ValueError("compute_cer: normalized reference is empty (cannot compute CER)")
    edits = char_edit_distance(hyp_norm, ref_norm)
    cer_raw = edits / ref_len
    cer = min(cer_raw, 1.0)
    return {
        "cer": cer,
        "cer_raw": cer_raw,
        "clipped": cer_raw > 1.0,
        "edits": edits,
        "ref_len": ref_len,
        "hyp_norm": hyp_norm,
        "ref_norm": ref_norm,
    }


def compute_cer_batch(
    utterances: List[Dict[str, Any]],
    numeral_policy: NumeralPolicy = "keep",
) -> List[Dict[str, Any]]:
    """Compute CER for every utterance carrying a non-null ``ref_text``.

    Utterances without ``ref_text`` are passed through unchanged (no ``cer``
    key added) -- callers must check for the key's presence, never assume
    every row was scored (the no-reference honesty rule, §2.5).
    """
    out = []
    for u in utterances:
        u2 = dict(u)
        if u.get("ref_text") is not None:
            u2.update(compute_cer(u["hyp_text"], u["ref_text"], numeral_policy=numeral_policy))
        out.append(u2)
    return out


def micro_cer(utterances: List[Dict[str, Any]]) -> Optional[float]:
    """Field-standard MICRO CER: total edits / total ref chars, over
    utterances carrying a computed ``edits``/``ref_len``. NOT certified
    (see gate.py/audit.py docstrings for the macro-vs-micro distinction) --
    reported alongside macro CER only as a transparency descriptive.

    Returns ``None`` if no utterance has been CER-scored.
    """
    total_edits = 0
    total_ref_len = 0
    for u in utterances:
        if "edits" in u and "ref_len" in u:
            total_edits += u["edits"]
            total_ref_len += u["ref_len"]
    if total_ref_len == 0:
        return None
    return total_edits / total_ref_len


def macro_cer(utterances: List[Dict[str, Any]]) -> Optional[float]:
    """CERTIFIED statistic: mean of per-utterance (clipped) CER.

    This -- NOT :func:`micro_cer` -- is the quantity G1/G2 bound. See
    module docstring and design §2.3 ("macro vs micro CER, stated once and
    everywhere").
    """
    cers = [u["cer"] for u in utterances if "cer" in u]
    if not cers:
        return None
    return sum(cers) / len(cers)
