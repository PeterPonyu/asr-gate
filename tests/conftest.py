"""Synthetic decode fixtures shared across asr-gate tests.

No network, no GPU: everything is generated in-process with a controllable
correlation between confidence score and true CER, so certificates and
audits have KNOWN ground truth to check against.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

_VOCAB = list("今天天气很好我们明去学校上课看书写字读音乐听音听风雨云山水火土人心手口目日月年岁月光影声色香味道路车站城市")


def _random_text(rng: np.random.Generator, length: int) -> str:
    idx = rng.integers(0, len(_VOCAB), size=length)
    return "".join(_VOCAB[i] for i in idx)


def _corrupt(rng: np.random.Generator, text: str, p_err: float) -> str:
    chars = list(text)
    for i in range(len(chars)):
        if rng.random() < p_err:
            chars[i] = _VOCAB[rng.integers(0, len(_VOCAB))]
    return "".join(chars)


def make_synthetic_utterances(
    n: int = 2000,
    n_speakers: int = 40,
    seed: int = 0,
    correlation_strength: float = 1.0,
    degraded_frac: float = 0.1,
    with_refs: bool = True,
) -> List[Dict[str, Any]]:
    """Build a canonical utterance table with a KNOWN score<->CER relationship.

    ``s1``/``s2``/``s4`` (derived from ``token_logps``) are driven by a
    latent per-utterance "quality" ``q`` that ALSO drives the true
    character-substitution error rate used to build ``hyp_text`` from
    ``ref_text`` -- so at ``correlation_strength = 1.0`` those scores are
    genuinely informative about true CER. ``s3`` (N-best margin) is driven
    by an INDEPENDENT random second-best logp gap, uncorrelated with true
    CER by construction -- a built-in "noise" comparator (matches
    ``ope-audit``'s good-vs-noise pattern) for the audit e2e assertions.
    ``correlation_strength`` in ``[0, 1]`` interpolates the error rate
    between fully quality-driven (1.0) and quality-independent random noise
    (0.0), for tests that need a genuinely uninformative s1/s2/s4 too.

    Speakers are assigned round-robin (``n // n_speakers`` utterances each)
    so cal/tune speaker-disjoint splits are always feasible with either
    pool getting a non-trivial share.
    """
    rng = np.random.default_rng(seed)
    speaker_ids = [f"SPK{s:03d}" for s in range(n_speakers)]
    genders = {sp: ("M" if i % 2 == 0 else "F") for i, sp in enumerate(speaker_ids)}

    utterances: List[Dict[str, Any]] = []
    for i in range(n):
        q = float(rng.uniform(0.0, 1.0))
        p_err_quality = 0.30 * (1.0 - q)
        p_err_random = float(rng.uniform(0.0, 0.30))
        p_err = correlation_strength * p_err_quality + (1.0 - correlation_strength) * p_err_random

        ref_len = int(rng.integers(8, 15))
        ref_text = _random_text(rng, ref_len)
        hyp_text = _corrupt(rng, ref_text, p_err) if with_refs else _random_text(rng, ref_len)

        mean_logp = -0.05 - 3.0 * (1.0 - q)
        token_logps = np.minimum(
            mean_logp + rng.normal(0.0, 0.3, size=max(len(hyp_text), 1)), 0.0
        ).tolist()
        logp1 = float(np.sum(token_logps))

        nbest = [{"text": hyp_text, "logp": logp1, "token_logps": token_logps}]
        if rng.random() > degraded_frac:
            hyp2 = _corrupt(rng, ref_text, min(p_err + 0.15, 0.9))
            logp2 = logp1 - abs(float(rng.normal(2.0, 1.0)))  # independent of q -> s3 is noise
            nbest.append({"text": hyp2, "logp": logp2, "token_logps": None})

        speaker_id = speaker_ids[i % n_speakers]
        utterances.append(
            {
                "utt_id": f"UTT{i:05d}",
                "speaker_id": speaker_id,
                "duration_s": float(rng.uniform(2.0, 10.0)),
                "hyp_text": hyp_text,
                "nbest": nbest,
                "ref_text": ref_text if with_refs else None,
                "gender": genders[speaker_id],
                "region": None,
            }
        )
    return utterances
