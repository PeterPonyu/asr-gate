#!/usr/bin/env python3
"""Minimal regression smoke for run_contrast helpers."""
import numpy as np

from run_contrast import ALPHAS, crc_lambda, debyte, risk_at


def test_debyte():
    assert debyte("hello") == "hello"
    assert debyte("a<0xE4><0xB8><0xAD>") == "a中"


def test_crc_lambda_basic():
    scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    losses = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    lam = crc_lambda(scores, losses, 0.5)
    assert np.isfinite(lam)
    assert risk_at(scores, losses, lam) <= 0.5 + 1e-9


def test_empty_hyp_polarity_constants():
    assert ALPHAS == (0.05, 0.10, 0.20)


if __name__ == "__main__":
    test_debyte()
    test_crc_lambda_basic()
    test_empty_hyp_polarity_constants()
    print("smoke ok")
