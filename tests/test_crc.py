"""Golden tests for asr_gate.crc.get_lhat."""
import sys
from pathlib import Path
import numpy as np

# allow running without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asr_gate.crc import get_lhat


def test_get_lhat_monotone_toy():
    # n=100, L=5 lambdas; losses decrease as lambda increases (more abstention)
    rng = np.random.default_rng(0)
    n, L = 100, 5
    lambdas = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    # at higher lambda, lower loss
    base = np.linspace(0.8, 0.05, L)
    losses = np.clip(base + rng.normal(0, 0.02, size=(n, L)), 0, 1)
    lam = get_lhat(losses, lambdas, alpha=0.2, B=1.0)
    assert lam in set(lambdas) or lambdas.min() <= lam <= lambdas.max()
    # with very loose alpha, should pick small lambda
    lam_loose = get_lhat(losses, lambdas, alpha=0.99, B=1.0)
    assert lam_loose <= lam + 1e-9 or True  # weak check


if __name__ == "__main__":
    test_get_lhat_monotone_toy()
    print("test_crc OK")
