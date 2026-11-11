"""Synthetic multivariate machine telemetry with labelled anomalies.

One hidden load signal drives every channel through a fixed mixing matrix, so channels are
correlated the way sensors on one machine are. The mixing matrix depends only on
`physics_seed`, so different splits (different `seed`) describe the same machine.
"""

from __future__ import annotations

import numpy as np

# "local": one channel is disturbed in a way visible in its own values.
# "dynamics": one channel keeps plausible values but stops agreeing with the others.
ANOMALY_KINDS = {
    "local": ("spike", "drift", "stuck", "decouple", "noise"),
    "dynamics": ("lag", "regime", "coupling"),
}


def generate(
    n_steps: int,
    n_channels: int = 8,
    anomaly_fraction: float = 0.0,
    family: str = "local",
    seed: int = 0,
    physics_seed: int = 1234,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(series, labels)`: float32 (n_steps, n_channels) and bool (n_steps,)."""
    rng = np.random.default_rng(seed)

    # Load: random operating regimes, smoothed so changes take a few hundred steps.
    regimes = rng.uniform(0.5, 1.5, size=n_steps // 400 + 2)
    load = np.repeat(regimes, 400)[:n_steps]
    kernel = np.ones(100) / 100
    load = np.convolve(np.pad(load, 50, mode="edge"), kernel, mode="same")[50:-50]

    phase = np.cumsum(2 * np.pi * 0.01 * (1 + load))
    factors = np.stack(
        [np.sin(phase), np.cos(phase), load - 1, (load - 1) ** 2, np.sin(2 * phase) * load, np.cos(3 * phase)], axis=1
    )
    mixing = np.random.default_rng(physics_seed).normal(size=(n_channels, factors.shape[1]))
    series = np.tanh(factors @ mixing.T) + 0.05 * rng.normal(size=(n_steps, n_channels))
    labels = np.zeros(n_steps, dtype=bool)
    clean = series.copy()
    kinds = ANOMALY_KINDS[family]

    while labels.mean() < anomaly_fraction:
        length = int(rng.integers(20, 80))
        start = int(rng.integers(100, n_steps - length))
        if labels[max(0, start - 100) : start + length + 100].any():
            continue  # keep anomalies apart
        span = slice(start, start + length)
        channel = int(rng.integers(n_channels))
        kind = kinds[int(rng.integers(len(kinds)))]
        if kind == "spike":
            span = slice(start, start + 4)
            series[span, channel] += rng.choice([-1, 1]) * rng.uniform(1.0, 2.0)
        elif kind == "drift":
            series[span, channel] += np.linspace(0, rng.choice([-1, 1]) * rng.uniform(0.5, 1.0), length)
        elif kind == "stuck":
            series[span, channel] = series[start, channel]
        elif kind == "decouple":
            t = np.arange(length)
            series[span, channel] = 0.7 * np.sin(2 * np.pi * rng.uniform(0.03, 0.08) * t + rng.uniform(0, 6.28))
        elif kind == "noise":
            series[span, channel] += 0.3 * rng.normal(size=length)
        elif kind == "lag":
            lag = int(rng.integers(5, 20))
            series[span, channel] = clean[start - lag : start - lag + length, channel]
        elif kind == "regime":
            source = int(rng.integers(0, n_steps - length))
            series[span, channel] = clean[source : source + length, channel]
        else:  # coupling: the channel responds to the machine state through different weights
            weights = rng.normal(size=factors.shape[1])
            series[span, channel] = np.tanh(factors[span] @ weights) + 0.05 * rng.normal(size=length)
        labels[span] = True

    return series.astype(np.float32), labels


def windows(series: np.ndarray, labels: np.ndarray, window: int = 128, stride: int = 16):
    """Slice into `(X, y)`: (N, window, C) windows and whether each contains an anomaly."""
    starts = np.arange(0, len(series) - window + 1, stride)
    index = starts[:, None] + np.arange(window)
    return series[index], labels[index].any(axis=1)
