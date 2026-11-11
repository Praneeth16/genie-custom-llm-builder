import numpy as np
import torch

from jepa_wm.data.synthetic_sensor import generate, windows
from jepa_wm.sensor_jepa import SensorMaskedModel
from jepa_wm.sigreg import sigreg


def test_sigreg_about_one_for_gaussian_large_for_collapse():
    torch.manual_seed(0)
    gaussian = sigreg(torch.randn(2048, 32), seed=0)
    collapsed = sigreg(torch.randn(2048, 1).repeat(1, 32) * 0.01, seed=0)
    low_rank = sigreg(torch.randn(2048, 2) @ torch.randn(2, 32), seed=0)
    assert 0.7 < gaussian < 1.6  # null expectation is about 1.06
    assert collapsed > 100 * gaussian
    assert low_rank > 50 * gaussian


def test_sigreg_same_seed_same_directions():
    z = torch.randn(256, 16)
    assert sigreg(z, seed=3) == sigreg(z, seed=3)
    assert sigreg(z, seed=3) != sigreg(z, seed=4)


def test_generate_is_clean_without_anomalies_and_labelled_with():
    series, labels = generate(20_000, seed=1)
    assert series.shape == (20_000, 8) and not labels.any()
    noisy, noisy_labels = generate(20_000, anomaly_fraction=0.05, seed=1)
    assert 0.05 <= noisy_labels.mean() < 0.07
    # the two series share physics and noise, so they differ only where labelled
    assert np.array_equal(noisy[~noisy_labels], series[~noisy_labels])
    assert not np.array_equal(noisy[noisy_labels], series[noisy_labels])


def test_windows_label_any_anomaly_inside():
    series = np.zeros((64, 2), dtype=np.float32)
    labels = np.zeros(64, dtype=bool)
    labels[40] = True
    x, y = windows(series, labels, window=32, stride=16)
    assert x.shape == (3, 32, 2)
    assert y.tolist() == [False, True, True]


def test_block_masks_tile_the_window():
    model = SensorMaskedModel(n_channels=4, window=64, patch=8, dim=32, depth=1, predictor_depth=1)
    masks = model.block_masks()
    assert masks.shape == (4, 8)
    assert masks.sum(0).tolist() == [1] * 8


def test_loss_and_score_shapes_for_both_targets():
    x = torch.randn(16, 64, 4)
    for target in ("latent", "input"):
        model = SensorMaskedModel(n_channels=4, window=64, patch=8, dim=32, depth=1, predictor_depth=1, target=target)
        loss, parts = model.loss(x, model.block_masks()[1], seed=0)
        loss.backward()
        assert ("sigreg" in parts) == (target == "latent")
        assert model.score(x).shape == model.score(x, "prediction").shape == (16,)
        assert model.embed(x).shape == (16, 32)


def test_dynamics_anomalies_stay_in_the_normal_value_range():
    clean, _ = generate(20_000, seed=1)
    series, labels = generate(20_000, anomaly_fraction=0.05, family="dynamics", seed=1)
    assert 0.05 <= labels.mean() < 0.07
    assert np.array_equal(series[~labels], clean[~labels])
    assert not np.array_equal(series[labels], clean[labels])
    assert np.abs(series[labels]).max() <= np.abs(clean).max() + 0.2


def test_density_score_flags_windows_unlike_the_fitted_data():
    torch.manual_seed(0)
    model = SensorMaskedModel(n_channels=4, window=64, patch=8, dim=32, depth=1, predictor_depth=1)
    normal = torch.randn(512, 64, 4)
    model.fit_density(normal)
    assert model.score(torch.randn(64, 64, 4) * 5).median() > 3 * model.score(torch.randn(64, 64, 4)).median()
