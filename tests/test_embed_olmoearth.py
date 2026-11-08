"""Contract tests for the OlmoEarth embedding backbone.

Construction and the band-selection/grid-size contract must hold on the core
install (no torch import, no network). The encoder itself needs the optional
``embed`` extra and downloads weights on first use, so that path is skipped
when the extra is absent.
"""

import subprocess
import sys

import numpy as np
import pytest

from geosae.data.embed.base import S2_L2A_BANDS, build_backbone
from geosae.data.embed.olmoearth import OlmoEarthBackbone


def test_construction_is_core_only():
    """Building the backbone must not import the optional model package or hit the network.

    ``geosae`` itself is torch-based core, so torch is already on the path once
    ``geosae`` is imported; the property worth pinning is that construction does
    not need the ``embed``-extra-only ``olmoearth_pretrain_minimal`` package
    (loaded lazily, only inside :meth:`OlmoEarthBackbone.embed`). Run in a fresh
    subprocess so the check is not polluted by other test modules that may have
    already imported it in the shared session.
    """
    script = (
        "import geosae.data.embed.olmoearth as m; m.OlmoEarthBackbone(variant='nano'); "
        "import sys; print('olmoearth_pretrain_minimal' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False", "constructing the backbone imported olmoearth_pretrain_minimal"


def test_band_order_is_a_subset_permutation():
    backbone = OlmoEarthBackbone()
    assert set(backbone.band_order) <= set(S2_L2A_BANDS)
    assert len(set(backbone.band_order)) == len(backbone.band_order)


def test_select_bands_reorders_columns():
    backbone = OlmoEarthBackbone()
    # Value at each canonical-band position equals that band's index, so the
    # reordered output must equal the expected permutation of indices.
    reflectance = np.zeros((1, 2, 2, len(S2_L2A_BANDS)), dtype=np.float32)
    for i in range(len(S2_L2A_BANDS)):
        reflectance[..., i] = i
    selected = backbone.select_bands(reflectance)
    expected = [S2_L2A_BANDS.index(b) for b in backbone.band_order]
    assert selected.shape[-1] == len(backbone.band_order)
    for j, exp in enumerate(expected):
        assert np.all(selected[..., j] == exp)


def test_output_grid_matches_stride_below_cap():
    backbone = OlmoEarthBackbone(patch_size=4)
    assert backbone.stride == 4
    assert backbone.output_grid(32) == 8


def test_output_grid_rejects_non_multiple():
    backbone = OlmoEarthBackbone(patch_size=4)
    with pytest.raises(ValueError):
        backbone.output_grid(65)


def test_output_grid_refuses_a_grid_beyond_the_pretrained_range():
    """OlmoEarth v1 pretraining crops covered at most 12 tokens/side (arXiv:2511.13655,
    Sec. 3.1). Beyond that the position encoding is out of distribution, so the
    backbone must refuse rather than return tokens that look valid and are not.
    """
    with pytest.raises(ValueError, match="pretrained on"):
        OlmoEarthBackbone(patch_size=4).output_grid(64)
    with pytest.raises(ValueError, match="pretrained on"):
        OlmoEarthBackbone(patch_size=8).output_grid(128)


def test_output_grid_accepts_the_pretrained_range_and_names_a_reachable_fix():
    assert OlmoEarthBackbone(patch_size=8).output_grid(96) == 12  # exactly at the budget
    # The advice must be a reachable configuration: patch_size is capped at 8 px,
    # so the remedy is a smaller input extent, never a coarser patch size.
    with pytest.raises(ValueError, match=r"Reduce the input extent to at most 96 px"):
        OlmoEarthBackbone(patch_size=8).output_grid(128)
    with pytest.raises(ValueError, match=r"patch_size cannot be coarsened"):
        OlmoEarthBackbone(patch_size=4).output_grid(128)


def test_embed_refuses_an_oversized_patch_before_loading_weights():
    """The guard must fire on shape alone, so it costs no download."""
    backbone = OlmoEarthBackbone(patch_size=4)
    with pytest.raises(ValueError, match="pretrained on"):
        backbone.embed(np.zeros((1, 128, 128, 12), dtype=np.float32))


def test_latent_dim_and_stride_are_positive_ints():
    for variant, latent_dim in [("nano", 128), ("tiny", 192), ("base", 768), ("large", 1024)]:
        backbone = OlmoEarthBackbone(variant=variant, patch_size=4)
        assert backbone.latent_dim == latent_dim
        assert isinstance(backbone.latent_dim, int) and backbone.latent_dim > 0
        assert isinstance(backbone.stride, int) and backbone.stride > 0


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        OlmoEarthBackbone(variant="huge")


def test_build_backbone_returns_olmoearth_instance():
    backbone = build_backbone("olmoearth")
    assert isinstance(backbone, OlmoEarthBackbone)


@pytest.mark.skip(reason="downloads weights and runs a real forward pass; run manually with the embed extra installed")
def test_embed_produces_finite_spatial_token_grid():
    """Slow, network-dependent: encodes one batch on CPU and checks shape/finiteness."""
    pytest.importorskip("olmoearth_pretrain_minimal")
    backbone = OlmoEarthBackbone(variant="nano", patch_size=4, device="cpu")
    rng = np.random.default_rng(0)
    reflectance = rng.random((2, 64, 64, len(S2_L2A_BANDS))).astype(np.float32)
    out = backbone.embed(reflectance)
    grid = backbone.output_grid(64)
    assert out.shape == (2, grid, grid, backbone.latent_dim)
    assert np.isfinite(out).all()


@pytest.mark.skip(reason="downloads weights and runs a real forward pass; run manually with the embed extra installed")
def test_embed_places_a_spatial_signal_at_the_right_token_without_flip_or_transpose():
    """The token grid keeps the spatial layout of the patch (no corner artifact).

    A synthetic patch with a bright quadrant confined to the top-right (small
    row index, large column index in pixel space) must show elevated per-token
    deviation from the grand-mean token in the matching top-right *token* block,
    with no row/col flip or transpose introduced by the ``(B, H', W', T, S, D)``
    -> ``(B, H', W', D)`` reshape in :meth:`OlmoEarthBackbone.embed`. This is a
    spatial-correspondence check, independent of the grid-size capping tested
    above by :func:`test_output_grid_caps_at_trained_grid_budget`.
    """
    pytest.importorskip("olmoearth_pretrain_minimal")
    backbone = OlmoEarthBackbone(variant="base", patch_size=8, device="cpu")
    rng = np.random.default_rng(0)
    height = width = 64
    base = 0.15 + 0.02 * rng.standard_normal((1, height, width, len(S2_L2A_BANDS))).astype(np.float32)
    signal = base.copy()
    # Bright quadrant: rows < height/2 (top half), cols >= width/2 (right half).
    signal[:, : height // 2, width // 2 :, :] += 0.3
    signal = np.clip(signal, 0.0, 1.0)

    out = backbone.embed(signal)[0]  # (grid, grid, latent_dim)
    grid = out.shape[0]
    deviation = np.linalg.norm(out - out.mean(axis=(0, 1), keepdims=True), axis=-1)
    top_right = deviation[: grid // 2, grid // 2 :]
    elsewhere_mask = np.ones_like(deviation, dtype=bool)
    elsewhere_mask[: grid // 2, grid // 2 :] = False
    assert top_right.mean() > deviation[elsewhere_mask].mean() * 1.2
