"""Contract tests for the MMEarth embedding backbone.

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
from geosae.data.embed.mmearth import MMEarthBackbone


def test_construction_is_core_only():
    """Building the backbone must not import the optional model package or hit the network.

    ``geosae`` itself is torch-based core, so torch (and its always-imported
    ``torch.hub`` submodule) is already on the path once ``geosae`` is
    imported; the property worth pinning is that construction does not need
    the ``embed``-extra-only ``timm`` package, loaded lazily only inside
    :meth:`MMEarthBackbone.embed`. Run in a fresh subprocess so the check is
    not polluted by other test modules that may already have imported it.
    """
    script = (
        "import geosae.data.embed.mmearth as m; m.MMEarthBackbone(device='cpu'); "
        "import sys; print('timm' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False", "constructing the backbone imported timm"


def test_band_order_is_a_subset_permutation():
    backbone = MMEarthBackbone(device="cpu")
    assert set(backbone.band_order) <= set(S2_L2A_BANDS)
    assert len(set(backbone.band_order)) == len(backbone.band_order)


def test_band_order_places_b8a_before_b8():
    # MMEarth's own (non-standard) Sentinel-2 ordering -- easy to get silently wrong.
    backbone = MMEarthBackbone(device="cpu")
    assert backbone.band_order.index("B8A") < backbone.band_order.index("B08")


def test_select_bands_reorders_columns():
    backbone = MMEarthBackbone(device="cpu")
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


def test_output_grid_matches_stride():
    backbone = MMEarthBackbone(device="cpu")
    assert backbone.stride == 8
    assert backbone.output_grid(128) == 128 // backbone.stride


def test_output_grid_rejects_non_multiple():
    backbone = MMEarthBackbone(device="cpu")
    with pytest.raises(ValueError):
        backbone.output_grid(65)


def test_latent_dim_and_stride_are_positive_ints():
    for variant, latent_dim in [("atto", 320), ("tiny", 768)]:
        backbone = MMEarthBackbone(variant=variant, device="cpu")
        assert backbone.latent_dim == latent_dim
        assert isinstance(backbone.latent_dim, int) and backbone.latent_dim > 0
        assert isinstance(backbone.stride, int) and backbone.stride > 0


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        MMEarthBackbone(variant="huge", device="cpu")


def test_missing_band_stats_raises_documented_error():
    with pytest.raises(ValueError, match="band_stats"):
        MMEarthBackbone(checkpoint="pt-S2_atto_1M_64_uncertainty_56-8", device="cpu")


def test_incomplete_band_stats_raises():
    with pytest.raises(ValueError):
        MMEarthBackbone(device="cpu", band_stats={"B01": (0.0, 1.0)})


def test_build_backbone_returns_mmearth_instance():
    backbone = build_backbone("mmearth", device="cpu")
    assert isinstance(backbone, MMEarthBackbone)


@pytest.mark.skip(reason="downloads weights and runs a real forward pass; run manually with the embed extra installed")
def test_embed_produces_finite_spatial_token_grid():
    """Slow, network-dependent: encodes one batch on CPU and checks shape/finiteness."""
    pytest.importorskip("timm")
    backbone = MMEarthBackbone(device="cpu")
    rng = np.random.default_rng(0)
    reflectance = rng.random((2, 128, 128, len(S2_L2A_BANDS))).astype(np.float32)
    out = backbone.embed(reflectance)
    assert out.shape == (2, 128 // backbone.stride, 128 // backbone.stride, backbone.latent_dim)
    assert np.isfinite(out).all()
