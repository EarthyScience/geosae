"""Contract tests for the backbone interface and the store-embedding loop."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from geosae.data.embed import S2_L2A_BANDS, EmbeddingBackbone, build_backbone, embed_store, embedding_schema


class MeanPoolBackbone(EmbeddingBackbone):
    """Reference backbone: block-mean of the selected bands, padded to ``latent_dim``."""

    def __init__(self, stride: int = 4, latent_dim: int = 5, band_order=("B04", "B03", "B02")):
        self.stride = stride
        self.latent_dim = latent_dim
        self.band_order = tuple(band_order)

    def embed(self, reflectance: np.ndarray) -> np.ndarray:
        x = self.select_bands(reflectance)
        b, h, w, c = x.shape
        s = self.stride
        blocks = x.reshape(b, h // s, s, w // s, s, c).mean(axis=(2, 4))
        out = np.zeros((b, h // s, w // s, self.latent_dim), dtype=np.float32)
        out[..., :c] = blocks
        return out


def test_select_bands_reorders_into_backbone_order():
    backbone = MeanPoolBackbone(band_order=("B08", "B02", "B12"))
    # Each band plane carries its own index, so the output must be the permutation.
    x = np.broadcast_to(np.arange(len(S2_L2A_BANDS), dtype=np.float32), (2, 8, 8, len(S2_L2A_BANDS))).copy()
    out = backbone.select_bands(x)
    assert out.shape == (2, 8, 8, 3)
    expected = [S2_L2A_BANDS.index(b) for b in ("B08", "B02", "B12")]
    np.testing.assert_array_equal(out[0, 0, 0], expected)


def test_select_bands_rejects_wrong_band_count_and_unknown_band():
    backbone = MeanPoolBackbone()
    with pytest.raises(ValueError, match="bands on the last axis"):
        backbone.select_bands(np.zeros((1, 4, 4, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="unknown Sentinel-2 bands"):
        MeanPoolBackbone(band_order=("B99",)).select_bands(np.zeros((1, 4, 4, 12), dtype=np.float32))


def test_output_grid_divides_by_stride_and_rejects_remainder():
    backbone = MeanPoolBackbone(stride=32)
    assert backbone.output_grid(128) == 4
    with pytest.raises(ValueError, match="not a multiple of stride"):
        backbone.output_grid(100)


def test_build_backbone_rejects_unknown_name():
    with pytest.raises(KeyError, match="unknown backbone"):
        build_backbone("not_a_model")


def test_embedding_schema_matches_token_grid():
    schema = embedding_schema("olmoearth", n_cells=100, grid=16, dim=768)
    assert schema.shape == (100, 16, 16, 768)
    assert schema.chunks[1:] == (16, 16, 768)
    assert 1 <= schema.chunks[0] <= 100
    assert schema.dim_names == ("cell", "y", "x", "feature_olmoearth")


def _reflectance_store(tmp_path, n_cells=6, patch=8, fill=-32768):
    group = zarr.open_group(tmp_path / "s.zarr", mode="w")
    group.create_array("s2", shape=(n_cells, patch, patch, 12), chunks=(2, patch, patch, 12), dtype="int16")
    group.create_array("mask_s2", shape=(n_cells,), chunks=(n_cells,), dtype="bool")
    rng = np.random.default_rng(0)
    group["s2"][:] = rng.integers(1, 8000, size=(n_cells, patch, patch, 12), dtype=np.int16)
    mask = np.ones(n_cells, dtype=bool)
    mask[-1] = False  # one cell the fetcher failed on
    group["mask_s2"][:] = mask
    group["s2"][2, 0, 0, :] = fill  # a few pixels with no observation
    return group, mask


def test_embed_store_writes_token_grid_for_valid_cells_only(tmp_path):
    group, mask = _reflectance_store(tmp_path)
    backbone = MeanPoolBackbone(stride=4, latent_dim=5)

    embed_store(group, backbone, "toy", batch_cells=4)

    out = np.asarray(group["toy"][:])
    assert out.shape == (6, 2, 2, 5)
    np.testing.assert_array_equal(np.asarray(group["mask_toy"][:]), mask)
    assert np.isfinite(out[mask]).all()
    assert np.isnan(out[~mask]).all()
    np.testing.assert_array_equal(np.asarray(group["feature_toy"][:]), np.arange(5))


def test_embed_store_matches_a_direct_call_on_the_same_reflectance(tmp_path):
    group, mask = _reflectance_store(tmp_path)
    backbone = MeanPoolBackbone(stride=4, latent_dim=5)
    embed_store(group, backbone, "toy", batch_cells=2)

    rows = np.where(mask)[0]
    dn = np.asarray(group["s2"].oindex[rows]).astype(np.float32)
    dn[dn == -32768] = 0.0
    expected = backbone.embed(dn / 10_000.0)
    np.testing.assert_allclose(np.asarray(group["toy"].oindex[rows]), expected, rtol=0, atol=1e-3)


def test_embed_store_requires_the_source_array(tmp_path):
    group, _ = _reflectance_store(tmp_path)
    with pytest.raises(KeyError, match="no array 'missing'"):
        embed_store(group, MeanPoolBackbone(), "toy", source_array="missing")


def test_center_crop_takes_the_middle_and_rejects_bad_sizes():
    import numpy as np

    from geosae.data.embed import center_crop

    x = np.arange(1 * 8 * 8 * 2, dtype=np.float32).reshape(1, 8, 8, 2)
    out = center_crop(x, 4)
    assert out.shape == (1, 4, 4, 2)
    np.testing.assert_array_equal(out, x[:, 2:6, 2:6, :])
    with pytest.raises(ValueError, match="exceeds"):
        center_crop(x, 12)
    with pytest.raises(ValueError, match="pixel-centred"):
        center_crop(x, 5)


def test_embed_store_crop_shrinks_the_grid_and_records_the_footprint(tmp_path):
    group, mask = _reflectance_store(tmp_path, patch=8)
    backbone = MeanPoolBackbone(stride=2, latent_dim=3)

    embed_store(group, backbone, "toy", batch_cells=4, crop_px=4)

    out = np.asarray(group["toy"][:])
    assert out.shape == (6, 2, 2, 3)  # 4 px encoded at stride 2, not 8 px
    assert group["toy"].attrs["crop_px"] == 4
    # The tokens must come from the centre of the patch, not its corner.
    rows = np.where(mask)[0]
    dn = np.asarray(group["s2"].oindex[rows]).astype(np.float32)
    dn[dn == -32768] = 0.0
    from geosae.data.embed import center_crop

    expected = backbone.embed(center_crop(dn / 10_000.0, 4))
    np.testing.assert_allclose(np.asarray(group["toy"].oindex[rows]), expected, rtol=0, atol=1e-3)
