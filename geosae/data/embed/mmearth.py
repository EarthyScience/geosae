"""MMEarth (Nedungadi et al., ECCV 2024) ConvNeXt V2 encoder as an embedding backbone."""

from __future__ import annotations

import re

import numpy as np

from geosae.data.embed.base import S2_REFLECTANCE_SCALE, EmbeddingBackbone

#: MMEarth's Sentinel-2 input-modality band order (`INP_MODALITIES["sentinel2"]` in
#: https://github.com/vishalned/MMEarth-train/blob/main/MODALITIES.py), mapped onto this
#: project's zero-padded :data:`~geosae.data.embed.base.S2_L2A_BANDS` spelling. Note that
#: B8A precedes B8 -- this is MMEarth's own (non-standard) ordering, not a typo.
_MMEARTH_BAND_ORDER: tuple[str, ...] = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B08",
    "B09",
    "B11",
    "B12",
)

#: Width of the final ConvNeXt V2 stage (`dims[-1]` in the `convnextv2_<variant>` factory
#: functions of https://github.com/vishalned/MMEarth-train/blob/main/models/convnextv2.py),
#: i.e. the token width returned by :meth:`MMEarthBackbone.embed`.
_VARIANT_LATENT_DIM: dict[str, int] = {"atto": 320, "tiny": 768}

#: Trailing `_<img_size>-<patch_size>` suffix MMEarth checkpoint names encode, e.g. the
#: default checkpoint `pt-all_mod_atto_1M_64_uncertainty_56-8` was trained at `img_size=56`
#: with `patch_size=8`.
_CHECKPOINT_GEOMETRY_RE = re.compile(r"_(\d+)-(\d+)$")

#: Official per-band Sentinel-2 L2A statistics for MMEarth's 1.2M-tile, 64x64 px release
#: ("MMEarth64", file `data_1M_v001_64.h5`) -- the dataset the default checkpoint,
#: `pt-all_mod_atto_1M_64_uncertainty_56-8`, was pretrained on. Source:
#: https://sid.erda.dk/share_redirect/bX5JzPuwJF/data_1M_v001_64_band_stats.json (the direct
#: download linked from the "MMEarth64" row of
#: https://github.com/vishalned/MMEarth-data#data-download), key `sentinel2_l2a`. That file
#: stores one `(mean, std)` pair per band in ascending raw order (`B01..B12`, 13 entries
#: including `B10`); `B10` is NaN/Inf there because Sentinel-2 L2A does not carry it, so it is
#: dropped here. Values are on the raw Sentinel-2 digital-number scale MMEarth trained on, not
#: `[0, ~1]` reflectance -- see :meth:`MMEarthBackbone.embed`.
_MMEARTH64_L2A_RAW_BAND_ORDER: tuple[str, ...] = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
)
_MMEARTH64_L2A_MEAN: tuple[float, ...] = (
    1349.3977794889083,
    1479.9521800379623,
    1720.3688077425966,
    1899.1848715975957,
    2253.9309600057886,
    2808.2001963620132,
    3003.424149045887,
    3149.5364927329806,
    3110.840562275062,
    3213.7636154015954,
    float("nan"),
    2399.086213373806,
    1811.7986415136786,
)
_MMEARTH64_L2A_STD: tuple[float, ...] = (
    2340.2916479338087,
    2375.872101251672,
    2256.8997709659416,
    2354.181051828758,
    2292.99569489449,
    2033.2166835293804,
    1920.1736418230105,
    1816.6152354201365,
    1988.1938283738782,
    1947.9031620588928,
    float("nan"),
    1473.224812450967,
    1390.6781165633136,
)

#: Official ``{checkpoint_name: {band_name: (mean, std)}}`` registry, keyed by exact MMEarth
#: checkpoint name. Only checkpoints this module has verified official statistics for are
#: listed; anything else needs an explicit ``band_stats`` argument.
_OFFICIAL_BAND_STATS: dict[str, dict[str, tuple[float, float]]] = {
    "pt-all_mod_atto_1M_64_uncertainty_56-8": {
        band: (mean, std)
        for band, mean, std in zip(_MMEARTH64_L2A_RAW_BAND_ORDER, _MMEARTH64_L2A_MEAN, _MMEARTH64_L2A_STD, strict=True)
        if band != "B10"
    }
}


def _parse_checkpoint_geometry(checkpoint: str) -> tuple[int, int]:
    """Extract ``(img_size, patch_size)`` from a `<img_size>-<patch_size>` checkpoint suffix.

    Args:
        checkpoint: MMEarth checkpoint folder name, e.g.
            ``"pt-all_mod_atto_1M_64_uncertainty_56-8"``.

    Returns:
        ``(img_size, patch_size)`` the checkpoint was pretrained with.

    Raises:
        ValueError: If ``checkpoint`` has no `_<img_size>-<patch_size>` suffix.
    """
    match = _CHECKPOINT_GEOMETRY_RE.search(checkpoint)
    if match is None:
        raise ValueError(f"cannot parse img_size/patch_size from checkpoint name {checkpoint!r}")
    return int(match.group(1)), int(match.group(2))


class MMEarthBackbone(EmbeddingBackbone):
    """ConvNeXt V2 encoder pretrained on MMEarth (Multi-Pretext Masked Autoencoder).

    Loads a `vishalned/mmearth-train <https://github.com/vishalned/MMEarth-train>`_ checkpoint
    via ``torch.hub`` and returns the final convolutional stage's feature map -- i.e. the
    encoder's spatial token grid, *before* the global-average-pool ``forward_features``
    applies. ``forward_features`` is unsuitable for a per-pixel SAE because it collapses the token grid
    to one pooled vector per patch; this backbone instead replays the same stem/stage/downsample
    submodules and stops one step earlier.

    The encoder is fully convolutional, so it runs on any input size, not only the 56x56 (or
    112x112) crop it was pretrained on. Its stem starts with an un-padded 3x3 convolution that
    shrinks the input by 2 px; for checkpoints whose patch size is 8 (the default), the
    downstream stem convolution has kernel size 1 and cannot re-absorb that shrinkage, so this
    backbone reflect-pads the input by 1 px on each spatial side first -- this exactly cancels
    the shrinkage and restores a clean, patch-size-multiple token grid (verified: 64 px -> 8x8,
    128 px -> 16x16 tokens for the default checkpoint). Checkpoints with a larger patch size
    already reabsorb the shrinkage through their own stem padding and skip this step.

    Attributes:
        band_order: MMEarth's Sentinel-2 band order (B8A precedes B8), spelled with this
            project's zero-padded band names.
        latent_dim: Width of the final ConvNeXt V2 stage (320 for atto, 768 for tiny).
        stride: Input pixels per output token, equal to the checkpoint's patch size.

    Example:
        >>> backbone = MMEarthBackbone(device="cpu")
        >>> tokens = backbone.embed(reflectance)  # (B, H // 8, W // 8, 320)
    """

    def __init__(
        self,
        variant: str = "atto",
        checkpoint: str = "pt-all_mod_atto_1M_64_uncertainty_56-8",
        device: str = "cuda",
        batch_size: int | None = None,
        cache_dir: str | None = None,
        band_stats: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Configure the backbone. No weights are downloaded or loaded until :meth:`embed`.

        Args:
            variant: ConvNeXt V2 width, ``"atto"`` or ``"tiny"``.
            checkpoint: MMEarth checkpoint folder name (see the model table at
                https://github.com/vishalned/MMEarth-train#pre-trained-models).
            device: Torch device string used to run the model.
            batch_size: If set, ``embed`` runs the model in chunks of this many patches
                instead of one call over the whole input batch.
            cache_dir: Directory ``torch.hub`` downloads and caches the checkpoint in.
                Defaults to ``torch.hub``'s own cache (``~/.cache/torch/hub``).
            band_stats: Optional ``{band_name: (mean, std)}`` override, on the raw Sentinel-2
                digital-number scale, used in place of this module's bundled official
                statistics (see :data:`_OFFICIAL_BAND_STATS`). Required for any checkpoint this
                module does not carry official statistics for.

        Raises:
            ValueError: If ``variant`` is not recognized, or if ``band_stats`` is ``None`` and
                no official statistics are bundled for ``checkpoint``, or if the resolved
                statistics do not cover every band in :attr:`band_order`.
        """
        if variant not in _VARIANT_LATENT_DIM:
            raise ValueError(f"unknown MMEarth variant {variant!r}; expected one of {sorted(_VARIANT_LATENT_DIM)}")
        self.variant = variant
        self.checkpoint = checkpoint
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = cache_dir

        self.band_order = _MMEARTH_BAND_ORDER
        self.latent_dim = _VARIANT_LATENT_DIM[variant]
        _, patch_size = _parse_checkpoint_geometry(checkpoint)
        self.stride = patch_size

        if band_stats is None:
            band_stats = _OFFICIAL_BAND_STATS.get(checkpoint)
            if band_stats is None:
                raise ValueError(
                    f"no official band statistics are bundled for checkpoint {checkpoint!r}; "
                    "pass `band_stats` explicitly as a {band_name: (mean, std)} mapping on the "
                    "raw Sentinel-2 digital-number scale (e.g. computed over your own patches)."
                )
        missing = set(self.band_order) - set(band_stats)
        if missing:
            raise ValueError(f"band_stats is missing bands {sorted(missing)} required by band_order")
        self._mean = np.array([band_stats[b][0] for b in self.band_order], dtype=np.float32)
        self._std = np.array([band_stats[b][1] for b in self.band_order], dtype=np.float32)

        self._model = None
        self._pad_edge = patch_size // 8 <= 1

    def _load_model(self):
        """Download (if needed), build and cache the ``torch.hub`` MMEarth encoder."""
        if self._model is not None:
            return self._model
        import torch

        if self.cache_dir is not None:
            torch.hub.set_dir(self.cache_dir)
        img_size, patch_size = _parse_checkpoint_geometry(self.checkpoint)
        model = torch.hub.load(
            "vishalned/mmearth-train",
            "MPMAE",
            model_name=f"convnextv2_{self.variant}",
            ckpt_name=self.checkpoint,
            pretrained=True,
            linear_probe=True,
            num_classes=1,
            in_chans=len(self.band_order),
            patch_size=patch_size,
            img_size=img_size,
            trust_repo=True,
        )
        model.eval()
        model.to(self.device)
        self._model = model
        return model

    def _pre_pool_features(self, model, x):
        """Replay ``model``'s stem/stage/downsample path, stopping before the final pool.

        Mirrors ``ConvNeXtV2.forward_features`` (see the module docstring) up to but excluding
        its trailing ``self.norm(x.mean([-2, -1]))`` line, using only the loaded model's own
        public submodules.
        """
        import torch.nn.functional as F

        if self._pad_edge:
            x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        if model.use_orig_stem:
            x = model.stem_orig(x)
        else:
            x = model.initial_conv(x)
            x = model.stem(x)
        x = model.stages[0](x)
        for i in range(3):
            x = model.downsample_layers[i](x)
            x = model.stages[i + 1](x)
        return x

    def embed(self, reflectance: np.ndarray) -> np.ndarray:
        """Encode a batch of reflectance patches into a spatial token grid.

        Args:
            reflectance: ``(B, H, W, 12)`` float32 surface reflectance in ``[0, ~1]``, bands
                ordered as :data:`~geosae.data.embed.base.S2_L2A_BANDS`.

        Returns:
            ``(B, H // stride, W // stride, latent_dim)`` float32 token grid.
        """
        import torch

        model = self._load_model()
        x = self.select_bands(reflectance.astype(np.float32, copy=False))
        x = (x * S2_REFLECTANCE_SCALE - self._mean) / self._std
        x = np.transpose(x, (0, 3, 1, 2))  # (B, H, W, C) -> (B, C, H, W)

        batch_size = self.batch_size or x.shape[0]
        chunks = []
        with torch.inference_mode():
            for start in range(0, x.shape[0], batch_size):
                chunk = torch.from_numpy(x[start : start + batch_size]).to(self.device, dtype=torch.float32)
                feat = self._pre_pool_features(model, chunk)  # (n, C, H', W')
                chunks.append(feat.permute(0, 2, 3, 1).contiguous().cpu().numpy())
        return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


__all__ = ["MMEarthBackbone"]
