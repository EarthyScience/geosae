"""Smoke tests of the helpers behind ``scripts/neurips2026`` on tiny synthetic run directories."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "neurips2026"


@pytest.fixture(scope="module")
def common():
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield importlib.import_module("_common")
    finally:
        sys.path.remove(str(SCRIPTS))


def _run(path: Path, model: dict, seed: int) -> Path:
    path.mkdir(parents=True)
    cfg = {"seed": seed, "data": {"zarr_path": "store.zarr"}, "model": {"latent_dim": 1024, "k": 16, **model}}
    (path / "config.yaml").write_text(yaml.safe_dump(cfg))
    (path / "best.pth").write_bytes(b"")
    return path


def test_find_runs_keys_a_sweep_by_variant_size_sparsity_and_seed(common, tmp_path):
    flat = _run(tmp_path / "matryoshka_M1024_k16_s0", {"type": "matryoshka"}, 0)
    _run(tmp_path / "frozen_M1024_k16_s1" / "20260101-000000", {"type": "batchtopk", "freeze_decoder": True}, 1)
    latest = _run(
        tmp_path / "frozen_M1024_k16_s1" / "20260102-000000", {"type": "batchtopk", "freeze_decoder": True}, 1
    )
    (tmp_path / "not_a_run").mkdir()

    runs = common.find_runs(tmp_path)
    assert set(runs) == {common.RunKey("matryoshka", 1024, 16, 0), common.RunKey("frozen", 1024, 16, 1)}
    assert common.get_run(runs, "matryoshka") == flat
    assert common.get_run(runs, "frozen", seed=1) == latest
    with pytest.raises(KeyError, match="relu_l1"):
        common.get_run(runs, "relu_l1")


def test_stage_outputs_are_read_back_and_missing_ones_named(common, tmp_path):
    rows = [{"name": "pca_M16_s0", "fve": 0.7}]
    common.write_json({"rows": rows}, tmp_path / "panel" / "eval" / "panel.json")
    assert common.panel_rows(tmp_path / "panel") == rows
    with pytest.raises(FileNotFoundError, match="reproduce-figures.md"):
        common.read_json(tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="geosae predict"):
        common.predictions(tmp_path)


def test_write_json_accepts_numpy_values(common, tmp_path):
    np = pytest.importorskip("numpy")
    path = common.write_json({"a": np.float32(0.5), "b": np.arange(3)}, tmp_path / "x.json")
    assert json.loads(path.read_text()) == {"a": 0.5, "b": [0, 1, 2]}


def test_style_and_save_write_every_format(common, tmp_path):
    pytest.importorskip("nicefigs")
    import matplotlib.pyplot as plt

    scale = common.set_style(11.0)
    assert scale == pytest.approx(2.0) and common.fs("label") == pytest.approx(15.0)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], **common.variant_style("relu_l1"))
    common.panel_title(ax, "a", "title")
    path = common.save(fig, tmp_path, "fig", formats=("pdf", "png"))
    assert path.exists() and (tmp_path / "fig.png").exists()


@pytest.mark.parametrize("script", sorted(p.stem for p in SCRIPTS.glob("[!_]*.py")))
def test_every_script_imports_without_running(common, script):
    pytest.importorskip("nicefigs")
    module = importlib.import_module(script)
    assert callable(module.main)


def test_true_colour_reads_rgb_else_sentinel2_bands_else_nothing(common):
    np = pytest.importorskip("numpy")
    zarr = pytest.importorskip("zarr")
    g = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
    assert common.true_colour(g, 0) is None
    s2 = np.full((1, 4, 4, 12), 100, dtype="int16")
    s2[..., 3] = 400  # B04, the red band
    g.create_array("s2", data=s2)
    img = common.true_colour(g, 0)
    assert img.shape == (4, 4, 3) and img[..., 0].max() == pytest.approx(1.0) and img[..., 2].max() < 1.0
    g.create_array("rgb", data=np.full((1, 4, 4, 3), 200, dtype="uint8"))
    np.testing.assert_allclose(common.true_colour(g, 0), 1.0)


def test_true_colour_bands_follow_the_sentinel2_fetcher(common):
    pytest.importorskip("odc.geo")
    from geosae.data.download.fetchers.sentinel2 import SENTINEL2_BANDS

    assert [SENTINEL2_BANDS[i] for i in common.S2_TRUE_COLOUR] == ["B04", "B03", "B02"]


def test_paper_file_finds_subfolder_files_in_either_layout(common, tmp_path):
    (tmp_path / "nested" / "products").mkdir(parents=True)
    (tmp_path / "nested" / "products" / "a.json").write_text("{}")
    (tmp_path / "flat").mkdir()
    (tmp_path / "flat" / "products__a.json").write_text("{}")
    assert common.paper_file(tmp_path / "nested", "products/a.json") == tmp_path / "nested" / "products" / "a.json"
    assert common.paper_file(tmp_path / "flat", "products/a.json") == tmp_path / "flat" / "products__a.json"
    with pytest.raises(FileNotFoundError, match="paper-data"):
        common.paper_file(tmp_path / "flat", "products/b.json")


def test_stage_arguments_are_required_unless_paper_data_is_given(common, tmp_path):
    ap = common.parser("x")
    common.add_paper_data_argument(ap)
    ap.add_argument("--panel-alphaearth", default=None)
    common.require(ap, ap.parse_args(["--paper-data", str(tmp_path)]), "panel-alphaearth")
    common.require(ap, ap.parse_args(["--panel-alphaearth", "p"]), "panel-alphaearth")
    with pytest.raises(SystemExit):
        common.require(ap, ap.parse_args([]), "panel-alphaearth")


def test_paper_store_is_the_preset_then_the_imagery(common):
    build_store = importlib.import_module("build_store")
    first, second = build_store.download_configs("s.zarr", gee_project="p", rows=3)
    assert first["download"]["preset"] == second["download"]["preset"] == "majortom210k"
    assert first["download"]["products"] is None and second["download"]["products"] == ["sentinel2_l2a"]
    assert first["download"]["out"] == "s.zarr" and first["download"]["rows"] == 3
    assert first["download"]["gee_project"] == "p"


# --------------------------------------------------------------------------- --paper-data: Figs. 1-3, Table S3


def _paper_reconstruction(root: Path) -> Path:
    """Tiny ``real_products.json`` and ``dense_topk.json`` in the camera-ready format."""
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        {"kind": "sae", "arch": arch, "latent_dim": 1024, "k": k, "frozen": frozen, "seed": seed}
        | {"l0": float(k), "fve": 0.9 - 0.1 * frozen + 0.001 * seed, "recon_cosine": 0.95}
        for arch, frozen in (("batchtopk", False), ("matryoshka", False), ("relu_l1", False), ("batchtopk", True))
        for k in (8, 16, 32)
        for seed in (0, 1, 2)
    ]
    rows.append({"kind": "pca", "arch": "pca", "latent_dim": 64, "seed": 0, "l0": 16.0, "fve": 0.5})
    dense = {m: {f"rank_{k}": [{"fve": 0.5 + k / 100, "l0": 7.0}] for k in (8, 16, 32)} for m in ("raw", "pca", "nmf")}
    products = {emb: {"rows": rows} for emb in ("alphaearth", "tessera")}
    (root / "real_products.json").write_text(json.dumps({"products": products}))
    (root / "dense_topk.json").write_text(json.dumps({"products": {emb: dense for emb in products}}))
    stats = {"clean": {"mse": 1e-3, "fve": 0.9, "cos": 0.95, "l0": 16.0}}
    keys = ("batchtopk", "matryoshka", "relu", "frozen_decoder", "raw", "pca", "ica", "nmf")
    (root / "recon_table.json").write_text(json.dumps({e: dict.fromkeys(keys, stats) for e in ("ae", "te")}))
    return root


def _paper_variogram(root: Path, embedding: str) -> None:
    """A tiny ``variogram_<embedding>.npz`` with one live raw dimension per arm and three frozen seeds."""
    np = pytest.importorskip("numpy")
    d_km = np.array([0.01, 0.32, 0.63, 13.0, 100.0, 1000.0])
    arrays = {}
    for arm, n_by_run in (("raw", [2]), ("batchtopk", [3]), ("frozen", [2, 3, 1])):
        n = sum(n_by_run)
        arrays |= {
            f"{arm}/d_km": d_km,
            f"{arm}/curve": np.tile(np.linspace(0.1, 1.0, len(d_km)), (n, 1)).astype("float32"),
            f"{arm}/within": np.full(n, 0.3),
            f"{arm}/nugget": np.full(n, 0.05, dtype="float32"),
            f"{arm}/length_km": np.full(n, 100.0),
            f"{arm}/status": np.zeros(n, dtype="int8"),
            f"{arm}/n_by_run": np.array(n_by_run),
        }
    np.savez(root / f"variogram_{embedding}.npz", **arrays)


def test_fig2_paper_rows_feed_the_panel_summary(common, tmp_path):
    pytest.importorskip("nicefigs")
    fig2 = importlib.import_module("fig2_pareto")
    cells = fig2.summarise(fig2.paper_rows(_paper_reconstruction(tmp_path), "alphaearth"))
    assert set(cells["sae"]) == {(v, 1024, k) for v in ("batchtopk", "matryoshka", "relu_l1") for k in (8, 16, 32)}
    assert cells["sae"][("batchtopk", 1024, 16)]["n"] == 3
    assert cells["frozen"][16]["fve"] == pytest.approx(0.8)  # seed 0 only
    assert set(cells["dense"]) == {"raw", "pca", "nmf"}  # the store-dimension PCA row is not a k-component baseline
    assert cells["dense"]["pca"][16]["l0"] == 16.0 and cells["dense"]["pca"][16]["fve"] == pytest.approx(0.66)


def test_table_reconstruction_reads_the_clean_paper_rows(common, tmp_path):
    table = importlib.import_module("table_reconstruction")
    rows = table.paper_table_rows(_paper_reconstruction(tmp_path), "tessera")
    assert [k for k, _ in table.ROWS] == list(rows)
    assert rows["frozen"] == {"recon_mse": 1e-3, "fve": 0.9, "pixel_cosine": 0.95, "l0": 16.0}


def test_fig3_load_paper_splits_pooled_frozen_seeds(common, tmp_path):
    pytest.importorskip("nicefigs")
    fig3 = importlib.import_module("fig3_variogram")
    _paper_variogram(tmp_path, "ae")
    d = fig3.load_paper(tmp_path / "variogram_ae.npz")
    assert d["n_within"] == 3
    assert len(d["frozen"]["status"]) == 2 and sorted(d["frozen_seeds"]) == [0, 1, 2]
    assert [len(f["within"]) for f in d["frozen_seeds"].values()] == [2, 3, 1]
    assert set(fig3.summary(d)) == {"raw", "batchtopk", "frozen", "frozen_seeds"}


@pytest.mark.parametrize(
    ("script", "outputs"),
    [
        ("fig2_pareto", ["fig2_pareto.pdf", "fig2_pareto.json"]),
        ("table_reconstruction", ["table_reconstruction.tex", "table_reconstruction.json"]),
        (
            "fig3_variogram",
            ["fig3_correlation_length.pdf", "figS_variogram_alphaearth.pdf", "figS_variogram_tessera.pdf"],
        ),
    ],
)
def test_paper_data_main_writes_the_outputs(common, tmp_path, monkeypatch, script, outputs):
    pytest.importorskip("nicefigs")
    paper = _paper_reconstruction(tmp_path / "paper")
    for emb in ("ae", "te"):
        _paper_variogram(paper, emb)
    module = importlib.import_module(script)
    monkeypatch.setattr(sys, "argv", [script, "--paper-data", str(paper), "--out", str(tmp_path / "out")])
    module.main()
    for name in outputs:
        assert (tmp_path / "out" / name).exists()


def test_fig1_paper_regional_features_name_the_regions(common, tmp_path):
    pytest.importorskip("nicefigs")
    np = pytest.importorskip("numpy")
    fig1 = importlib.import_module("fig1_overview")
    np.savez(
        tmp_path / "regional_features.npz",
        features=np.array([437, 635]),
        lon=np.zeros(3, "float32"),
        lat=np.zeros(3, "float32"),
        prevalence=np.ones((2, 3), "float32"),
    )
    d = fig1.paper_regional_features(tmp_path)
    assert [c["feature"] for c in d["chosen"]] == [437, 635]
    assert [c["region"] for c in d["chosen"]] == [r[0] for r in fig1.REGIONS]
    assert d["prevalence"].shape == (2, 3)


@pytest.mark.parametrize("script", ["fig1_overview", "fig2_pareto", "table_reconstruction", "fig3_variogram"])
def test_stage_outputs_or_paper_data_are_required(common, monkeypatch, script):
    pytest.importorskip("nicefigs")
    module = importlib.import_module(script)
    monkeypatch.setattr(sys, "argv", [script])
    with pytest.raises(SystemExit):
        module.main()


# --------------------------------------------------------------------------- --paper-data: alignment, Moran, tables


def _run_main(module, monkeypatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", [module.__name__, *map(str, argv)])
    module.main()


def _product(dim: int, *, aligned: bool) -> dict:
    def arm(fve: float, l0: float, n: int) -> list[dict]:
        align = {"n_live": 10, "le1": 0.5, "eq1": 0.25, "n_aligned": 4, "median": 1.0}
        return [{"fve": fve + 0.01 * s, "l0": l0, **({"align": align} if aligned else {})} for s in range(n)]

    d = {"btk": arm(0.9, 185.0, 3), "frozen": arm(0.5, 180.0, 3)}
    d |= {"btk16": arm(0.7, 16.2, 1), "frozen16": arm(0.2, 15.8, 1)}
    d |= {f"{m}_{w}": {"fve": f} for w in (16, 185) for m, f in (("pca", 0.6), ("raw", 0.1))}
    if aligned:
        align = {"n_live": dim, "le1": 0.1, "eq1": 0.1, "n_aligned": dim, "median": 5.0}
        d |= {"align_raw": align, "align_pca": align}
    return d


@pytest.mark.parametrize("flat", [False, True], ids=["subfolder", "flat"])
def test_further_embeddings_tables_from_paper_data(common, tmp_path, monkeypatch, flat):
    module = importlib.import_module("table_further_embeddings")
    for p, aligned in (("copernicusfm", False), ("olmoearth", True), ("mmearth", True)):
        path = tmp_path / "pd" / (f"products__{p}.json" if flat else f"products/{p}.json")
        common.write_json(_product(module.DIMS[p], aligned=aligned), path)
    _run_main(module, monkeypatch, "--paper-data", tmp_path / "pd", "--out", tmp_path / "out")

    out = json.loads((tmp_path / "out" / "table_further_embeddings.json").read_text())
    k16, k192 = out["fve"]["olmoearth"]
    assert (k16["k"], k16["M"], k192["k"]) == (16, 4096, 192)
    assert k16["pca_width"] == 16 and k192["sae"] == pytest.approx(0.91) and k192["raw"] == pytest.approx(0.1)
    assert set(out["alignment"]) == {"olmoearth", "mmearth"}
    assert out["alignment"]["mmearth"]["sae"]["ge1"] == 4 and out["alignment"]["mmearth"]["raw"]["live"] == 320
    assert "Copernicus-FM" in (tmp_path / "out" / "table_further_embeddings_fve.tex").read_text()


def test_further_embeddings_needs_stage_outputs_or_paper_data(common, monkeypatch):
    module = importlib.import_module("table_further_embeddings")
    with pytest.raises(SystemExit):
        _run_main(module, monkeypatch, "--panel-olmoearth", "panel")


def test_ucov_table_from_paper_data(common, tmp_path, monkeypatch):
    module = importlib.import_module("table_ucov")
    row = {"ucov_060": 140, "ucov_065": 116}
    keys = ("raw", "pca", "ica", "nmf", "relu", "batchtopk", "matryoshka")
    tables = {"ucov": {"ae": {k: row for k in (*keys, "frozen_decoder")}, "te": {k: row for k in keys}}}
    common.write_json({**tables, "n_classes_ae": [159]}, tmp_path / "pd" / "tables.json")
    null = [{"arm": a, "n_features": m, "emf_sum": float(m)} for a in ("trained", "frozen decoder") for m in (64, 1024)]
    common.write_json(null, tmp_path / "pd" / "null_coverage.json")
    _run_main(module, monkeypatch, "--paper-data", tmp_path / "pd", "--out", tmp_path / "out")

    out = json.loads((tmp_path / "out" / "table_ucov.json").read_text())
    assert list(out) == ["alphaearth", "tessera"]
    assert out["alphaearth"]["Frozen decoder"] == {"ucov_0.60": 140, "ucov_0.65": 116, "n_valid": 159, "emf": 1024.0}
    assert out["alphaearth"]["BatchTopK"]["emf"] == 1024.0 and out["alphaearth"]["Raw"]["emf"] is None
    assert "Frozen decoder" not in out["tessera"] and out["tessera"]["PCA-D"]["n_valid"] is None
    assert r"Frozen decoder & $140$ & $116$ & -- & --" in (tmp_path / "out" / "table_ucov.tex").read_text()


def test_betti_table_from_paper_data_reports_seed_spread(common, tmp_path, monkeypatch):
    module = importlib.import_module("table_betti")

    def fit(mean: float) -> dict:
        return {k: {"mean": mean, "max": 2 * mean, "n_scored": 10} for k in ("geoms_500km", "ms_image_set")}

    for short in ("ae", "te"):
        arms = {"raw": {"seeds": [fit(0.01)]}, "frozen": {"seeds": [fit(0.1), fit(0.2), fit(0.3)]}}
        if short == "ae":
            arms["sweep_batchtopk"] = {"seeds": [fit(0.1), fit(0.1), fit(0.1)]}
        score = {
            "n_patches": {"moran_set": 1500, "image_set": 3000},
            "mean_pairwise_similarity": {"geoms_500km": 0.008, "ms_image_set": 0.262},
            "arms": arms,
        }
        common.write_json(score, tmp_path / "pd" / f"betti_scores_{short}.json")
    common.write_json({s: fit(0.12) for s in ("ae", "te")}, tmp_path / "pd" / "betti_relu_fixed.json")
    _run_main(module, monkeypatch, "--paper-data", tmp_path / "pd", "--out", tmp_path / "out")

    out = json.loads((tmp_path / "out" / "table_betti.json").read_text())
    assert list(out["alphaearth"]["rows"]) == ["Raw", "Frozen decoder", r"ReLU+$\ell_1$", "BatchTopK, 3 seeds"]
    assert "BatchTopK, 3 seeds" not in out["tessera"]["rows"]
    assert module.aggregate(out["alphaearth"]["rows"]["Frozen decoder"], "geoms", "mean") == pytest.approx((0.2, 0.1))
    tex = (tmp_path / "out" / "table_betti.tex").read_text()
    assert r"\multicolumn{4}{c}{AlphaEarth}" in tex and r"$0.200{\scriptstyle\pm0.100}$" in tex


def test_moran_figure_from_paper_data(common, tmp_path, monkeypatch):
    pytest.importorskip("nicefigs")
    np = pytest.importorskip("numpy")
    module = importlib.import_module("figS_moran")
    rng = np.random.default_rng(0)
    arrays = {}
    for short, dim in (("ae", 64), ("te", 128)):
        arrays |= {f"{short}/{m}": rng.uniform(0, 1, dim).astype("float32") for m in module.ORDER}
        arrays[f"{short}/frozen_s1"] = np.full(8, 0.3, dtype="float32")
    arrays["ae/matryoshka"][0] = np.nan
    (tmp_path / "pd").mkdir()
    np.savez(tmp_path / "pd" / "spatial_coherence.npz", **arrays)
    _run_main(module, monkeypatch, "--paper-data", tmp_path / "pd", "--out", tmp_path / "out")

    out = json.loads((tmp_path / "out" / "figS_moran.json").read_text())
    assert out["alphaearth"]["n_features"]["matryoshka"] == 63 and out["tessera"]["n_features"]["raw"] == 128
    assert out["tessera"]["frozen_median_by_seed"]["1"] == pytest.approx(0.3)
    assert (tmp_path / "out" / "figS_moran.png").exists()


def test_monosemanticity_figure_from_paper_data_pools_frozen_seeds(common, tmp_path, monkeypatch):
    pytest.importorskip("nicefigs")
    np = pytest.importorskip("numpy")
    module = importlib.import_module("fig4_monosemanticity")
    rng = np.random.default_rng(0)
    groups = module.GROUPS[:14]

    def counts(n: int) -> dict:
        return {"n_groups": rng.integers(0, 4, n).tolist(), "max_auc": rng.uniform(0.5, 1.0, n).tolist()}

    d = {
        "methods": {
            "sae": counts(30),
            "raw": counts(16),
            "pca": counts(16),
            "frozen": {**counts(20), "seed": 0},
            "frozen_other_seeds": {"s1": counts(25), "s2": counts(22)},
        },
        "full": {
            "groups": groups,
            "sae": {"auc": rng.uniform(0.5, 1.0, (30, len(groups))).tolist()},
            "raw": {"auc": rng.uniform(0.5, 1.0, (16, len(groups))).tolist()},
        },
    }
    common.write_json(d, tmp_path / "pd" / "monosemanticity_matryoshka_te.json")
    args = ("--paper-data", tmp_path / "pd", "--embedding", "tessera", "--out", tmp_path / "out", "--name", "s7")
    _run_main(module, monkeypatch, *args, "--frozen-seeds", "0", "2")

    out = json.loads((tmp_path / "out" / "s7.json").read_text())
    assert out["sae"]["n_live"] == 30 and out["frozen"]["n_live"] == 42 and out["frozen_s1"]["n_live"] == 25
    assert len(out["shown_groups"]) == module.N_SHOW and set(out["shown_groups"]) <= set(groups)
    assert (tmp_path / "out" / "s7.png").exists()


# ------------------------------------------------- --paper-data: Fig. 5, Figs. S9-S11, Table S8


def _woody(feat: int, share: list[float]) -> dict:
    return {"feat": feat, "auc": 0.7, "class": "lc_esa_wc:10", "pair_share": share}


def _paper_subconcepts(root: Path) -> Path:
    """Tiny camera-ready inputs of the sub-concept and gallery figures, in the Zenodo figure-data layout."""
    np = pytest.importorskip("numpy")
    n = 50
    for stem, feats in (("subconcepts", [139, 124, 35]), ("subconcepts_frozen", [124, 653, 283])):
        np.savez(
            root / f"{stem}.npz",
            features=np.array(feats),
            lon=np.linspace(-170, 170, n, dtype="float32"),
            lat=np.linspace(-50, 70, n, dtype="float32"),
            tops=np.arange(30, dtype="int32").reshape(3, 10),
            matched=np.zeros((3, n), dtype="int8"),
            references=np.array(["chelsa_kg", "resolve_biomes", "pnv"]),
            gallery=np.arange(6, dtype="int32").reshape(3, 2),
            gallery_maps=np.ones((3, 2, 64, 64), dtype="float16"),
        )
    classes = [{"class": "Af", "share": 0.8, "lift": 15.6}, {"class": "Am", "share": 0.16, "lift": 4.2}]
    table = [
        {"feat": f, "mean_abs_lat": 3.0, "chelsa_kg": classes, "resolve_biomes": classes, "pnv": classes}
        for f in (139, 124, 35)
    ]
    (root / "subconcepts.json").write_text(json.dumps({"top_n": 2000, "table": table}))
    features = {
        "trained": [_woody(139, [0.96, 0.04, 0.0]), _woody(124, [0.0, 0.63, 0.0]), _woody(35, [0.0, 0.0, 0.91])],
        "frozen": [_woody(124, [0.99, 0.0, 0.0]), _woody(653, [0.0, 0.98, 0.0]), _woody(283, [0.0, 0.0, 0.99])],
    }
    (root / "subconcepts_frozen.json").write_text(json.dumps({"frozen_picks": [124, 653, 283], "features": features}))
    concepts = [
        {
            "name": f"c{i}",
            "product": "lc_esa_wc",
            "code": 10 * i,
            "feat": i,
            "feat_auc": 0.9,
            "dim": i,
            "dim_sign": -1.0,
            "dim_auc": 0.8,
            "n_present": 3,
            "cell": list(range(6)),
            "lon": [10.0] * 6,
            "lat": [20.0] * 6,
            "class_frac": [0.2, 0.2, 0.2, 0.0, 0.0, 0.0],
        }
        for i in range(6)
    ]
    for bb in ("ae", "te"):
        np.savez(
            root / f"qualitative_{bb}.npz",
            meta=np.array(json.dumps({"backbone": bb, "concepts": concepts})),
            sae=np.ones((6, 6, 64, 64), dtype="float16"),
            raw=np.ones((6, 6, 64, 64), dtype="float16"),
        )
    return root


@pytest.fixture
def offline(common, monkeypatch):
    """No network: Esri tiles fail, so the paper-data figures draw grey imagery."""

    def fail(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(common, "esri_imagery", fail)


def test_fig5_paper_data_loads_the_camera_ready_gallery_and_subconcepts(common, offline, tmp_path):
    pytest.importorskip("nicefigs")
    f5 = importlib.import_module("fig5_aligned_features")
    gallery, d = f5.from_paper_data(_paper_subconcepts(tmp_path))
    assert [c["name"] for c in gallery] == [c[0] for c in f5.CONCEPTS]
    assert gallery[0]["sae"].shape == (6, 64, 64) and gallery[0]["rgb"].shape == (6, 64, 64, 3)
    assert d["features"] == [139, 124, 35] and d["gallery_rgb"].shape == (3, 2, 64, 64, 3)
    report = f5.report(gallery, d)
    assert [r["feat"] for r in report["picked"]] == [139, 124, 35]
    assert report["woody"]["n_woody"] == 3


def test_frozen_subconcepts_from_paper_data_have_no_agreement_table(common, offline, tmp_path):
    pytest.importorskip("nicefigs")
    f5 = importlib.import_module("fig5_aligned_features")
    d = f5.paper_subconcepts(_paper_subconcepts(tmp_path), "frozen")
    assert d["features"] == [124, 653, 283] and d["table"] == []
    assert f5.woody_summary(d["woody"])["n_pair_share_ge_0.8"] == [1, 1, 1]


def test_table_subconcepts_main_writes_the_table_from_paper_data(common, tmp_path, monkeypatch):
    pytest.importorskip("nicefigs")
    table = importlib.import_module("table_subconcepts")
    paper = _paper_subconcepts(tmp_path)
    monkeypatch.setattr(sys, "argv", ["table_subconcepts.py", "--paper-data", str(paper), "--out", str(tmp_path / "o")])
    table.main()
    tex = (tmp_path / "o" / "table_subconcepts.tex").read_text()
    assert "f139" in tex and "Af 0.80 (15.6); Am 0.16 (4.2)" in tex
    out = json.loads((tmp_path / "o" / "table_subconcepts.json").read_text())
    assert [r["feat"] for r in out["frozen"]["picked"]] == [124, 653, 283]


def test_table_subconcepts_needs_runs_or_paper_data(common, monkeypatch):
    pytest.importorskip("nicefigs")
    table = importlib.import_module("table_subconcepts")
    monkeypatch.setattr(sys, "argv", ["table_subconcepts.py"])
    with pytest.raises(SystemExit):
        table.main()


def test_galleries_main_draws_the_tessera_gallery_from_paper_data(common, offline, tmp_path, monkeypatch):
    pytest.importorskip("nicefigs")
    galleries = importlib.import_module("figS_galleries")
    paper = _paper_subconcepts(tmp_path)
    argv = ["figS_galleries.py", "--paper-data", str(paper), "--embedding", "tessera", "--out", str(tmp_path / "o")]
    monkeypatch.setattr(sys, "argv", argv)
    galleries.main()
    assert (tmp_path / "o" / "figS_qual_only_tessera.png").exists()
    out = json.loads((tmp_path / "o" / "figS_qual_only_tessera.json").read_text())
    assert out[0]["feature"] == 0 and out[0]["dim_sign"] == -1.0


def _gain(median: float) -> dict:
    return {"n": 88, "wins": 60, "median": median, "ci": [median - 0.05, median + 0.05]}


def _applicability(path: Path) -> None:
    codes = {"sae": _gain(0.2), "random_sparse": _gain(0.1), "kmeans": _gain(-0.1)}
    targets = {t: codes for t in ("canopy_height", "elevation", "gpp")}
    path.write_text(json.dumps({"metric": "rho", "alphaearth": targets, "tessera": targets}))


def test_aoa_gains_are_read_from_the_paper_data(common, tmp_path):
    _applicability(tmp_path / "applicability.json")
    gains = importlib.import_module("fig6_aoa_gain").paper_gains(tmp_path)
    assert set(gains) == {"alphaearth", "tessera"}
    assert set(gains["tessera"]) == {"canopy_height", "elevation"}
    assert gains["alphaearth"]["elevation"]["random"]["median"] == pytest.approx(0.1)
    assert set(gains["alphaearth"]["elevation"]) == {"sae", "random"}


def test_aoa_gain_figure_plots_from_the_paper_data(common, tmp_path, monkeypatch):
    pytest.importorskip("nicefigs")
    _applicability(tmp_path / "applicability.json")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["fig6_aoa_gain.py", "--paper-data", str(tmp_path), "--out", str(out)])
    importlib.import_module("fig6_aoa_gain").main()
    assert (out / "fig6_aoa_gain.pdf").exists()
    assert json.loads((out / "fig6_aoa_gain.json").read_text())["tessera"]["canopy_height"]["sae"]["n"] == 88


@pytest.mark.parametrize("script", ["fig6_aoa_gain", "figS_aoa", "figS_aoa_gain", "figS_pcs_vs_atoms"])
def test_scripts_need_stage_outputs_or_paper_data(common, script, monkeypatch, capsys):
    pytest.importorskip("nicefigs")
    monkeypatch.setattr(sys, "argv", [f"{script}.py"])
    with pytest.raises(SystemExit):
        importlib.import_module(script).main()
    assert "--paper-data" in capsys.readouterr().err


def test_paper_data_names_the_missing_file(common, tmp_path):
    with pytest.raises(FileNotFoundError, match="figure-data"):
        importlib.import_module("fig6_aoa_gain").paper_gains(tmp_path)


def test_pcs_vs_atoms_paper_data_maps_the_closest_feature_of_each_component(common, tmp_path):
    pytest.importorskip("nicefigs")
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    top, best = [5, 7, 9, 11, 13, 15], [7, 9, 5, 99, 13, 15]
    corr = rng.uniform(-1, 1, (6, 6))
    meta = {
        "n_pixels": 20,
        "pc_var_share": [0.3, 0.2, 0.1, 0.05, 0.04, 0.03],
        "top_atoms": top,
        "atom_energy_share": [0.1] * 6,
        "atom_energy_share_cum": 0.6,
        "cos_top": rng.uniform(0, 1, (6, 6)).tolist(),
        "corr_top": corr.tolist(),
        "pc_best_atom": best,
        "pc_best_cos": [0.6] * 6,
    }
    atom = rng.uniform(0, 1, (20, 6)).astype("float16")
    np.savez(tmp_path / "pcs_vs_atoms.npz", lon=np.zeros(20, "float32"), lat=np.zeros(20, "float32"),
             pc=rng.normal(size=(20, 6)).astype("float16"), atom=atom, meta=np.array(json.dumps(meta)))  # fmt: skip
    d = importlib.import_module("figS_pcs_vs_atoms").from_paper_data(tmp_path)
    assert d["top"] == top and d["meta"]["pc_best_feature"] == best
    np.testing.assert_allclose(d["act"], atom[:, [1, 2, 0]].astype("float32"))
    assert d["meta"]["pc_best_corr"][:3] == pytest.approx([corr[0, 1], corr[1, 2], corr[2, 0]])
    assert d["meta"]["pc_best_corr"][3] is None and d["meta"]["pc_best_in_top"][3] is False
