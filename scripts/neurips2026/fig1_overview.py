"""Fig. 1 (fig1_overview): sparse features, the evaluation protocol and two regional features.

Panels a and b are schematics drawn as vectors (globes with cartopy). Panel c maps the prevalence of
two regional Matryoshka features on AlphaEarth: the share of a test patch's pixels on which the
feature is active (``feat_patch_stats`` of the run's ``geosae predict`` store), averaged over the
test patches in each 2° cell. The two features are chosen by a fixed rule: for each region box
(``REGIONS``: central Australia and the Tibetan Plateau), the feature with the largest
share of its summed prevalence inside the box, among features active on at least a fifth of the
pixels of the box's patches on average; a feature is shown once. ``--paper-data`` maps the camera-ready
``regional_features.npz`` instead.

    python scripts/neurips2026/fig1_overview.py runs/alphaearth/matryoshka_M1024_k16_s0 --out figures
    python scripts/neurips2026/fig1_overview.py --paper-data paper_data --out figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nicefigs as nf
import numpy as np
import zarr
from _common import (
    add_paper_data_argument,
    panel_title,
    paper_file,
    parser,
    predictions,
    require,
    save,
    set_style,
    write_json,
)
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

#: Region boxes of panel c: (label, lon min, lon max, lat min, lat max).
REGIONS = (("central Australia", 122.0, 138.0, -30.0, -20.0), ("Tibetan Plateau", 78.0, 100.0, 28.0, 37.0))
#: Smallest mean prevalence over a box's patches for a feature to be eligible for that box.
MIN_PREVALENCE = 0.2
PIXELS = 64 * 64

# Type sizes in figure points, set by ``build_figure`` from the printed type scale.
FS_ANNOTATION = FS_AXIS_LABEL = FS_LOWER = 10.0

C_BOX = "#222222"
C_SOFT = "#8a8a8a"
C_FILL = "#f6f3ee"
C_ENC = "#dfe7ef"
C_DEC = "#e9e2d5"
C_DENSE = "#4a6b8a"
C_SAE = "#c9612b"
C_WATER = "#2f78a8"
C_FOREST = "#2f8a4a"
C_BAD = "#b23434"
C_OK = "#2f8a4a"
C_NULL = "#9a9a9a"
C_NULL_FILL = "#e4e4e4"
C_NULL_TEXT = "#555555"


def _arrow(ax, p, q, *, color=C_BOX, lw=0.8, style="-|>", ms=6, dashed=False, alpha=1.0, zorder=4, rad=None):
    kw = dict(
        arrowstyle=style, mutation_scale=ms, color=color, linewidth=lw, alpha=alpha, zorder=zorder, shrinkA=0, shrinkB=0
    )
    if dashed:
        kw["linestyle"] = (0, (2.0, 1.4))
    if rad is not None:
        kw["connectionstyle"] = f"arc3,rad={rad}"
    a = FancyArrowPatch(p, q, **kw)
    ax.add_patch(a)
    return a


def _text(ax, x, y, s, *, size=FS_ANNOTATION, ha="center", va="center", color=C_BOX, **kw):
    return ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=color, **kw)


def _trapezoid(ax, x0, x1, y_left_c, y_right_c, h_left, h_right, *, fc=C_ENC, ec=C_BOX, lw=0.7, zorder=3):
    pts = [
        (x0, y_left_c - h_left / 2),
        (x1, y_right_c - h_right / 2),
        (x1, y_right_c + h_right / 2),
        (x0, y_left_c + h_left / 2),
    ]
    poly = Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(poly)
    return poly


def _feature_column(
    ax, cx, cy_bot, w, n, *, active=None, colors=None, inactive_fc="#e4e4e4", ec=C_BOX, lw=0.3, zorder=4, gap=0.0015
):
    if active is None:
        active = list(range(n))
    active = set(active)
    colors = colors or {}
    for i in range(n):
        y = cy_bot + i * (w + gap)
        fc = colors.get(i, C_SAE) if i in active else inactive_fc
        ax.add_patch(
            Rectangle(
                (cx - w / 2, y),
                w,
                w,
                facecolor=fc,
                edgecolor=ec,
                linewidth=lw,
                zorder=zorder,
            )
        )


def _feature_row(
    ax, cx_left, cy, w, n, *, active=None, colors=None, inactive_fc="#e4e4e4", ec=C_BOX, lw=0.3, zorder=4, gap=0.0015
):
    """Horizontal vector: n squares laid left → right starting at cx_left."""
    if active is None:
        active = list(range(n))
    active = set(active)
    colors = colors or {}
    for i in range(n):
        x = cx_left + i * (w + gap)
        fc = colors.get(i, C_SAE) if i in active else inactive_fc
        ax.add_patch(
            Rectangle(
                (x, cy - w / 2),
                w,
                w,
                facecolor=fc,
                edgecolor=ec,
                linewidth=lw,
                zorder=zorder,
            )
        )


def _land_tile(ax, cx, cy, half, kind):
    palettes = {
        "water": [(0.18, 0.42, 0.65), (0.22, 0.50, 0.72), (0.28, 0.56, 0.78)],
        "forest": [(0.18, 0.45, 0.22), (0.23, 0.55, 0.28), (0.30, 0.62, 0.32)],
    }
    cols = palettes.get(kind, palettes["forest"])
    rng = np.random.default_rng({"water": 1, "forest": 2}.get(kind, 5))
    arr = np.zeros((6, 6, 3))
    for i in range(6):
        for j in range(6):
            c = cols[rng.integers(0, len(cols))]
            arr[i, j] = [
                c[0] + 0.05 * rng.standard_normal(),
                c[1] + 0.05 * rng.standard_normal(),
                c[2] + 0.05 * rng.standard_normal(),
            ]
    arr = np.clip(arr, 0, 1)
    ax.imshow(arr, extent=(cx - half, cx + half, cy - half, cy + half), interpolation="nearest", zorder=2)
    ax.add_patch(
        Rectangle(
            (cx - half, cy - half),
            2 * half,
            2 * half,
            fill=False,
            linewidth=0.6,
            edgecolor=C_BOX,
            zorder=2.5,
        )
    )


# Cartopy Ortho-projection globes are inserted as figure-level insets
# after the main panels are drawn (so their data→figure transforms are
# final). Panel code queues a request; build_figure() processes it.
_GLOBE_QUEUE: list = []


def _request_globe(parent_ax, cx, cy, r, *, central_lon=15.0, central_lat=10.0, pin=None, pin_color=C_BAD):
    """Queue a cartopy Ortho globe at (cx, cy) with data-coord radius r.

    ``pin`` is a ``(lon, lat)`` tuple in degrees.
    """
    _GLOBE_QUEUE.append(
        {
            "parent_ax": parent_ax,
            "cx": cx,
            "cy": cy,
            "r": r,
            "central_lon": central_lon,
            "central_lat": central_lat,
            "pin": pin,
            "pin_color": pin_color,
        }
    )


def _process_globes(fig):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    for g in _GLOBE_QUEUE:
        pax = g["parent_ax"]
        cx, cy, r = g["cx"], g["cy"], g["r"]
        bbox_data = Bbox.from_bounds(cx - r, cy - r, 2 * r, 2 * r)
        bbox_disp = pax.transData.transform_bbox(bbox_data)
        bbox_fig = bbox_disp.transformed(fig.transFigure.inverted())

        proj = ccrs.Orthographic(central_longitude=g["central_lon"], central_latitude=g["central_lat"])
        gax = fig.add_axes(bbox_fig.bounds, projection=proj, zorder=10)
        gax.set_global()
        gax.add_feature(cfeature.OCEAN, facecolor="#d4e4ef", edgecolor="none", zorder=1)
        gax.add_feature(cfeature.LAND, facecolor="#9fc28a", edgecolor="#6a9258", linewidth=0.25, zorder=2)
        gax.spines["geo"].set_linewidth(0.5)
        gax.spines["geo"].set_edgecolor(C_BOX)

        if g["pin"] is not None:
            pin_lon, pin_lat = g["pin"]
            gax.plot(
                pin_lon,
                pin_lat,
                "o",
                markersize=4.0,
                markerfacecolor=g["pin_color"],
                markeredgecolor=C_BOX,
                markeredgewidth=0.3,
                transform=ccrs.PlateCarree(),
                zorder=10,
            )
    _GLOBE_QUEUE.clear()


# ---------------------------------------------------------------------------
# Panel (a)
# ---------------------------------------------------------------------------


def _panel_a(ax):
    ax.set_xlim(0, 1.8)
    ax.set_ylim(0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---------- TOP: main pipeline, uniform spacing ----------
    y_pipe = 0.75
    step = 0.19  # uniform distance between component centres

    cx_globe = 0.11
    cx_e = cx_globe + step
    cx_enc = cx_e + step
    cx_z = cx_enc + step
    cx_dec = cx_z + step
    cx_ehat = cx_dec + step

    _request_globe(ax, cx_globe, y_pipe, 0.065, central_lon=15, central_lat=10, pin=(20, 5), pin_color=C_BAD)

    _land_tile(ax, cx_e, y_pipe, 0.050, "forest")
    _text(ax, cx_e, y_pipe + 0.095, r"$\mathbf{e}$", size=FS_AXIS_LABEL)

    _trapezoid(ax, cx_enc - 0.085, cx_enc + 0.085, y_pipe, y_pipe, 0.14, 0.34, fc=C_ENC)
    _text(ax, cx_enc, y_pipe, r"$\mathbf{W}_{\mathrm{enc}}$", size=FS_ANNOTATION)

    # Sparse latent as a vertical stack (main pipeline).
    n_cells = 10
    col_w = 0.040
    col_bot = y_pipe - (n_cells * col_w + (n_cells - 1) * 0.0015) / 2
    _feature_column(ax, cx_z, col_bot, col_w, n_cells, active=[1, 4, 7], colors={1: C_WATER, 4: C_FOREST, 7: "#c9a46b"})
    _text(ax, cx_z, col_bot + n_cells * col_w + 0.040, r"$\mathbf{z}$", size=FS_AXIS_LABEL, color=C_SAE)

    _trapezoid(ax, cx_dec - 0.085, cx_dec + 0.085, y_pipe, y_pipe, 0.34, 0.14, fc=C_DEC)
    _text(ax, cx_dec, y_pipe, r"$\mathbf{W}_{\mathrm{dec}}$", size=FS_ANNOTATION)

    _land_tile(ax, cx_ehat, y_pipe, 0.050, "forest")
    _text(ax, cx_ehat, y_pipe + 0.095, r"$\hat{\mathbf{e}}$", size=FS_AXIS_LABEL)

    # Uniform-length arrows.
    def _pipe_arrow(x_from, x_to):
        _arrow(ax, (x_from, y_pipe), (x_to, y_pipe), lw=0.9, ms=6)

    _pipe_arrow(cx_globe + 0.075, cx_e - 0.055)
    _pipe_arrow(cx_e + 0.055, cx_enc - 0.095)
    _pipe_arrow(cx_enc + 0.095, cx_z - col_w / 2 - 0.010)
    _pipe_arrow(cx_z + col_w / 2 + 0.010, cx_dec - 0.095)
    _pipe_arrow(cx_dec + 0.095, cx_ehat - 0.055)

    # ---------- BOTTOM: polysemantic vs monosemantic cartoon ----------
    # Horizontal vectors (rotated 90°). Two rows: water + forest.
    # Dense vectors on the LEFT (anchored under [e,W_enc]); sparse vectors
    # BELOW the pipeline's z column.
    ax.plot([0.03, cx_ehat + 0.08], [0.525, 0.525], color=C_SOFT, lw=0.3, linestyle=(0, (2, 2)), zorder=1)

    # Row y positions.
    row_y_top = 0.35  # "water"
    row_y_bot = 0.13  # "forest"

    # Bottom cartoon: dense vector anchored under [e, W_enc], sparse vector
    # under z. Dense sits well left of sparse so the curved arrows have a
    # visible run and the two vectors do not overlap.
    mg_x = 0.16
    d_cx_left = 0.33

    # Mini-globes — loc. 1 pins an ocean point, loc. 2 pins a forest.
    _request_globe(ax, mg_x, row_y_top, 0.040, central_lon=-25, central_lat=20, pin=(-35, 25), pin_color=C_WATER)
    _request_globe(ax, mg_x, row_y_bot, 0.040, central_lon=15, central_lat=5, pin=(20, 2), pin_color=C_FOREST)
    _text(ax, mg_x, row_y_top - 0.015 - 0.060, "water pixel", size=FS_LOWER, color=C_WATER)
    _text(ax, mg_x, row_y_bot - 0.015 - 0.065, "forest pixel", size=FS_LOWER, color=C_FOREST)

    # Dense vectors: every cell fires (muted) except the shared
    # polysemantic cell which fires at full saturation.
    d_n = 8
    d_w = 0.030
    # Both rows share the SAME active dense cell → polysemantic.
    shared_dim = 3
    C_DENSE_MUTED = "#b5c6d8"

    def _dense_colors(highlight):
        cs = {i: C_DENSE_MUTED for i in range(d_n)}
        cs[shared_dim] = C_DENSE
        for k in highlight:
            cs[k] = C_DENSE
        return cs

    _feature_row(
        ax,
        d_cx_left,
        row_y_top,
        d_w,
        d_n,
        active=list(range(d_n)),
        colors=_dense_colors([1]),
    )
    _feature_row(
        ax,
        d_cx_left,
        row_y_bot,
        d_w,
        d_n,
        active=list(range(d_n)),
        colors=_dense_colors([5]),
    )
    # Red outline on the shared cell in both rows.
    for row_y in (row_y_top, row_y_bot):
        sx = d_cx_left + shared_dim * (d_w + 0.0015)
        ax.add_patch(
            Rectangle(
                (sx - 0.0025, row_y - d_w / 2 - 0.0025),
                d_w + 0.005,
                d_w + 0.005,
                fill=False,
                edgecolor=C_BAD,
                linewidth=1.0,
                zorder=6,
            )
        )
    # Dashed vertical bracket + "poly-\nsemantic" label.
    sx_center = d_cx_left + shared_dim * (d_w + 0.0015) + d_w / 2
    _arrow(
        ax,
        (sx_center, row_y_top - d_w / 2),
        (sx_center, row_y_bot + d_w / 2),
        color=C_BAD,
        lw=0.6,
        style="-",
        ms=0,
        dashed=True,
    )
    # Label centred on the dashed connector, on a white patch so the line
    # visibly runs through it.
    ax.text(
        sx_center,
        (row_y_top + row_y_bot) / 2 - 0.035,
        "polysemantic",
        fontsize=FS_LOWER,
        ha="center",
        va="center",
        color=C_BAD,
        style="italic",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.6),
        zorder=6,
    )
    # Centred above the dense row, like "sparse z" above the sparse row.
    _text(
        ax,
        d_cx_left + (d_n * d_w) / 2,
        row_y_top + d_w / 2 + 0.065,
        r"dense $\mathbf{e}$",
        size=FS_LOWER,
        color=C_DENSE,
    )

    # Sparse vectors (horizontal rows of 12 cells) — aligned under z.
    s_n = 12
    s_w = 0.030
    # Right-aligned with the end of the pipeline so the lower half fills the panel width.
    s_cx_left = cx_ehat + 0.05 - (s_n * s_w + (s_n - 1) * 0.0015)
    water_sae_dim = 2
    forest_sae_dim = 9
    _feature_row(
        ax,
        s_cx_left,
        row_y_top,
        s_w,
        s_n,
        active=[water_sae_dim],
        colors={water_sae_dim: C_WATER},
    )
    _feature_row(
        ax,
        s_cx_left,
        row_y_bot,
        s_w,
        s_n,
        active=[forest_sae_dim],
        colors={forest_sae_dim: C_FOREST},
    )
    _text(
        ax, s_cx_left + (s_n * s_w) / 2, row_y_top + s_w / 2 + 0.065, r"sparse $\mathbf{z}$", size=FS_LOWER, color=C_SAE
    )

    # Curved blue + green arrows from the polysemantic dense cell (red
    # outline) to the sparse active cells.
    src_xy_top = (sx_center, row_y_top + d_w / 2 + 0.003)
    src_xy_bot = (sx_center, row_y_bot - d_w / 2 - 0.003)
    water_cell_x = s_cx_left + water_sae_dim * (s_w + 0.0015) + s_w / 2
    forest_cell_x = s_cx_left + forest_sae_dim * (s_w + 0.0015) + s_w / 2
    dst_xy_top = (water_cell_x, row_y_top + s_w / 2)
    dst_xy_bot = (forest_cell_x, row_y_bot - d_w / 2)
    # Water arrow arcs up and over; forest arcs down and under.
    _arrow(ax, src_xy_top, dst_xy_top, color=C_WATER, lw=0.9, ms=6, rad=-0.2)
    _arrow(ax, src_xy_bot, dst_xy_bot, color=C_FOREST, lw=0.9, ms=6, rad=0.2)

    # Green dashed line connecting the two monosemantic active cells,
    # with "mono-\nsemantic" label on an opaque white patch so the line
    # stays invisible behind the text.
    water_cell_top = (water_cell_x, row_y_top - s_w / 2)
    forest_cell_top = (forest_cell_x, row_y_bot + s_w / 2)
    ax.plot(
        [water_cell_top[0], forest_cell_top[0]],
        [water_cell_top[1], forest_cell_top[1]],
        color=C_OK,
        linestyle=(0, (2.0, 1.4)),
        lw=0.6,
        zorder=4,
    )
    mono_x = (water_cell_top[0] + forest_cell_top[0]) / 2
    mono_y = (row_y_top + row_y_bot) / 2
    ax.text(
        mono_x,
        mono_y + 0.03,
        "monosemantic",
        fontsize=FS_LOWER,
        ha="center",
        va="center",
        color=C_OK,
        style="italic",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.6),
        zorder=6,
    )


# ---------------------------------------------------------------------------
# Panel (b)
# ---------------------------------------------------------------------------


def _panel_b(ax):
    """Evaluation-protocol schematic.

    Two-row layout:
      * Upper block: three boxes, one per criterion (reconstruction at
        matched sparsity, concept alignment, spatial scale), each compared
        with a frozen-decoder baseline.
      * Lower row: zoom-in on the concept-alignment criterion — the
        reference registry on the left feeds a per-feature × class
        AUROC matrix on the right; that matrix is what produces the
        single ``concepts uniquely covered'' number on the
        monosemanticity axis.
    """
    ax.set_xlim(0, 1.8)
    ax.set_ylim(0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # ============== UPPER BLOCK: four scoring-axis boxes, 2 x 2 ==========
    box_h = 0.18
    row_gap = 0.016
    box_y_top = 0.97
    beige_cx = 0.56  # centre of the beige fill (x-range -0.04 .. 1.16)
    box_gap = 0.016
    row_w = 1.18
    box_w = (row_w - box_gap) / 2
    x_left = beige_cx - row_w / 2
    x_right = beige_cx + row_w / 2
    box_left = [x_left, x_left + box_w + box_gap] * 2
    box_top = [box_y_top] * 2 + [box_y_top - box_h - row_gap] * 2
    box_y_bot = box_y_top - 2 * box_h - row_gap  # bottom of the lower row
    axis_centers = [lx + box_w / 2 for lx in box_left]
    icon_w = 0.20
    icon_cx = [lx + 0.04 + icon_w / 2 for lx in box_left]
    icon_ys = [t - 0.125 for t in box_top]

    # Three criteria: reconstruction at matched sparsity spans the upper row
    # (slots 0+1); concept alignment (slot 2, tied to the lower zoom) and
    # spatial scale (slot 3) share the lower row.
    box_w_eff = {0: row_w, 2: box_w, 3: box_w}
    for idx in (0, 2, 3):
        is_focus = idx == 2
        ax.add_patch(
            Rectangle(
                (box_left[idx], box_top[idx] - box_h),
                box_w_eff[idx],
                box_h,
                facecolor="white",
                edgecolor=C_SAE if is_focus else C_BOX,
                linewidth=1.4 if is_focus else 0.7,
                zorder=3,
            )
        )

    # The baseline every criterion is compared with: one grey strip under the boxes.
    strip_h = 0.068
    strip_y_top = box_y_bot - 0.014
    strip_y_bot = strip_y_top - strip_h
    ax.add_patch(
        Rectangle(
            (x_left, strip_y_bot),
            x_right - x_left,
            strip_h,
            facecolor=C_NULL_FILL,
            edgecolor=C_NULL,
            linewidth=0.5,
            zorder=2.8,
        )
    )
    _text(
        ax,
        x_left + 0.03,
        (strip_y_top + strip_y_bot) / 2 - 0.002,
        "each vs. a frozen-decoder baseline",
        size=FS_ANNOTATION,
        color=C_NULL_TEXT,
        ha="left",
    )

    AXIS_TITLES = {0: "reconstruction at matched sparsity", 2: "concept alignment", 3: "spatial scale"}
    AXIS_METRICS = {2: "AUROC", 3: "variogram"}
    for idx in (0, 2, 3):
        is_focus = idx == 2
        cx_title = box_left[idx] + box_w_eff[idx] / 2
        _text(
            ax,
            cx_title,
            box_top[idx] - 0.042,
            AXIS_TITLES[idx],
            size=FS_ANNOTATION,
            color=C_SAE if is_focus else C_BOX,
            ha="center",
        )
        if idx in AXIS_METRICS:
            _text(
                ax,
                icon_cx[idx] + icon_w / 2 + 0.035,
                icon_ys[idx],
                AXIS_METRICS[idx],
                size=FS_ANNOTATION,
                color=C_SOFT,
                ha="left",
                style="italic",
            )
    # Upper row carries two metrics: FVE (left half) at a given L0 (right half).
    _text(
        ax,
        icon_cx[0] + icon_w / 2 + 0.035,
        icon_ys[0],
        "FVE",
        size=FS_ANNOTATION,
        color=C_SOFT,
        ha="left",
        style="italic",
    )
    _text(
        ax,
        icon_cx[1] + icon_w / 2 + 0.035,
        icon_ys[1],
        r"at $L_{0}$",
        size=FS_ANNOTATION,
        color=C_SOFT,
        ha="left",
        style="italic",
    )

    # --- reconstruction: a small bar with a fraction filled ---
    cx0, icon_y = icon_cx[0], icon_ys[0]
    bar_w = icon_w * 0.9
    bar_h = 0.040
    bar_x = cx0 - bar_w / 2
    ax.add_patch(
        Rectangle(
            (bar_x, icon_y - bar_h / 2),
            bar_w * 0.30,
            bar_h,
            facecolor=C_DENSE,
            edgecolor=C_BOX,
            linewidth=0.5,
            zorder=3.2,
        )
    )
    ax.add_patch(
        Rectangle((bar_x, icon_y - bar_h / 2), bar_w, bar_h, fill=False, edgecolor=C_BOX, linewidth=0.5, zorder=3.4)
    )

    # --- matched sparsity: 8 dots, 2 active ---
    cx1, icon_y = icon_cx[1], icon_ys[1]
    n_dot = 8
    dot_w_total = icon_w * 0.95
    dot_step = dot_w_total / (n_dot - 1)
    dot_x0 = cx1 - dot_w_total / 2
    active = {2, 5}
    for k in range(n_dot):
        ax.add_patch(
            Circle(
                (dot_x0 + k * dot_step, icon_y),
                0.011,
                facecolor=C_SAE if k in active else "#dcdcdc",
                edgecolor=C_BOX,
                linewidth=0.4,
                zorder=3.2,
            )
        )

    # --- concept alignment: a row of feature blocks, most highlighted ---
    cx2, icon_y = icon_cx[2], icon_ys[2]
    n_emf = 7
    emf_box = 0.020
    emf_step = emf_box * 1.30
    emf_total = (n_emf - 1) * emf_step
    emf_x0 = cx2 - emf_total / 2
    emf_active = {0, 1, 2, 4, 5, 6}
    for k in range(n_emf):
        ax.add_patch(
            Rectangle(
                (emf_x0 + k * emf_step - emf_box / 2, icon_y - emf_box / 2),
                emf_box,
                emf_box,
                facecolor=C_OK if k in emf_active else "#dcdcdc",
                edgecolor=C_BOX,
                linewidth=0.4,
                zorder=3.2,
            )
        )

    # --- Axis 4: spatial coherence (small 4x3 patch grid) ---
    cx3, icon_y = icon_cx[3], icon_ys[3]
    n_map = 4
    map_w_total = icon_w * 0.62
    map_cell = map_w_total / n_map
    map_x0 = cx3 - map_w_total / 2
    map_y0 = icon_y - 1.5 * map_cell
    rng4 = np.random.default_rng(31)
    base = np.linspace(0.2, 0.95, n_map * 3).reshape(3, n_map)
    for k in range(3):
        for m in range(n_map):
            v = float(np.clip(base[k, m] + 0.05 * rng4.standard_normal(), 0, 1))
            ax.add_patch(
                Rectangle(
                    (map_x0 + m * map_cell, map_y0 + k * map_cell),
                    map_cell,
                    map_cell,
                    facecolor=plt.cm.YlGn(v),
                    edgecolor="white",
                    linewidth=0.12,
                    zorder=3.2,
                )
            )
    ax.add_patch(
        Rectangle((map_x0, map_y0), map_w_total, 3 * map_cell, fill=False, edgecolor=C_BOX, linewidth=0.4, zorder=3.4)
    )

    # ============== LOWER ROW: monosemanticity zoom ======================
    zoom_y_top = 0.47
    zoom_y_bot = 0.015
    zoom_y_c = (zoom_y_top + zoom_y_bot) / 2

    # Connect the highlighted upper box (index 2) to the lower zoom.
    cx_focus = axis_centers[2]
    fcs_x_lo = cx_focus - box_w / 2
    fcs_x_hi = cx_focus + box_w / 2
    ax.plot([fcs_x_lo, x_left + 0.04], [box_y_bot, zoom_y_top + 0.005], color=C_SAE, lw=0.6, ls=":", zorder=2.5)
    ax.plot([fcs_x_hi, x_right - 0.04], [box_y_bot, zoom_y_top + 0.005], color=C_SAE, lw=0.6, ls=":", zorder=2.5)
    ax.add_patch(
        Rectangle(
            (x_left + 0.02, zoom_y_bot),
            x_right - x_left - 0.04,
            zoom_y_top - zoom_y_bot,
            facecolor="white",
            edgecolor=C_SAE,
            linewidth=0.8,
            zorder=2,
        )
    )

    # ----------- LEFT half of zoom: reference-product registry -----------
    reg_x = 0.16
    reg_h = 0.09
    reg_gap = 0.018
    reg_w_half = 0.055
    # Title sits above the first thumb, inside the orange zoom box.
    reg_label_y = zoom_y_top - 0.060
    reg_top = reg_label_y - reg_h / 2 - 0.030

    _text(ax, x_left + 0.05, reg_label_y, "reference registry", size=FS_ANNOTATION, color=C_BOX, ha="left")

    def _cat_thumb(cy, palette, label, n=6):
        rng = np.random.default_rng(int(cy * 1000))
        cell_h = reg_h / n
        cell_v = (2 * reg_w_half) / n
        for i in range(n):
            for j in range(n):
                c = palette[rng.integers(0, len(palette))]
                ax.add_patch(
                    Rectangle(
                        (reg_x - reg_w_half + j * cell_v, cy - reg_h / 2 + i * cell_h),
                        cell_v,
                        cell_h,
                        facecolor=c,
                        edgecolor="white",
                        linewidth=0.1,
                        zorder=3.2,
                    )
                )
        ax.add_patch(
            Rectangle(
                (reg_x - reg_w_half, cy - reg_h / 2),
                2 * reg_w_half,
                reg_h,
                fill=False,
                edgecolor=C_BOX,
                linewidth=0.5,
                zorder=3.5,
            )
        )
        _text(ax, reg_x + reg_w_half + 0.022, cy, label, size=FS_ANNOTATION, color=C_SOFT, ha="left")

    def _cont_thumb(cy, label, cmap=plt.cm.YlGn):
        n = 6
        cell_v = (2 * reg_w_half) / n
        cell_h = reg_h / n
        rng_l = np.random.default_rng(int(cy * 1000) + 7)
        vals = rng_l.random((n, n))
        for i in range(n):
            for j in range(n):
                vals[i, j] = 0.4 * vals[i, j] + 0.6 * (j / (n - 1))
        for i in range(n):
            for j in range(n):
                ax.add_patch(
                    Rectangle(
                        (reg_x - reg_w_half + j * cell_v, cy - reg_h / 2 + i * cell_h),
                        cell_v,
                        cell_h,
                        facecolor=cmap(vals[i, j]),
                        edgecolor="white",
                        linewidth=0.1,
                        zorder=3.2,
                    )
                )
        ax.add_patch(
            Rectangle(
                (reg_x - reg_w_half, cy - reg_h / 2),
                2 * reg_w_half,
                reg_h,
                fill=False,
                edgecolor=C_BOX,
                linewidth=0.5,
                zorder=3.5,
            )
        )
        _text(ax, reg_x + reg_w_half + 0.022, cy, label, size=FS_ANNOTATION, color=C_SOFT, ha="left")

    _cat_thumb(reg_top - 0 * (reg_h + reg_gap), ["#3878a8", "#23582a", "#cda85a", "#888888", "#dccd57"], "land cover")
    _cat_thumb(reg_top - 1 * (reg_h + reg_gap), ["#7d4f24", "#a87544", "#c9a069", "#5a3a1a", "#bda47a"], "geomorphon")
    _cat_thumb(
        reg_top - 2 * (reg_h + reg_gap), ["#1f6f4a", "#c2a85a", "#5a8d3a", "#a85a3a", "#4a78a8"], "soil class", n=3
    )
    _text(
        ax,
        reg_x,
        reg_top - 3 * (reg_h + reg_gap) + reg_h / 2 - 0.012,
        r"$\cdots$",
        size=FS_ANNOTATION + 5,
        color=C_BOX,
        ha="center",
        weight="bold",
    )

    # ----------- RIGHT half of zoom: per-feature × class AUROC matrix ---
    mat_cx, mat_cy = 0.90, zoom_y_c - 0.005
    n_feat, n_cls = 8, 6
    cell_w = 0.030
    mat_w = n_cls * cell_w
    mat_h = n_feat * cell_w
    mat_x0 = mat_cx - mat_w / 2
    mat_y0 = mat_cy - mat_h / 2

    rng = np.random.default_rng(11)
    auc = 0.50 + 0.08 * rng.random((n_feat, n_cls))
    diag_pairs = [(0, 0), (1, 1), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 5)]
    for i, j in diag_pairs:
        auc[i, j] = 0.78 + 0.18 * rng.random()
    cmap = plt.cm.viridis
    auc_norm = (auc - 0.5) / 0.5
    for i in range(n_feat):
        for j in range(n_cls):
            ax.add_patch(
                Rectangle(
                    (mat_x0 + j * cell_w, mat_y0 + i * cell_w),
                    cell_w,
                    cell_w,
                    facecolor=cmap(auc_norm[i, j]),
                    edgecolor="white",
                    linewidth=0.15,
                    zorder=3.2,
                )
            )
    ax.add_patch(
        Rectangle(
            (mat_x0, mat_y0),
            mat_w,
            mat_h,
            fill=False,
            edgecolor=C_BOX,
            linewidth=0.6,
            zorder=3.5,
        )
    )
    # Highlight one cell: a feature uniquely covering one concept.
    ax.add_patch(
        Rectangle(
            (mat_x0 + 3 * cell_w, mat_y0 + 4 * cell_w),
            cell_w,
            cell_w,
            fill=False,
            edgecolor=C_SAE,
            linewidth=1.4,
            zorder=4.5,
        )
    )

    _text(ax, mat_cx, mat_y0 + mat_h + 0.055, "AUROC", size=FS_ANNOTATION, color=C_BOX, ha="center")
    _text(ax, mat_x0 - 0.030, mat_cy, r"feature $j$", size=FS_ANNOTATION, color=C_SOFT, ha="right", rotation=90)
    _text(ax, mat_cx, mat_y0 - 0.040, r"class $c$", size=FS_ANNOTATION, color=C_SOFT, ha="center")

    # Arrow registry → matrix. Short hop: starts at the right edge of
    # the *geomorphon* thumb (row 1, the middle reference) and stops well
    # before the "feature j" axis label so the two never overlap.
    geomorphon_cy = reg_top - 1 * (reg_h + reg_gap)
    arrow_start_x = reg_x + reg_w_half + 0.012 + 0.37
    _arrow(ax, (arrow_start_x, geomorphon_cy), (mat_x0 - 0.115, mat_cy), color=C_SOFT, lw=0.7, ms=5, rad=-0.10)


# ---------------------------------------------------------------------------
# Panel (c)
# ---------------------------------------------------------------------------


def regional_features(pred: Path, regions=REGIONS, min_prevalence: float = MIN_PREVALENCE) -> dict:
    """Prevalence of every test patch and the feature chosen for each region (see the module docstring)."""
    g = zarr.open_group(str(pred), mode="r")
    lon, lat = np.asarray(g["lon"][:]), np.asarray(g["lat"][:])
    prevalence = np.asarray(g["feat_patch_stats"][:, :, 1], dtype=np.float32) / PIXELS  # (F, N)
    total = prevalence.sum(1)
    chosen: list[dict] = []
    for name, lon0, lon1, lat0, lat1 in regions:
        box = (lon >= lon0) & (lon <= lon1) & (lat >= lat0) & (lat <= lat1)
        inside = prevalence[:, box]
        share = np.where(total > 0, inside.sum(1) / np.maximum(total, 1e-12), 0.0)
        share[inside.mean(1) < min_prevalence] = -1.0
        share[[c["feature"] for c in chosen]] = -1.0
        f = int(np.argmax(share))
        if share[f] < 0:
            raise SystemExit(f"no feature reaches a mean prevalence of {min_prevalence} in {name}")
        chosen.append(
            {
                "feature": f,
                "region": name,
                "share_in_box": float(share[f]),
                "mean_prevalence_in_box": float(inside[f].mean()),
                "n_patches_in_box": int(box.sum()),
                "share_of_patches_active": float((prevalence[f] > 0).mean()),
            }
        )
        print(chosen[-1])
    return {"lon": lon, "lat": lat, "prevalence": prevalence[[c["feature"] for c in chosen]], "chosen": chosen}


def paper_regional_features(paper_data: Path, regions=REGIONS) -> dict:
    """The structure of :func:`regional_features` from the camera-ready ``regional_features.npz``.

    The file holds the chosen feature of each region in the order of ``regions``, and their prevalence.
    """
    with np.load(paper_file(paper_data, "regional_features.npz")) as z:
        chosen = [{"feature": int(f), "region": r[0]} for f, r in zip(z["features"], regions, strict=True)]
        return {"lon": z["lon"], "lat": z["lat"], "prevalence": z["prevalence"], "chosen": chosen}


def _panel_c(fig, ax, data: dict) -> None:
    """Two maps of mean prevalence on a 2° grid, with one colour bar."""
    from matplotlib.colors import Normalize

    pos = ax.get_position()
    gs = fig.add_gridspec(2, 1, left=pos.x0, right=pos.x1, bottom=pos.y0 + 0.09, top=pos.y1 - 0.01, hspace=0.12)
    norm = Normalize(0.0, 1.0)
    for i, c in enumerate(data["chosen"]):
        axm, ccrs = nf.make_map_axes(fig, gs[i, 0], projection="robinson")
        nf.draw_background(axm, ccrs, land=True, coastline=True)
        axm.set_extent([-180, 180, -58, 84], crs=ccrs.PlateCarree())
        nf.binned_map(
            data["lon"], data["lat"], data["prevalence"][i], ax=axm, ccrs=ccrs, cmap="magma_r", norm=norm, bin_deg=2.0
        )
        box = {"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.8}
        label = f"f{c['feature']}: {c['region']}"
        axm.text(0.06, 0.10, label, transform=axm.transAxes, fontsize=FS_ANNOTATION, va="bottom", ha="left", bbox=box)
    cax = fig.add_axes([pos.x0 + 0.04 * pos.width, pos.y0 + 0.012, 0.92 * pos.width, 0.022])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="magma_r"), cax=cax, orientation="horizontal")
    cb.set_label("active pixels per area", fontsize=FS_ANNOTATION)
    cb.ax.tick_params(labelsize=FS_ANNOTATION)


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

#: Panel letter, title and the x-range of its data coordinates that the panel uses.
PANELS = (
    ("a", "Sparse features", (-0.02, 1.14)),
    ("b", "Evaluation protocol", (-0.04, 1.16)),
    ("c", "Regional features", (0.02, 0.92)),
)


def build_figure(data: dict) -> plt.Figure:
    """The three panels on a 9.5 in canvas printed at the text width."""
    global FS_ANNOTATION, FS_AXIS_LABEL, FS_LOWER
    _GLOBE_QUEUE.clear()
    width = 9.5
    scale = set_style(width)
    FS_ANNOTATION, FS_AXIS_LABEL, FS_LOWER = 6.2 * scale, 6.8 * scale, 5.8 * scale
    fig = plt.figure(figsize=(width, 3.1))
    plt.rcParams["font.size"] = FS_ANNOTATION
    gs = fig.add_gridspec(
        1,
        3,
        left=0.005,
        right=0.995,
        bottom=0.02,
        top=0.88,
        width_ratios=[x1 - x0 for _, _, (x0, x1) in PANELS],
        wspace=0.02,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axes[1].add_patch(
        Rectangle((0, 0), 1, 1, transform=axes[1].transAxes, facecolor=C_FILL, edgecolor="none", zorder=0)
    )
    _panel_a(axes[0])
    _panel_b(axes[1])
    axes[2].axis("off")
    for ax, (_, _, xlim) in zip(axes, PANELS, strict=True):
        ax.set_xlim(*xlim)
    fig.canvas.draw()
    _process_globes(fig)
    _panel_c(fig, axes[2], data)
    for ax, (letter, title, _) in zip(axes, PANELS, strict=True):
        panel_title(ax, letter, title)
    return fig


def main() -> None:
    ap = parser(__doc__.split("\n")[0])
    ap.add_argument("run", type=Path, nargs="?", help="Matryoshka run directory with predictions/test.zarr")
    ap.add_argument("--name", default="fig1_overview")
    add_paper_data_argument(ap)
    args = ap.parse_args()
    require(ap, args, "run")
    data = paper_regional_features(args.paper_data) if args.paper_data else regional_features(predictions(args.run))
    save(build_figure(data), args.out, args.name)
    write_json({"features": data["chosen"]}, args.out / f"{args.name}.json")


if __name__ == "__main__":
    main()
