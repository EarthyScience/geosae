"""What a store build will fetch, before it fetches anything.

:func:`build_plan` lists, per product of a build, its source, the credentials
it needs, its licence, how many rows it fills, the uncompressed size of its
store array and the one-time source download it caches. :func:`check_credentials`
says which credential classes are configured on this machine,
:func:`missing_credentials` names the products a build cannot fetch yet, and
:func:`format_plan` renders the plan as the table ``geosae download`` prints in
plan mode. Pure python + numpy: no network, no geospatial stack.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from geosae.data.download.sources import Auth, Kind, get_product


def build_plan(
    products: list[str],
    n_cells: int,
    *,
    reference_rows: tuple[int, int] | None = None,
    rows: np.ndarray | None = None,
    patch_size: int = 64,
) -> list[dict[str, Any]]:
    """One record per product of a build.

    Args:
        products: Registry product keys.
        n_cells: Store rows.
        reference_rows: ``(start, end)`` rows the non-embedding products cover
            (``None``: every row), as in :func:`~geosae.data.download.store.build_store`.
        rows: Rows of this run (``None``: all).
        patch_size: Patch side length in pixels.

    Returns:
        Records with ``key``, ``array``, ``kind``, ``auth``, ``licence``,
        ``redistribution``, ``source``, ``rows``, ``store_gb`` (uncompressed)
        and ``cache_gb`` (one-time source download, ``None`` if streamed).
    """
    requested = np.arange(n_cells) if rows is None else np.unique(np.asarray(rows, dtype=np.int64))
    records = []
    for key in products:
        spec = get_product(key)
        todo = requested
        if spec.kind is not Kind.EMBEDDING and reference_rows is not None:
            todo = requested[(requested >= reference_rows[0]) & (requested < reference_rows[1])]
        bands = int(np.prod(spec.shape[2:])) if len(spec.shape) > 2 else 1
        cell_bytes = np.dtype(spec.dtype).itemsize * patch_size * patch_size * bands
        records.append(
            {
                "key": key,
                "array": spec.array,
                "kind": spec.kind.value,
                "auth": spec.auth.value,
                "licence": spec.licence,
                "redistribution": spec.redistribution.value,
                "source": spec.source,
                "rows": int(todo.size),
                "store_gb": todo.size * cell_bytes / 1e9,
                "cache_gb": spec.cache_gb,
            }
        )
    return records


def check_credentials(home: str | Path | None = None) -> dict[str, str]:
    """Which credential classes this machine is configured for.

    Checks for the files the fetchers read; it does not log in.

    Args:
        home: Home directory to look in (default: the user's).

    Returns:
        ``{auth: status}`` for every :class:`~geosae.data.download.sources.Auth`
        class, ``status`` being ``"ok"`` or what is missing.
    """
    home = Path(home) if home is not None else Path.home()
    ee = home / ".config" / "earthengine" / "credentials"
    netrc = Path(os.environ.get("NETRC", home / ".netrc"))
    has_urs = netrc.exists() and "urs.earthdata.nasa.gov" in netrc.read_text(errors="ignore")
    return {
        Auth.PUBLIC.value: "ok",
        Auth.GEE.value: "ok" if ee.exists() else f"missing {ee} (run `earthengine authenticate`)",
        Auth.EARTHDATA.value: "ok" if has_urs else f"no urs.earthdata.nasa.gov entry in {netrc}",
        Auth.CEDA.value: "not checked",
    }


#: How to configure each credential class, for error messages.
CREDENTIAL_SETUP: dict[str, str] = {
    Auth.GEE.value: "run `earthengine authenticate` (pip install earthengine-api) and pass a Google Earth Engine "
    "project (gee_project= / -o download.gee_project=<project>, or $GEOSAE_GEE_PROJECT)",
    Auth.EARTHDATA.value: "add `machine urs.earthdata.nasa.gov login <user> password <password>` to ~/.netrc "
    "(a free NASA Earthdata Login)",
}


def missing_credentials(products: list[str], credentials: dict[str, str] | None = None) -> dict[str, str]:
    """Products whose credentials are not configured on this machine, and how to set them up.

    Args:
        products: Registry product keys.
        credentials: Output of :func:`check_credentials`; checked here when ``None``.

    Returns:
        ``{product: message}``, empty when every product can be fetched.
    """
    credentials = check_credentials() if credentials is None else credentials
    out = {}
    for key in products:
        auth = get_product(key).auth.value
        status = credentials.get(auth, "ok")
        if status not in ("ok", "not checked"):
            out[key] = f"{status}; {CREDENTIAL_SETUP.get(auth, 'see the product source')}"
    return out


def format_plan(records: list[dict[str, Any]], credentials: dict[str, str] | None = None) -> str:
    """Render a plan as a plain-text table, with totals and the credentials it needs.

    Args:
        records: Output of :func:`build_plan`.
        credentials: Output of :func:`check_credentials`; ``None`` omits the check.

    Returns:
        The table.
    """
    header = f"{'product':<20} {'array':<20} {'auth':<10} {'rows':>8} {'store GB':>9} {'cache GB':>9}  licence"
    lines = [header, "-" * len(header)]
    for r in records:
        cache = "-" if r["cache_gb"] is None else f"{r['cache_gb']:.2f}"
        lines.append(
            f"{r['key']:<20} {r['array']:<20} {r['auth']:<10} {r['rows']:>8} {r['store_gb']:>9.1f} {cache:>9}"
            f"  {r['licence']} ({r['redistribution']})"
        )
    store = sum(r["store_gb"] for r in records)
    cache = sum(r["cache_gb"] or 0.0 for r in records)
    lines.append("-" * len(header))
    lines.append(f"{'total (uncompressed)':<41} {'':>10} {store:>8.1f} {cache:>9.2f}")
    if credentials is not None:
        needed = sorted({r["auth"] for r in records})
        lines.append("")
        lines.extend(f"credentials {auth:<10} {credentials.get(auth, 'unknown')}" for auth in needed)
    return "\n".join(lines)


__all__ = ["CREDENTIAL_SETUP", "build_plan", "check_credentials", "format_plan", "missing_credentials"]
