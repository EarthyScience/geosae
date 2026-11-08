"""Config loading: OmegaConf YAML with layered merge + CLI dotted overrides.

Every experiment is one YAML file (optionally layered: ``-c base.yaml -c
override.yaml``, later wins), given as a path or as the name of a config that
ships with the package (``-c batchtopk``). ``-o a.b=c`` applies dotted overrides on top.
OmegaConf interpolations are resolved, so a value such as
``${oc.env:GEOSAE_STORE,data/majortom210k.zarr}`` reads an environment variable
with a default. The resolved config is a plain dict, consumed by the CLI
subcommands.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def bundled_config_dir() -> Path:
    """Directory of the configs that ship with ``geosae`` (``configs/`` of a source checkout)."""
    package = Path(__file__).resolve().parent
    installed = package / "configs"
    return installed if installed.is_dir() else package.parent / "configs"


def bundled_configs() -> list[str]:
    """Names of the bundled configs, as :func:`resolve_config` accepts them (e.g. ``"batchtopk"``)."""
    root = bundled_config_dir()
    if not root.is_dir():
        return []
    return sorted(str(p.relative_to(root).with_suffix("")) for p in root.rglob("*.yaml"))


def resolve_config(name: str | Path) -> Path:
    """The YAML file a config argument names: a path, or the name of a bundled config.

    A path that exists wins; otherwise ``name`` (with or without ``.yaml``) is
    looked up among :func:`bundled_configs`, so ``geosae train -c batchtopk``
    works without a copy of the repository.

    Raises:
        FileNotFoundError: ``name`` is neither a file nor a bundled config.
    """
    path = Path(name)
    if path.exists():
        return path
    bundled = bundled_config_dir() / (path if path.suffix == ".yaml" else path.with_suffix(".yaml"))
    if bundled.exists():
        return bundled
    raise FileNotFoundError(f"config file not found: {path} (bundled configs: {', '.join(bundled_configs())})")


def load_config(paths: list[str], overrides: list[str] | None = None) -> dict:
    """Merge YAML ``paths`` left-to-right, then apply ``key=value`` ``overrides``.

    Args:
        paths: YAML files or bundled config names (see :func:`resolve_config`),
            merged deeply; later files win.
        overrides: Dotted assignments such as ``train.k=32``, applied last.

    Returns:
        The fully resolved config as a plain dict.

    Raises:
        FileNotFoundError: A path in ``paths`` does not exist.
        ValueError: An override is not of the form ``key=value``.
    """
    cfg = OmegaConf.create({})
    for p in paths:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(str(resolve_config(p))))
    if overrides:
        # OmegaConf silently reads a bare "train.epochs" as "set it to null",
        # which would land in the run dir's reproduction record unnoticed.
        bad = [o for o in overrides if "=" not in o]
        if bad:
            raise ValueError(f"overrides must be key=value, got {bad}")
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return OmegaConf.to_container(cfg, resolve=True)


def save_config(cfg: dict, path: str) -> None:
    """Write a resolved config to ``path`` as YAML, creating parent directories."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(cfg), str(path))


__all__ = ["bundled_config_dir", "bundled_configs", "load_config", "resolve_config", "save_config"]
