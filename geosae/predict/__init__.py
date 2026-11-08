"""Encoding a split into sparse codes and writing them to a prediction store."""

from geosae.predict.sparsify import dense_to_sparse, densify_feature_map
from geosae.predict.writer import SparseWriter, run_prediction

__all__ = ["run_prediction", "SparseWriter", "dense_to_sparse", "densify_feature_map"]
