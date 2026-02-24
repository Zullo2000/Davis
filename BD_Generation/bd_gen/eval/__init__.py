"""Evaluation: validity checking and generation quality metrics."""

from bd_gen.eval.denoising_eval import denoising_eval, denoising_val_elbo
from bd_gen.eval.metrics import (
    conditional_edge_distances_topN,
    conditional_edge_kl,
    distribution_match,
    diversity,
    edge_present_rate_by_num_rooms,
    graph_structure_mmd,
    inside_validity,
    mode_coverage,
    novelty,
    per_class_accuracy,
    spatial_transitivity,
    spatial_transitivity_by_num_rooms,
    type_conditioned_degree_kl,
    validity_by_num_rooms,
    validity_rate,
)
from bd_gen.eval.validity import check_validity, check_validity_batch

__all__ = [
    "check_validity",
    "check_validity_batch",
    "conditional_edge_distances_topN",
    "denoising_eval",
    "denoising_val_elbo",
    "conditional_edge_kl",
    "distribution_match",
    "diversity",
    "edge_present_rate_by_num_rooms",
    "graph_structure_mmd",
    "inside_validity",
    "mode_coverage",
    "novelty",
    "per_class_accuracy",
    "spatial_transitivity",
    "spatial_transitivity_by_num_rooms",
    "type_conditioned_degree_kl",
    "validity_by_num_rooms",
    "validity_rate",
]
