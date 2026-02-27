"""Inference-time guidance for bubble diagram generation (SVDD).

This package implements SVDD-style K-candidate reweighting to enforce
architectural constraints during sampling without retraining the denoiser.

Main exports:
    Constraint: ABC for constraint primitives
    ConstraintResult: Evaluation result dataclass
    ExactCount, CountRange, RequireAdj, ForbidAdj: Concrete constraints
    RewardComposer: Combines constraint violations into energy/reward
    compile_constraints: Convert Pydantic specs to Constraint objects
    load_guidance_config: Load constraint config from YAML/JSON
    build_effective_probs: Build per-position distributions from tokens + logits
    build_effective_probs_batch: Batched version for K*B candidates
    hard_decode_x0: Argmax-decode MASK positions for hard reward mode
    guided_sample: SVDD K-candidate reweighting loop
    GuidanceStats: Diagnostics dataclass for guided runs
    calibrate_from_samples: Compute P90 normalizers from decoded graphs
    save_calibration: Save calibration dict to JSON
    load_calibration: Load calibration dict from JSON
"""

from bd_gen.guidance.constraints import (
    Constraint,
    ConstraintResult,
    CountRange,
    ExactCount,
    ForbidAdj,
    RequireAdj,
)
from bd_gen.guidance.reward import RewardComposer
from bd_gen.guidance.constraint_schema import compile_constraints, load_guidance_config
from bd_gen.guidance.soft_violations import (
    build_effective_probs,
    build_effective_probs_batch,
    hard_decode_x0,
)
from bd_gen.guidance.guided_sampler import guided_sample, GuidanceStats
from bd_gen.guidance.calibration import (
    calibrate_from_samples,
    save_calibration,
    load_calibration,
)

__all__ = [
    "Constraint",
    "ConstraintResult",
    "ExactCount",
    "CountRange",
    "RequireAdj",
    "ForbidAdj",
    "RewardComposer",
    "compile_constraints",
    "load_guidance_config",
    "build_effective_probs",
    "build_effective_probs_batch",
    "hard_decode_x0",
    "guided_sample",
    "GuidanceStats",
    "calibrate_from_samples",
    "save_calibration",
    "load_calibration",
]
