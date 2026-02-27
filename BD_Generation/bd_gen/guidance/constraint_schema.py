"""Pydantic models for validating constraint configurations from YAML/JSON files.

Provides schema validation for guidance constraint specs and a compilation
function that converts validated specs into executable Constraint objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Union

import json
import yaml

from pydantic import BaseModel, Field, field_validator, model_validator

from bd_gen.data.vocab import NODE_TYPES
from bd_gen.guidance.constraints import (
    Constraint,
    CountRange,
    ExactCount,
    ForbidAdj,
    RequireAdj,
)


# ---------------------------------------------------------------------------
# Constraint spec models
# ---------------------------------------------------------------------------


class ExactCountSpec(BaseModel):
    type: Literal["ExactCount"]
    name: str
    room_type: str
    target: int
    weight: float = 1.0

    @field_validator("room_type")
    @classmethod
    def _validate_room_type(cls, v: str) -> str:
        if v not in NODE_TYPES:
            raise ValueError(f"Unknown room type '{v}'. Valid: {NODE_TYPES}")
        return v

    @field_validator("target")
    @classmethod
    def _validate_target(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"target must be >= 0, got {v}")
        return v


class CountRangeSpec(BaseModel):
    type: Literal["CountRange"]
    name: str
    room_type: str
    lo: int
    hi: int
    weight: float = 1.0

    @field_validator("room_type")
    @classmethod
    def _validate_room_type(cls, v: str) -> str:
        if v not in NODE_TYPES:
            raise ValueError(f"Unknown room type '{v}'. Valid: {NODE_TYPES}")
        return v

    @model_validator(mode="after")
    def _validate_range(self) -> "CountRangeSpec":
        if self.lo < 0:
            raise ValueError(f"lo must be >= 0, got {self.lo}")
        if self.hi < 0:
            raise ValueError(f"hi must be >= 0, got {self.hi}")
        if self.lo > self.hi:
            raise ValueError(f"lo ({self.lo}) must be <= hi ({self.hi})")
        return self


class RequireAdjSpec(BaseModel):
    type: Literal["RequireAdj"]
    name: str
    type_a: str
    type_b: str
    weight: float = 1.0

    @field_validator("type_a", "type_b")
    @classmethod
    def _validate_room_type(cls, v: str) -> str:
        if v not in NODE_TYPES:
            raise ValueError(f"Unknown room type '{v}'. Valid: {NODE_TYPES}")
        return v


class ForbidAdjSpec(BaseModel):
    type: Literal["ForbidAdj"]
    name: str
    type_a: str
    type_b: str
    weight: float = 1.0

    @field_validator("type_a", "type_b")
    @classmethod
    def _validate_room_type(cls, v: str) -> str:
        if v not in NODE_TYPES:
            raise ValueError(f"Unknown room type '{v}'. Valid: {NODE_TYPES}")
        return v


# Discriminated union on the 'type' field
ConstraintSpec = Annotated[
    Union[ExactCountSpec, CountRangeSpec, RequireAdjSpec, ForbidAdjSpec],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Top-level guidance configuration
# ---------------------------------------------------------------------------


class GuidanceConfig(BaseModel):
    constraints: list[ConstraintSpec]
    num_candidates: int = 8  # K
    alpha: float = 1.0  # guidance temperature
    phi: Literal["linear", "quadratic", "log1p"] = "linear"
    reward_mode: Literal["soft", "hard"] = "soft"
    calibration_file: str | None = None


# ---------------------------------------------------------------------------
# Compilation & loading
# ---------------------------------------------------------------------------


def compile_constraints(config: GuidanceConfig) -> list[Constraint]:
    """Convert validated specs into executable Constraint objects.

    Maps room type strings to indices via ``NODE_TYPES.index()``.
    """
    constraints: list[Constraint] = []
    for spec in config.constraints:
        if isinstance(spec, ExactCountSpec):
            constraints.append(
                ExactCount(
                    name=spec.name,
                    room_type_idx=NODE_TYPES.index(spec.room_type),
                    target=spec.target,
                    weight=spec.weight,
                )
            )
        elif isinstance(spec, CountRangeSpec):
            constraints.append(
                CountRange(
                    name=spec.name,
                    room_type_idx=NODE_TYPES.index(spec.room_type),
                    lo=spec.lo,
                    hi=spec.hi,
                    weight=spec.weight,
                )
            )
        elif isinstance(spec, RequireAdjSpec):
            constraints.append(
                RequireAdj(
                    name=spec.name,
                    type_a_idx=NODE_TYPES.index(spec.type_a),
                    type_b_idx=NODE_TYPES.index(spec.type_b),
                    weight=spec.weight,
                )
            )
        elif isinstance(spec, ForbidAdjSpec):
            constraints.append(
                ForbidAdj(
                    name=spec.name,
                    type_a_idx=NODE_TYPES.index(spec.type_a),
                    type_b_idx=NODE_TYPES.index(spec.type_b),
                    weight=spec.weight,
                )
            )
    return constraints


def load_guidance_config(path: str | Path) -> GuidanceConfig:
    """Load and validate guidance config from a JSON or YAML file.

    Detects format by file extension (``.json``, ``.yaml``, ``.yml``).
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(text)
    elif path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")
    return GuidanceConfig(**data)
