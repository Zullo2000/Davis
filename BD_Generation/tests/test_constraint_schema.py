"""Tests for bd_gen.guidance.constraint_schema — constraint config validation.

Covers spec tests 33-39 plus additional validation and round-trip tests.
"""

from __future__ import annotations

import json

import pytest
import yaml
from pathlib import Path
from pydantic import ValidationError

from bd_gen.guidance.constraint_schema import (
    ExactCountSpec,
    CountRangeSpec,
    RequireAdjSpec,
    ForbidAdjSpec,
    GuidanceConfig,
    compile_constraints,
    load_guidance_config,
)
from bd_gen.guidance.constraints import ExactCount, CountRange, RequireAdj, ForbidAdj
from bd_gen.data.vocab import NODE_TYPES


# ---------------------------------------------------------------------------
# Test 33: Valid ExactCountSpec
# ---------------------------------------------------------------------------


class TestValidSpecs:
    """Verify that well-formed specs parse without errors."""

    def test_valid_exact_count_spec(self):
        """Test 33: ExactCountSpec with valid room type and target parses OK."""
        spec = ExactCountSpec(
            type="ExactCount",
            name="one_kitchen",
            room_type="Kitchen",
            target=1,
        )
        assert spec.type == "ExactCount"
        assert spec.name == "one_kitchen"
        assert spec.room_type == "Kitchen"
        assert spec.target == 1
        assert spec.weight == 1.0  # default

    def test_valid_count_range_spec(self):
        """CountRangeSpec with valid lo <= hi parses OK."""
        spec = CountRangeSpec(
            type="CountRange",
            name="bathrooms_1_to_4",
            room_type="Bathroom",
            lo=1,
            hi=4,
        )
        assert spec.type == "CountRange"
        assert spec.name == "bathrooms_1_to_4"
        assert spec.room_type == "Bathroom"
        assert spec.lo == 1
        assert spec.hi == 4
        assert spec.weight == 1.0

    def test_valid_require_adj_spec(self):
        """RequireAdjSpec with valid room types parses OK."""
        spec = RequireAdjSpec(
            type="RequireAdj",
            name="kitchen_near_living",
            type_a="Kitchen",
            type_b="LivingRoom",
        )
        assert spec.type == "RequireAdj"
        assert spec.name == "kitchen_near_living"
        assert spec.type_a == "Kitchen"
        assert spec.type_b == "LivingRoom"
        assert spec.weight == 1.0

    def test_valid_forbid_adj_spec(self):
        """ForbidAdjSpec with valid room types parses OK."""
        spec = ForbidAdjSpec(
            type="ForbidAdj",
            name="no_bath_kitchen",
            type_a="Bathroom",
            type_b="Kitchen",
        )
        assert spec.type == "ForbidAdj"
        assert spec.name == "no_bath_kitchen"
        assert spec.type_a == "Bathroom"
        assert spec.type_b == "Kitchen"
        assert spec.weight == 1.0

    def test_exact_count_target_zero(self):
        """ExactCountSpec with target=0 is valid (forbid a room type)."""
        spec = ExactCountSpec(
            type="ExactCount",
            name="no_storage",
            room_type="Storage",
            target=0,
        )
        assert spec.target == 0

    def test_count_range_equal_lo_hi(self):
        """CountRangeSpec with lo == hi is valid (equivalent to ExactCount)."""
        spec = CountRangeSpec(
            type="CountRange",
            name="exactly_two_balconies",
            room_type="Balcony",
            lo=2,
            hi=2,
        )
        assert spec.lo == 2
        assert spec.hi == 2

    def test_custom_weight(self):
        """Specs accept non-default weight values."""
        spec = ExactCountSpec(
            type="ExactCount",
            name="important_kitchen",
            room_type="Kitchen",
            target=1,
            weight=2.5,
        )
        assert spec.weight == 2.5


# ---------------------------------------------------------------------------
# Test 34: Invalid room type
# ---------------------------------------------------------------------------


class TestInvalidRoomType:
    """Room type validation against NODE_TYPES."""

    def test_invalid_room_type(self):
        """Test 34: Unknown room type raises ValidationError."""
        with pytest.raises(ValidationError, match="Unknown room type"):
            ExactCountSpec(
                type="ExactCount",
                name="bad_room",
                room_type="InvalidRoom",
                target=1,
            )

    def test_invalid_room_type_count_range(self):
        """CountRangeSpec with unknown room type raises ValidationError."""
        with pytest.raises(ValidationError, match="Unknown room type"):
            CountRangeSpec(
                type="CountRange",
                name="bad_range",
                room_type="Dungeon",
                lo=0,
                hi=3,
            )

    def test_invalid_room_type_require_adj_type_a(self):
        """RequireAdjSpec with invalid type_a raises ValidationError."""
        with pytest.raises(ValidationError, match="Unknown room type"):
            RequireAdjSpec(
                type="RequireAdj",
                name="bad_adj",
                type_a="Garage",
                type_b="Kitchen",
            )

    def test_invalid_room_type_require_adj_type_b(self):
        """RequireAdjSpec with invalid type_b raises ValidationError."""
        with pytest.raises(ValidationError, match="Unknown room type"):
            RequireAdjSpec(
                type="RequireAdj",
                name="bad_adj",
                type_a="Kitchen",
                type_b="Garage",
            )

    def test_invalid_room_type_forbid_adj(self):
        """ForbidAdjSpec with invalid room type raises ValidationError."""
        with pytest.raises(ValidationError, match="Unknown room type"):
            ForbidAdjSpec(
                type="ForbidAdj",
                name="bad_forbid",
                type_a="Kitchen",
                type_b="Pool",
            )

    def test_all_node_types_accepted(self):
        """Every entry in NODE_TYPES is accepted as a valid room_type."""
        for room_type in NODE_TYPES:
            spec = ExactCountSpec(
                type="ExactCount",
                name=f"test_{room_type}",
                room_type=room_type,
                target=1,
            )
            assert spec.room_type == room_type


# ---------------------------------------------------------------------------
# Test 35: Unknown constraint type in GuidanceConfig
# ---------------------------------------------------------------------------


class TestUnknownConstraintType:
    """Discriminated union rejects unknown type values."""

    def test_unknown_constraint_type(self):
        """Test 35: GuidanceConfig with type='MaxDistance' raises ValidationError."""
        with pytest.raises(ValidationError):
            GuidanceConfig(
                constraints=[
                    {
                        "type": "MaxDistance",
                        "name": "too_far",
                        "room_type": "Kitchen",
                        "target": 3,
                    }
                ]
            )

    def test_unknown_constraint_type_mixed(self):
        """A list with one valid and one unknown type still fails."""
        with pytest.raises(ValidationError):
            GuidanceConfig(
                constraints=[
                    {
                        "type": "ExactCount",
                        "name": "ok",
                        "room_type": "Kitchen",
                        "target": 1,
                    },
                    {
                        "type": "MinArea",
                        "name": "bad",
                        "room_type": "Kitchen",
                        "target": 20,
                    },
                ]
            )


# ---------------------------------------------------------------------------
# Test 36: Negative target
# ---------------------------------------------------------------------------


class TestNegativeTarget:
    """Negative values for count-based fields are rejected."""

    def test_negative_target(self):
        """Test 36: ExactCountSpec with target=-1 raises ValidationError."""
        with pytest.raises(ValidationError, match="target must be >= 0"):
            ExactCountSpec(
                type="ExactCount",
                name="negative",
                room_type="Kitchen",
                target=-1,
            )

    def test_negative_lo(self):
        """CountRangeSpec with lo=-1 raises ValidationError."""
        with pytest.raises(ValidationError, match="lo must be >= 0"):
            CountRangeSpec(
                type="CountRange",
                name="negative_lo",
                room_type="Bathroom",
                lo=-1,
                hi=3,
            )

    def test_negative_hi(self):
        """CountRangeSpec with hi=-2 raises ValidationError."""
        with pytest.raises(ValidationError, match="hi must be >= 0"):
            CountRangeSpec(
                type="CountRange",
                name="negative_hi",
                room_type="Bathroom",
                lo=0,
                hi=-2,
            )


# ---------------------------------------------------------------------------
# Test 37: lo > hi
# ---------------------------------------------------------------------------


class TestLoGtHi:
    """Range constraint with lo > hi is rejected."""

    def test_lo_gt_hi(self):
        """Test 37: CountRangeSpec with lo=5, hi=3 raises ValidationError."""
        with pytest.raises(ValidationError, match="lo.*must be <= hi"):
            CountRangeSpec(
                type="CountRange",
                name="bad_range",
                room_type="Bathroom",
                lo=5,
                hi=3,
            )


# ---------------------------------------------------------------------------
# Test 38: Compilation round-trip
# ---------------------------------------------------------------------------


class TestCompilationRoundtrip:
    """compile_constraints converts specs into Constraint objects."""

    def test_compilation_roundtrip(self):
        """Test 38: All 4 constraint types compile to correct classes and indices."""
        config = GuidanceConfig(
            constraints=[
                {
                    "type": "ExactCount",
                    "name": "one_kitchen",
                    "room_type": "Kitchen",
                    "target": 1,
                },
                {
                    "type": "CountRange",
                    "name": "bathrooms_1_to_3",
                    "room_type": "Bathroom",
                    "lo": 1,
                    "hi": 3,
                },
                {
                    "type": "RequireAdj",
                    "name": "kitchen_near_living",
                    "type_a": "Kitchen",
                    "type_b": "LivingRoom",
                },
                {
                    "type": "ForbidAdj",
                    "name": "no_bath_kitchen",
                    "type_a": "Bathroom",
                    "type_b": "Kitchen",
                },
            ]
        )

        compiled = compile_constraints(config)
        assert len(compiled) == 4

        # ExactCount: Kitchen -> index 2
        c0 = compiled[0]
        assert isinstance(c0, ExactCount)
        assert c0.name == "one_kitchen"
        assert c0.room_type_idx == NODE_TYPES.index("Kitchen")
        assert c0.room_type_idx == 2
        assert c0.target == 1

        # CountRange: Bathroom -> index 3
        c1 = compiled[1]
        assert isinstance(c1, CountRange)
        assert c1.name == "bathrooms_1_to_3"
        assert c1.room_type_idx == NODE_TYPES.index("Bathroom")
        assert c1.room_type_idx == 3
        assert c1.lo == 1
        assert c1.hi == 3

        # RequireAdj: Kitchen(2) <-> LivingRoom(0)
        c2 = compiled[2]
        assert isinstance(c2, RequireAdj)
        assert c2.name == "kitchen_near_living"
        assert c2.type_a_idx == NODE_TYPES.index("Kitchen")
        assert c2.type_a_idx == 2
        assert c2.type_b_idx == NODE_TYPES.index("LivingRoom")
        assert c2.type_b_idx == 0

        # ForbidAdj: Bathroom(3) <-> Kitchen(2)
        c3 = compiled[3]
        assert isinstance(c3, ForbidAdj)
        assert c3.name == "no_bath_kitchen"
        assert c3.type_a_idx == NODE_TYPES.index("Bathroom")
        assert c3.type_a_idx == 3
        assert c3.type_b_idx == NODE_TYPES.index("Kitchen")
        assert c3.type_b_idx == 2

    def test_compile_preserves_weight(self):
        """Custom weight on spec is preserved in compiled Constraint."""
        config = GuidanceConfig(
            constraints=[
                {
                    "type": "ExactCount",
                    "name": "weighted",
                    "room_type": "Kitchen",
                    "target": 1,
                    "weight": 2.5,
                },
            ]
        )
        compiled = compile_constraints(config)
        assert len(compiled) == 1
        assert compiled[0].weight == 2.5

    def test_compile_empty_constraints(self):
        """An empty constraints list compiles to an empty list."""
        config = GuidanceConfig(constraints=[])
        compiled = compile_constraints(config)
        assert compiled == []


# ---------------------------------------------------------------------------
# Test 39: YAML load
# ---------------------------------------------------------------------------


class TestYamlLoad:
    """Load guidance config from YAML files."""

    def test_yaml_load(self, tmp_path: Path):
        """Test 39: Load example_basic.yaml content -> valid GuidanceConfig."""
        yaml_content = """\
constraints:
  - type: ExactCount
    name: one_kitchen
    room_type: Kitchen
    target: 1
  - type: ExactCount
    name: one_living
    room_type: LivingRoom
    target: 1
  - type: RequireAdj
    name: kitchen_near_living
    type_a: Kitchen
    type_b: LivingRoom
  - type: ForbidAdj
    name: no_bath_kitchen
    type_a: Bathroom
    type_b: Kitchen
num_candidates: 8
alpha: 1.0
phi: linear
"""
        yaml_file = tmp_path / "example_basic.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        config = load_guidance_config(yaml_file)

        assert isinstance(config, GuidanceConfig)
        assert len(config.constraints) == 4
        assert config.num_candidates == 8
        assert config.alpha == 1.0
        assert config.phi == "linear"

        # Verify each constraint type
        assert config.constraints[0].type == "ExactCount"
        assert config.constraints[0].name == "one_kitchen"
        assert config.constraints[0].room_type == "Kitchen"

        assert config.constraints[1].type == "ExactCount"
        assert config.constraints[1].name == "one_living"
        assert config.constraints[1].room_type == "LivingRoom"

        assert config.constraints[2].type == "RequireAdj"
        assert config.constraints[2].name == "kitchen_near_living"

        assert config.constraints[3].type == "ForbidAdj"
        assert config.constraints[3].name == "no_bath_kitchen"

    def test_yaml_load_yml_extension(self, tmp_path: Path):
        """YAML files with .yml extension also load correctly."""
        yaml_content = """\
constraints:
  - type: ExactCount
    name: one_kitchen
    room_type: Kitchen
    target: 1
"""
        yaml_file = tmp_path / "config.yml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        config = load_guidance_config(yaml_file)
        assert len(config.constraints) == 1
        assert config.constraints[0].type == "ExactCount"


# ---------------------------------------------------------------------------
# Additional: JSON load
# ---------------------------------------------------------------------------


class TestJsonLoad:
    """Load guidance config from JSON files."""

    def test_json_load(self, tmp_path: Path):
        """JSON config file loads and validates correctly."""
        data = {
            "constraints": [
                {
                    "type": "ExactCount",
                    "name": "one_kitchen",
                    "room_type": "Kitchen",
                    "target": 1,
                },
                {
                    "type": "CountRange",
                    "name": "bathrooms",
                    "room_type": "Bathroom",
                    "lo": 1,
                    "hi": 3,
                },
            ],
            "num_candidates": 16,
            "alpha": 0.5,
        }
        json_file = tmp_path / "config.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        config = load_guidance_config(json_file)
        assert isinstance(config, GuidanceConfig)
        assert len(config.constraints) == 2
        assert config.num_candidates == 16
        assert config.alpha == 0.5

    def test_unsupported_extension(self, tmp_path: Path):
        """Unsupported file extension raises ValueError."""
        txt_file = tmp_path / "config.txt"
        txt_file.write_text("{}", encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported config format"):
            load_guidance_config(txt_file)


# ---------------------------------------------------------------------------
# Additional: GuidanceConfig defaults
# ---------------------------------------------------------------------------


class TestGuidanceConfigDefaults:
    """Verify default values for GuidanceConfig fields."""

    def test_guidance_config_defaults(self):
        """GuidanceConfig with only constraints uses correct defaults."""
        config = GuidanceConfig(
            constraints=[
                {
                    "type": "ExactCount",
                    "name": "one_kitchen",
                    "room_type": "Kitchen",
                    "target": 1,
                }
            ]
        )
        assert config.num_candidates == 8
        assert config.alpha == 1.0
        assert config.phi == "linear"
        assert config.reward_mode == "soft"
        assert config.calibration_file is None

    def test_guidance_config_override_defaults(self):
        """Explicit values override all defaults."""
        config = GuidanceConfig(
            constraints=[],
            num_candidates=32,
            alpha=2.0,
            phi="quadratic",
            reward_mode="hard",
            calibration_file="/path/to/cal.json",
        )
        assert config.num_candidates == 32
        assert config.alpha == 2.0
        assert config.phi == "quadratic"
        assert config.reward_mode == "hard"
        assert config.calibration_file == "/path/to/cal.json"

    def test_invalid_phi(self):
        """phi must be one of the allowed literals."""
        with pytest.raises(ValidationError):
            GuidanceConfig(
                constraints=[],
                phi="exponential",
            )

    def test_invalid_reward_mode(self):
        """reward_mode must be 'soft' or 'hard'."""
        with pytest.raises(ValidationError):
            GuidanceConfig(
                constraints=[],
                reward_mode="mixed",
            )
