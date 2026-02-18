"""Tests for eval_results/save_utils.py — V2 format, V1 upgrade, comparison table."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_results.save_utils import (
    _format_mean_std,
    build_comparison_table,
    load_eval_result,
    save_eval_result,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _v2_result(
    method: str = "mdlm_baseline",
    seeds: list[int] | None = None,
    include_js: bool = True,
) -> dict:
    """Build a synthetic V2 result dict."""
    seeds = seeds or [42, 123]
    per_seed = {}
    for s in seeds:
        m = {
            "eval/validity_rate": 0.99 + s * 1e-6,
            "eval/diversity": 0.97 + s * 1e-6,
            "eval/novelty": 0.94 + s * 1e-6,
            "eval/node_kl": 0.05 + s * 1e-6,
            "eval/edge_kl": 0.23 + s * 1e-6,
            "eval/num_rooms_kl": 0.006 + s * 1e-6,
            "eval/mmd_degree": 0.07 + s * 1e-6,
            "eval/mmd_clustering": 0.05 + s * 1e-6,
            "eval/mmd_spectral": 0.005 + s * 1e-6,
            "eval/transitivity_score": 0.995 + s * 1e-6,
            "eval/h_consistent": 0.999 + s * 1e-7,
            "eval/v_consistent": 0.996 + s * 1e-6,
            "eval/connected_rate": 1.0,
            "eval/valid_types_rate": 0.99 + s * 1e-6,
            "eval/no_mask_rate": 1.0,
            "eval/mode_coverage": 0.08 + s * 1e-6,
            "eval/mode_coverage_weighted": 0.70 + s * 1e-6,
            "eval/num_sample_modes": 60.0 + s * 0.01,
            "eval/conditional_edge_kl_weighted": 0.42 + s * 1e-6,
            "eval/degree_kl_per_type_weighted": 0.31 + s * 1e-6,
        }
        if include_js:
            m.update({
                "eval/node_js": 0.013 + s * 1e-6,
                "eval/edge_js": 0.031 + s * 1e-6,
                "eval/node_tv": 0.089 + s * 1e-6,
                "eval/edge_tv": 0.145 + s * 1e-6,
                "eval/rooms_w1": 0.032 + s * 1e-6,
                "eval/conditional_edge_js_weighted": 0.085 + s * 1e-6,
                "eval/conditional_edge_tv_weighted": 0.21 + s * 1e-6,
                "eval/degree_js_per_type_weighted": 0.095 + s * 1e-6,
                "eval/degree_tv_per_type_weighted": 0.22 + s * 1e-6,
            })
        per_seed[s] = m

    # Build summary as mean/std
    all_keys = set()
    for m in per_seed.values():
        all_keys.update(m.keys())

    summary = {}
    for k in sorted(all_keys):
        vals = [per_seed[s][k] for s in seeds if k in per_seed[s]]
        import numpy as np
        arr = np.array(vals, dtype=np.float64)
        summary[k] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}

    return {
        "format_version": 2,
        "method": method,
        "timestamp": "2026-02-18T12:00:00",
        "config": {
            "seeds": seeds,
            "num_samples": 1000,
            "sampling_steps": 100,
            "temperature": 0.0,
            "unmasking_mode": "random",
            "remasking_enabled": method != "mdlm_baseline",
            "remasking_strategy": "cap" if method != "mdlm_baseline" else None,
            "remasking_eta": 0.1 if method != "mdlm_baseline" else None,
            "checkpoint": "checkpoint_final.pt",
        },
        "per_seed": {str(s): per_seed[s] for s in seeds},
        "summary": summary,
        "denoising": {
            "denoise/acc_node@t=0.1": 0.95,
            "denoise/acc_edge@t=0.1": 0.89,
        },
    }


def _v1_result() -> dict:
    """Build a synthetic V1 result dict (old format)."""
    return {
        "method": "mdlm_baseline",
        "timestamp": "2026-02-18T00:41:11.504733",
        "config": {
            "seed": 42,
            "num_samples": 1000,
            "sampling_steps": 100,
            "temperature": 0.0,
            "unmasking_mode": "random",
            "remasking_enabled": False,
            "remasking_strategy": None,
            "remasking_eta": None,
            "checkpoint": "checkpoint_final.pt",
        },
        "metrics": {
            "eval/validity_rate": 0.995,
            "eval/diversity": 0.977,
            "eval/novelty": 0.943,
            "eval/node_kl": 0.0474,
            "eval/edge_kl": 0.2335,
            "eval/num_rooms_kl": 0.0065,
            "eval/connected_rate": 1.0,
            "eval/valid_types_rate": 0.995,
            "eval/no_mask_rate": 1.0,
        },
    }


# ---------------------------------------------------------------------------
# _format_mean_std
# ---------------------------------------------------------------------------

class TestFormatMeanStd:
    def test_percentage(self):
        assert _format_mean_std(0.993, 0.004, is_percentage=True) == "99.3 +/- 0.4%"

    def test_percentage_zero_std(self):
        assert _format_mean_std(0.995, 0.0, is_percentage=True) == "99.5%"

    def test_regular_float(self):
        result = _format_mean_std(0.0667, 0.008)
        assert "0.0667" in result
        assert "+/-" in result

    def test_regular_float_zero_std(self):
        result = _format_mean_std(0.0667, 0.0)
        assert result == "0.0667"
        assert "+/-" not in result

    def test_small_value(self):
        result = _format_mean_std(0.005, 0.001)
        assert "0.005000" in result
        assert "+/-" in result

    def test_small_value_zero_std(self):
        result = _format_mean_std(0.005, 0.0)
        assert result == "0.005000"

    def test_zero(self):
        # 0.0 with std=0.0 is treated as integer-valued
        assert _format_mean_std(0.0, 0.0) == "0"


# ---------------------------------------------------------------------------
# save / load roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoadV2:
    def test_roundtrip(self, tmp_path: Path):
        path = tmp_path / "test.json"
        per_seed = {42: {"eval/validity_rate": 0.99}, 123: {"eval/validity_rate": 0.98}}
        summary = {"eval/validity_rate": {"mean": 0.985, "std": 0.005}}

        save_eval_result(
            path=path,
            method="test_method",
            config_dict={"seeds": [42, 123], "num_samples": 100},
            per_seed_metrics=per_seed,
            summary_metrics=summary,
            denoising_metrics={"denoise/acc@t=0.1": 0.95},
        )

        loaded = load_eval_result(path)
        assert loaded["format_version"] == 2
        assert loaded["method"] == "test_method"
        assert "42" in loaded["per_seed"]
        assert "123" in loaded["per_seed"]
        assert loaded["summary"]["eval/validity_rate"]["mean"] == 0.985
        assert loaded["summary"]["eval/validity_rate"]["std"] == 0.005
        assert loaded["denoising"]["denoise/acc@t=0.1"] == 0.95

    def test_no_denoising(self, tmp_path: Path):
        path = tmp_path / "test.json"
        save_eval_result(
            path=path,
            method="test",
            config_dict={},
            per_seed_metrics={42: {}},
            summary_metrics={},
            denoising_metrics=None,
        )
        loaded = load_eval_result(path)
        assert loaded["denoising"] == {}


# ---------------------------------------------------------------------------
# V1 → V2 upgrade
# ---------------------------------------------------------------------------

class TestLoadV1:
    def test_upgrade_structure(self, tmp_path: Path):
        path = tmp_path / "v1.json"
        path.write_text(json.dumps(_v1_result()))

        loaded = load_eval_result(path)
        assert loaded["format_version"] == 2
        assert loaded["_upgraded_from_v1"] is True
        assert "42" in loaded["per_seed"]
        assert loaded["denoising"] == {}

    def test_upgrade_preserves_metrics(self, tmp_path: Path):
        path = tmp_path / "v1.json"
        v1 = _v1_result()
        path.write_text(json.dumps(v1))

        loaded = load_eval_result(path)
        for key, val in v1["metrics"].items():
            if isinstance(val, (int, float)):
                assert key in loaded["summary"]
                assert loaded["summary"][key]["mean"] == float(val)
                assert loaded["summary"][key]["std"] == 0.0

    def test_upgrade_std_zero(self, tmp_path: Path):
        path = tmp_path / "v1.json"
        path.write_text(json.dumps(_v1_result()))

        loaded = load_eval_result(path)
        for entry in loaded["summary"].values():
            assert entry["std"] == 0.0

    def test_loads_existing_v1_file(self):
        """Load the actual V1 JSON file from eval_results/."""
        path = Path(__file__).resolve().parent.parent / "eval_results" / "mdlm_baseline.json"
        if not path.exists():
            pytest.skip("mdlm_baseline.json not found")

        loaded = load_eval_result(path)
        assert loaded["format_version"] == 2
        assert loaded["method"] == "mdlm_baseline"
        assert "eval/validity_rate" in loaded["summary"]


# ---------------------------------------------------------------------------
# build_comparison_table
# ---------------------------------------------------------------------------

class TestBuildComparisonTable:
    def test_two_v2_methods(self, tmp_path: Path):
        p1 = tmp_path / "baseline.json"
        p2 = tmp_path / "remdm.json"
        p1.write_text(json.dumps(_v2_result("mdlm_baseline")))
        p2.write_text(json.dumps(_v2_result("remdm_cap_eta0.1")))

        md = build_comparison_table([p1, p2])

        assert "# Evaluation Comparison" in md
        assert "mdlm_baseline" in md
        assert "remdm_cap_eta0.1" in md
        # Check all section headers
        assert "## Validity" in md
        assert "## Coverage" in md
        assert "## Distribution (Primary: JS / TV / W1)" in md
        assert "## Graph Structure" in md
        assert "## Conditional" in md
        assert "## Denoising" in md
        # Check that JS/TV/W1 are present and bolded
        assert "**Node JS**" in md
        assert "**Edge JS**" in md
        assert "**Rooms W1**" in md
        # Check delta column
        assert "Delta" in md
        # Check footnotes
        assert "Auto-generated" in md

    def test_mixed_v1_v2(self, tmp_path: Path):
        p1 = tmp_path / "v1.json"
        p2 = tmp_path / "v2.json"
        p1.write_text(json.dumps(_v1_result()))
        p2.write_text(json.dumps(_v2_result("remdm_cap_eta0.1")))

        md = build_comparison_table([p1, p2])

        assert "# Evaluation Comparison" in md
        assert "V1 format" in md  # footnote about V1

    def test_primary_only(self, tmp_path: Path):
        p1 = tmp_path / "baseline.json"
        p2 = tmp_path / "remdm.json"
        p1.write_text(json.dumps(_v2_result("mdlm_baseline")))
        p2.write_text(json.dumps(_v2_result("remdm_cap_eta0.1")))

        md = build_comparison_table([p1, p2], primary_only=True)

        # Primary metrics should be present
        assert "**Node JS**" in md
        # Diagnostic KL metrics should be absent
        assert "Node KL (diag.)" not in md
        assert "Edge KL (diag.)" not in md
        assert "Rooms KL (diag.)" not in md
        assert "Cond. edge KL (wt., diag.)" not in md

    def test_single_method(self, tmp_path: Path):
        p1 = tmp_path / "baseline.json"
        p1.write_text(json.dumps(_v2_result("mdlm_baseline")))

        md = build_comparison_table([p1])

        assert "# Evaluation Comparison" in md
        assert "mdlm_baseline" in md
        # No delta column for single method
        assert "Delta" not in md

    def test_missing_metrics_show_dash(self, tmp_path: Path):
        """V1 result lacks JS/TV/W1; those cells should show dashes."""
        p1 = tmp_path / "v1.json"
        p2 = tmp_path / "v2.json"
        p1.write_text(json.dumps(_v1_result()))
        p2.write_text(json.dumps(_v2_result("remdm_cap_eta0.1")))

        md = build_comparison_table([p1, p2])

        # The distribution section should exist (V2 has JS data)
        assert "## Distribution" in md

    def test_no_results(self):
        md = build_comparison_table([])
        assert md == "No results to compare."

    def test_with_existing_files(self):
        """Integration test with actual eval_results/*.json files."""
        eval_dir = Path(__file__).resolve().parent.parent / "eval_results"
        jsons = sorted(eval_dir.glob("*.json"))
        if len(jsons) < 2:
            pytest.skip("Need at least 2 JSON files in eval_results/")

        md = build_comparison_table(jsons)
        assert "# Evaluation Comparison" in md
        assert "## Validity" in md
