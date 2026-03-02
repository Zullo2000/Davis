# Evaluation Comparison

**Generated**: 2026-02-28 07:22
**Methods**: llada_topp0.9_no_remask, llada_topp0.9_no_remask_guided_basic_K4_a0.1, llada_topp0.9_no_remask_guided_basic_K16_a0.1, llada_topp0.9_no_remask_guided_basic_K4_a1.0, llada_topp0.9_no_remask_guided_basic_K16_a1.0, llada_topp0.9_no_remask_guided_basic_K4_a5.0, llada_topp0.9_no_remask_guided_basic_K16_a5.0

## Configuration

| Parameter | llada_topp0.9_no_remask | llada_topp0.9_no_remask_guided_basic_K4_a0.1 | llada_topp0.9_no_remask_guided_basic_K16_a0.1 | llada_topp0.9_no_remask_guided_basic_K4_a1.0 | llada_topp0.9_no_remask_guided_basic_K16_a1.0 | llada_topp0.9_no_remask_guided_basic_K4_a5.0 | llada_topp0.9_no_remask_guided_basic_K16_a5.0 |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Seeds | [42, 123, 456, 789, 1337] | [42, 123, 456, 789, 1337] | [42, 123, 456, 789, 1337] | [42, 123, 456, 789, 1337] | [42, 123, 456, 789, 1337] | [42, 123, 456, 789, 1337] | [42, 123, 456, 789, 1337] |
| Num samples | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| Sampling steps | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Temperature | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Top-p | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 |
| Unmasking mode | llada | llada | llada | llada | llada | llada | llada |
| Remasking | False | False | False | False | False | False | False |
| Remasking strategy | -- | -- | -- | -- | -- | -- | -- |
| Remasking eta | -- | -- | -- | -- | -- | -- | -- |
| Remasking t_switch | -- | -- | -- | -- | -- | -- | -- |
| Checkpoint | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt |
| Guidance K | -- | 4 | 16 | 4 | 16 | 4 | 16 |
| Guidance alpha | -- | 0.1 | 0.1 | 1.0 | 1.0 | 5.0 | 5.0 |
| Reward mode | -- | soft | soft | soft | soft | soft | soft |
| Phi function | -- | linear | linear | linear | linear | linear | linear |
| Num constraints | -- | 4 | 4 | 4 | 4 | 4 | 4 |

## Validity

| Metric | llada_topp0.9_no_remask | llada_topp0.9_no_remask_guided_basic_K4_a0.1 | llada_topp0.9_no_remask_guided_basic_K16_a0.1 | llada_topp0.9_no_remask_guided_basic_K4_a1.0 | llada_topp0.9_no_remask_guided_basic_K16_a1.0 | llada_topp0.9_no_remask_guided_basic_K4_a5.0 | llada_topp0.9_no_remask_guided_basic_K16_a5.0 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Validity rate | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Connected rate | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Valid types rate | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| No MASK rate | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Inside validity | 99.4 +/- 0.3% | 99.6 +/- 0.1% | 99.8 +/- 0.2% | 99.3 +/- 0.4% | 99.3 +/- 0.1% | 99.4 +/- 0.3% | 99.3 +/- 0.2% |

## Coverage

| Metric | llada_topp0.9_no_remask | llada_topp0.9_no_remask_guided_basic_K4_a0.1 | llada_topp0.9_no_remask_guided_basic_K16_a0.1 | llada_topp0.9_no_remask_guided_basic_K4_a1.0 | llada_topp0.9_no_remask_guided_basic_K16_a1.0 | llada_topp0.9_no_remask_guided_basic_K4_a5.0 | llada_topp0.9_no_remask_guided_basic_K16_a5.0 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Diversity | 0.9454 +/- 0.0047 | 0.9034 +/- 0.0053 | 0.9090 +/- 0.0064 | 0.9376 +/- 0.0071 | 0.9322 +/- 0.0043 | 0.9440 +/- 0.0074 | 0.9356 +/- 0.0033 |
| Novelty | 0.9748 +/- 0.0027 | 0.9850 +/- 0.0027 | 0.9916 +/- 0.0048 | 0.9758 +/- 0.0038 | 0.9712 +/- 0.0057 | 0.9730 +/- 0.0049 | 0.9730 +/- 0.0052 |
| Mode coverage (unweighted) | 3.7 +/- 0.2% | 3.0 +/- 0.2% | 3.1 +/- 0.2% | 3.5 +/- 0.3% | 3.5 +/- 0.3% | 3.6 +/- 0.4% | 3.7 +/- 0.2% |
| Mode coverage (weighted) | 69.6 +/- 1.2% | 68.4 +/- 0.8% | 71.0 +/- 3.2% | 69.1 +/- 0.8% | 69.7 +/- 0.9% | 70.4 +/- 3.4% | 69.6 +/- 1.4% |
| Unique archetypes | 28.6000 +/- 1.2000 | 21.2000 +/- 1.1662 | 22.0000 +/- 1.2649 | 26.4000 +/- 2.0591 | 26.2000 +/- 2.9257 | 27.2000 +/- 3.4871 | 27.6000 +/- 1.2000 |

## Priority Metrics

| Metric | llada_topp0.9_no_remask | llada_topp0.9_no_remask_guided_basic_K4_a0.1 | llada_topp0.9_no_remask_guided_basic_K16_a0.1 | llada_topp0.9_no_remask_guided_basic_K4_a1.0 | llada_topp0.9_no_remask_guided_basic_K16_a1.0 | llada_topp0.9_no_remask_guided_basic_K4_a5.0 | llada_topp0.9_no_remask_guided_basic_K16_a5.0 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Mode coverage (weighted) | 69.6 +/- 1.2% | 68.4 +/- 0.8% | 71.0 +/- 3.2% | 69.1 +/- 0.8% | 69.7 +/- 0.9% | 70.4 +/- 3.4% | 69.6 +/- 1.4% |
| Spatial transitivity | 99.9 +/- 0.0% | 99.9 +/- 0.0% | 99.9 +/- 0.0% | 99.8 +/- 0.1% | 99.9 +/- 0.1% | 99.7 +/- 0.1% | 99.8 +/- 0.1% |
| Cond. edge TV (weighted) | 0.4719 +/- 0.0080 | 0.4866 +/- 0.0095 | 0.5174 +/- 0.0064 | 0.4729 +/- 0.0099 | 0.4770 +/- 0.0067 | 0.4698 +/- 0.0080 | 0.4744 +/- 0.0026 |
| Type-cond. degree TV (weighted) | 0.1594 +/- 0.0088 | 0.1733 +/- 0.0064 | 0.1753 +/- 0.0085 | 0.1566 +/- 0.0081 | 0.1532 +/- 0.0036 | 0.1536 +/- 0.0067 | 0.1532 +/- 0.0048 |
| Node TV | 0.1186 +/- 0.0027 | 0.1027 +/- 0.0027 | 0.1104 +/- 0.0021 | 0.1122 +/- 0.0024 | 0.1122 +/- 0.0036 | 0.1166 +/- 0.0035 | 0.1160 +/- 0.0025 |

## Constraint Satisfaction

| Metric | llada_topp0.9_no_remask | llada_topp0.9_no_remask_guided_basic_K4_a0.1 | llada_topp0.9_no_remask_guided_basic_K16_a0.1 | llada_topp0.9_no_remask_guided_basic_K4_a1.0 | llada_topp0.9_no_remask_guided_basic_K16_a1.0 | llada_topp0.9_no_remask_guided_basic_K4_a5.0 | llada_topp0.9_no_remask_guided_basic_K16_a5.0 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Satisfaction (all)** | 43.3 +/- 1.0% | 68.5 +/- 1.3% | 77.0 +/- 1.8% | 47.0 +/- 1.0% | 48.5 +/- 2.6% | 43.5 +/- 1.2% | 43.7 +/- 1.9% |
| Satisfaction: kitchen_near_living | 91.3 +/- 1.1% | 98.1 +/- 0.3% | 99.5 +/- 0.2% | 94.3 +/- 0.4% | 95.0 +/- 0.6% | 92.0 +/- 0.8% | 92.8 +/- 0.8% |
| Satisfaction: no_bath_kitchen | 52.0 +/- 1.1% | 70.3 +/- 1.3% | 77.5 +/- 1.7% | 52.6 +/- 1.3% | 53.4 +/- 2.2% | 51.4 +/- 1.3% | 50.9 +/- 1.4% |
| Satisfaction: one_kitchen | 91.3 +/- 1.0% | 98.1 +/- 0.3% | 99.5 +/- 0.2% | 94.4 +/- 0.4% | 95.0 +/- 0.7% | 92.1 +/- 0.7% | 92.8 +/- 0.8% |
| Satisfaction: one_living | 100.0 +/- 0.0% | 100.0% | 100.0% | 99.9 +/- 0.1% | 100.0 +/- 0.0% | 100.0 +/- 0.1% | 100.0% |
| Mean violation: kitchen_near_living | 0.0868 +/- 0.0106 | 0.0188 +/- 0.0027 | 0.005200 +/- 0.002227 | 0.0568 +/- 0.0043 | 0.0502 +/- 0.0064 | 0.0798 +/- 0.0080 | 0.0722 +/- 0.0080 |
| Mean violation: no_bath_kitchen | 0.5414 +/- 0.0052 | 0.3172 +/- 0.0150 | 0.2404 +/- 0.0190 | 0.5318 +/- 0.0097 | 0.5180 +/- 0.0200 | 0.5452 +/- 0.0124 | 0.5538 +/- 0.0124 |
| Mean violation: one_kitchen | 0.0866 +/- 0.0104 | 0.0188 +/- 0.0027 | 0.005200 +/- 0.002227 | 0.0562 +/- 0.0038 | 0.0498 +/- 0.0065 | 0.0794 +/- 0.0074 | 0.0722 +/- 0.0080 |
| Mean violation: one_living | 0.000200 +/- 0.000400 | 0 | 0 | 0.000600 +/- 0.000800 | 0.000400 +/- 0.000490 | 0.000400 +/- 0.000800 | 0 |
| Mean viol. (failed): kitchen_near_living | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| Mean viol. (failed): no_bath_kitchen | 1.1278 +/- 0.0176 | 1.0695 +/- 0.0165 | 1.0671 +/- 0.0199 | 1.1228 +/- 0.0167 | 1.1130 +/- 0.0126 | 1.1225 +/- 0.0136 | 1.1281 +/- 0.0103 |
| Mean viol. (failed): one_kitchen | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| Mean viol. (failed): one_living | 0.2000 +/- 0.4000 | 0 | 0 | 0.4000 +/- 0.4899 | 0.4000 +/- 0.4899 | 0.2000 +/- 0.4000 | 0 |

## Denoising (Model Quality, Seed-Independent)

| Metric | llada_topp0.9_no_remask | llada_topp0.9_no_remask_guided_basic_K4_a0.1 | llada_topp0.9_no_remask_guided_basic_K16_a0.1 | llada_topp0.9_no_remask_guided_basic_K4_a1.0 | llada_topp0.9_no_remask_guided_basic_K16_a1.0 | llada_topp0.9_no_remask_guided_basic_K4_a5.0 | llada_topp0.9_no_remask_guided_basic_K16_a5.0 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| acc_edge@t=0.1 | 0.7924 | -- | -- | -- | -- | -- | -- |
| acc_edge@t=0.3 | 0.6728 | -- | -- | -- | -- | -- | -- |
| acc_edge@t=0.5 | 0.5419 | -- | -- | -- | -- | -- | -- |
| acc_edge@t=0.7 | 0.3961 | -- | -- | -- | -- | -- | -- |
| acc_edge@t=0.9 | 0.2762 | -- | -- | -- | -- | -- | -- |
| acc_node@t=0.1 | 0.8253 | -- | -- | -- | -- | -- | -- |
| acc_node@t=0.3 | 0.7042 | -- | -- | -- | -- | -- | -- |
| acc_node@t=0.5 | 0.5660 | -- | -- | -- | -- | -- | -- |
| acc_node@t=0.7 | 0.4160 | -- | -- | -- | -- | -- | -- |
| acc_node@t=0.9 | 0.2816 | -- | -- | -- | -- | -- | -- |
| ce_edge@t=0.1 | 0.5952 | -- | -- | -- | -- | -- | -- |
| ce_edge@t=0.3 | 0.9435 | -- | -- | -- | -- | -- | -- |
| ce_edge@t=0.5 | 1.3254 | -- | -- | -- | -- | -- | -- |
| ce_edge@t=0.7 | 1.7861 | -- | -- | -- | -- | -- | -- |
| ce_edge@t=0.9 | 2.1794 | -- | -- | -- | -- | -- | -- |
| ce_node@t=0.1 | 0.5415 | -- | -- | -- | -- | -- | -- |
| ce_node@t=0.3 | 0.8277 | -- | -- | -- | -- | -- | -- |
| ce_node@t=0.5 | 1.1675 | -- | -- | -- | -- | -- | -- |
| ce_node@t=0.7 | 1.4838 | -- | -- | -- | -- | -- | -- |
| ce_node@t=0.9 | 1.7839 | -- | -- | -- | -- | -- | -- |

---
*Auto-generated by `scripts/compare.py`. Values shown as mean +/- std (population) over N seeds.*
*JS/TV/W1 are the primary distance measures. KL metrics marked "(diag.)" are diagnostic only.*
