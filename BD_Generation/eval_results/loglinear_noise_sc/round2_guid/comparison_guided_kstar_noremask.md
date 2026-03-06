# Evaluation Comparison

**Generated**: 2026-03-02 18:50
**Methods**: llada_topp0.9_no_remask, llada_topp0.9_no_remask_guided_kstar_K4_a0.1, llada_topp0.9_no_remask_guided_kstar_K8_a0.1, llada_topp0.9_no_remask_guided_kstar_K10_a0.1, llada_topp0.9_no_remask_guided_kstar_K12_a0.1, llada_topp0.9_no_remask_guided_kstar_K14_a0.1, llada_topp0.9_no_remask_guided_kstar_K16_a0.1, llada_topp0.9_no_remask_guided_kstar_K20_a0.1, llada_topp0.9_no_remask_guided_kstar_K24_a0.1

## Configuration

| Parameter | llada_topp0.9_no_remask | llada_topp0.9_no_remask_guided_kstar_K4_a0.1 | llada_topp0.9_no_remask_guided_kstar_K8_a0.1 | llada_topp0.9_no_remask_guided_kstar_K10_a0.1 | llada_topp0.9_no_remask_guided_kstar_K12_a0.1 | llada_topp0.9_no_remask_guided_kstar_K14_a0.1 | llada_topp0.9_no_remask_guided_kstar_K16_a0.1 | llada_topp0.9_no_remask_guided_kstar_K20_a0.1 | llada_topp0.9_no_remask_guided_kstar_K24_a0.1 |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Seeds | [42, 123, 456, 789, 1337] | [42, 123] | [42, 123] | [42, 123] | [42, 123] | [42, 123] | [42, 123] | [42, 123] | [42, 123] |
| Num samples | 1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Sampling steps | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Temperature | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Top-p | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 |
| Unmasking mode | llada | llada | llada | llada | llada | llada | llada | llada | llada |
| Remasking | False | False | False | False | False | False | False | False | False |
| Remasking strategy | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Remasking eta | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Remasking t_switch | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Checkpoint | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt |
| Guidance K | -- | 4 | 8 | 10 | 12 | 14 | 16 | 20 | 24 |
| Guidance alpha | -- | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| Reward mode | -- | soft | soft | soft | soft | soft | soft | soft | soft |
| Phi function | -- | linear | linear | linear | linear | linear | linear | linear | linear |
| Num constraints | -- | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |

## Validity

| Metric | llada_topp0.9_no_remask | llada_topp0.9_no_remask_guided_kstar_K4_a0.1 | llada_topp0.9_no_remask_guided_kstar_K8_a0.1 | llada_topp0.9_no_remask_guided_kstar_K10_a0.1 | llada_topp0.9_no_remask_guided_kstar_K12_a0.1 | llada_topp0.9_no_remask_guided_kstar_K14_a0.1 | llada_topp0.9_no_remask_guided_kstar_K16_a0.1 | llada_topp0.9_no_remask_guided_kstar_K20_a0.1 | llada_topp0.9_no_remask_guided_kstar_K24_a0.1 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Validity rate | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Connected rate | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Valid types rate | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| No MASK rate | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Inside validity | 99.4 +/- 0.3% | 98.5 +/- 0.5% | 100.0% | 99.0% | 99.0% | 99.5 +/- 0.5% | 99.5 +/- 0.5% | 99.5 +/- 0.5% | 99.0 +/- 1.0% |

## Coverage

| Metric | llada_topp0.9_no_remask | llada_topp0.9_no_remask_guided_kstar_K4_a0.1 | llada_topp0.9_no_remask_guided_kstar_K8_a0.1 | llada_topp0.9_no_remask_guided_kstar_K10_a0.1 | llada_topp0.9_no_remask_guided_kstar_K12_a0.1 | llada_topp0.9_no_remask_guided_kstar_K14_a0.1 | llada_topp0.9_no_remask_guided_kstar_K16_a0.1 | llada_topp0.9_no_remask_guided_kstar_K20_a0.1 | llada_topp0.9_no_remask_guided_kstar_K24_a0.1 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Diversity | 0.9454 +/- 0.0047 | 0.9900 | 0.9700 +/- 0.0100 | 0.9800 +/- 0.0100 | 0.9800 | 0.9800 | 0.9650 +/- 0.0150 | 0.9850 +/- 0.0150 | 0.9700 +/- 0.0100 |
| Novelty | 0.9748 +/- 0.0027 | 0.9950 +/- 0.0050 | 0.9900 +/- 0.0100 | 0.9950 +/- 0.0050 | 0.9950 +/- 0.0050 | 1 | 0.9900 | 0.9900 | 0.9900 |
| Mode coverage (unweighted) | 3.7 +/- 0.2% | 1.9 +/- 0.2% | 1.7 +/- 0.1% | 1.7 +/- 0.1% | 1.7 +/- 0.1% | 1.5 +/- 0.1% | 1.6 +/- 0.1% | 1.6 +/- 0.1% | 1.5% |
| Mode coverage (weighted) | 69.6 +/- 1.2% | 60.4 +/- 0.6% | 45.5 +/- 4.3% | 49.2 +/- 0.4% | 50.6 +/- 9.6% | 44.8 +/- 3.9% | 54.5 +/- 5.1% | 50.6 +/- 0.9% | 40.5 +/- 0.5% |
| Unique archetypes | 28.6000 +/- 1.2000 | 13.5000 +/- 1.5000 | 12.5000 +/- 0.5000 | 12.5000 +/- 0.5000 | 12.5000 +/- 0.5000 | 10.5000 +/- 0.5000 | 11.5000 +/- 0.5000 | 11.5000 +/- 0.5000 | 11 |

## Priority Metrics

| Metric | llada_topp0.9_no_remask | llada_topp0.9_no_remask_guided_kstar_K4_a0.1 | llada_topp0.9_no_remask_guided_kstar_K8_a0.1 | llada_topp0.9_no_remask_guided_kstar_K10_a0.1 | llada_topp0.9_no_remask_guided_kstar_K12_a0.1 | llada_topp0.9_no_remask_guided_kstar_K14_a0.1 | llada_topp0.9_no_remask_guided_kstar_K16_a0.1 | llada_topp0.9_no_remask_guided_kstar_K20_a0.1 | llada_topp0.9_no_remask_guided_kstar_K24_a0.1 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Mode coverage (weighted) | 69.6 +/- 1.2% | 60.4 +/- 0.6% | 45.5 +/- 4.3% | 49.2 +/- 0.4% | 50.6 +/- 9.6% | 44.8 +/- 3.9% | 54.5 +/- 5.1% | 50.6 +/- 0.9% | 40.5 +/- 0.5% |
| Spatial transitivity | 99.9 +/- 0.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Cond. edge TV (weighted) | 0.4719 +/- 0.0080 | 0.5330 +/- 0.0180 | 0.5772 +/- 0.0043 | 0.5732 +/- 0.0052 | 0.5697 +/- 0.0048 | 0.5800 +/- 0.0250 | 0.5671 +/- 0.0036 | 0.5607 +/- 0.0154 | 0.5705 +/- 0.0028 |
| Type-cond. degree TV (weighted) | 0.1594 +/- 0.0088 | 0.1761 +/- 0.0031 | 0.1944 +/- 0.0026 | 0.2173 +/- 0.0003 | 0.2006 +/- 0.0013 | 0.2132 +/- 0.0098 | 0.2060 +/- 0.0062 | 0.2071 +/- 0.0114 | 0.2014 +/- 0.0251 |
| Node TV | 0.1186 +/- 0.0027 | 0.1482 +/- 0.0035 | 0.1565 +/- 0.0033 | 0.1617 +/- 0.0004 | 0.1610 +/- 0.0108 | 0.1632 +/- 0.0026 | 0.1596 +/- 0.0123 | 0.1579 +/- 0.0072 | 0.1654 +/- 0.0034 |

## Constraint Satisfaction

| Metric | llada_topp0.9_no_remask | llada_topp0.9_no_remask_guided_kstar_K4_a0.1 | llada_topp0.9_no_remask_guided_kstar_K8_a0.1 | llada_topp0.9_no_remask_guided_kstar_K10_a0.1 | llada_topp0.9_no_remask_guided_kstar_K12_a0.1 | llada_topp0.9_no_remask_guided_kstar_K14_a0.1 | llada_topp0.9_no_remask_guided_kstar_K16_a0.1 | llada_topp0.9_no_remask_guided_kstar_K20_a0.1 | llada_topp0.9_no_remask_guided_kstar_K24_a0.1 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Satisfaction (all)** | 13.3 +/- 1.2% | 39.0 +/- 1.0% | 51.5 +/- 2.5% | 49.5 +/- 1.5% | 56.5 +/- 4.5% | 56.5 +/- 4.5% | 56.5 +/- 2.5% | 53.0 +/- 6.0% | 57.5 +/- 2.5% |
| Satisfaction: between_2_and_3_bathrooms | 49.2 +/- 1.7% | 75.5 +/- 0.5% | 84.5 +/- 2.5% | 84.0 +/- 1.0% | 83.5 +/- 1.5% | 84.0 +/- 4.0% | 85.0% | 84.0 +/- 3.0% | 84.5 +/- 0.5% |
| Satisfaction: kitchen_near_living | 91.3 +/- 1.1% | 97.5 +/- 1.5% | 99.0% | 100.0% | 99.5 +/- 0.5% | 100.0% | 100.0% | 99.0 +/- 1.0% | 99.5 +/- 0.5% |
| Satisfaction: no_bath_kitchen | 52.0 +/- 1.1% | 57.5 +/- 2.5% | 63.0 +/- 1.0% | 61.0 +/- 3.0% | 70.0 +/- 2.0% | 70.5 +/- 1.5% | 68.0 +/- 2.0% | 68.5 +/- 3.5% | 72.0 +/- 3.0% |
| Satisfaction: one_kitchen | 91.3 +/- 1.0% | 97.5 +/- 1.5% | 99.0% | 100.0% | 99.5 +/- 0.5% | 100.0% | 100.0% | 99.0 +/- 1.0% | 99.5 +/- 0.5% |
| Mean violation: between_2_and_3_bathrooms | 0.5084 +/- 0.0165 | 0.2450 +/- 0.0050 | 0.1550 +/- 0.0250 | 0.1600 +/- 0.0100 | 0.1650 +/- 0.0150 | 0.1600 +/- 0.0400 | 0.1500 | 0.1600 +/- 0.0300 | 0.1550 +/- 0.0050 |
| Mean violation: kitchen_near_living | 0.0868 +/- 0.0106 | 0.0250 +/- 0.0150 | 0.0100 | 0 | 0.005000 +/- 0.005000 | 0 | 0 | 0.0100 +/- 0.0100 | 0.005000 +/- 0.005000 |
| Mean violation: no_bath_kitchen | 0.5414 +/- 0.0052 | 0.4600 +/- 0.0200 | 0.3850 +/- 0.0150 | 0.4050 +/- 0.0250 | 0.3050 +/- 0.0250 | 0.3250 +/- 0.0150 | 0.3250 +/- 0.0150 | 0.3450 +/- 0.0250 | 0.2950 +/- 0.0350 |
| Mean violation: one_kitchen | 0.0866 +/- 0.0104 | 0.0250 +/- 0.0150 | 0.0100 | 0 | 0.005000 +/- 0.005000 | 0 | 0 | 0.0100 +/- 0.0100 | 0.005000 +/- 0.005000 |
| Mean viol. (failed): between_2_and_3_bathrooms | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| Mean viol. (failed): kitchen_near_living | 1 | 1 | 1 | 0 | 0.5000 +/- 0.5000 | 0 | 0 | 0.5000 +/- 0.5000 | 0.5000 +/- 0.5000 |
| Mean viol. (failed): no_bath_kitchen | 1.1278 +/- 0.0176 | 1.0833 +/- 0.0167 | 1.0402 +/- 0.0124 | 1.0397 +/- 0.0159 | 1.0156 +/- 0.0156 | 1.1020 +/- 0.0052 | 1.0167 +/- 0.0167 | 1.1000 +/- 0.0429 | 1.0523 +/- 0.0123 |
| Mean viol. (failed): one_kitchen | 1 | 1 | 1 | 0 | 0.5000 +/- 0.5000 | 0 | 0 | 0.5000 +/- 0.5000 | 0.5000 +/- 0.5000 |

## Denoising (Model Quality, Seed-Independent)

| Metric | llada_topp0.9_no_remask | llada_topp0.9_no_remask_guided_kstar_K4_a0.1 | llada_topp0.9_no_remask_guided_kstar_K8_a0.1 | llada_topp0.9_no_remask_guided_kstar_K10_a0.1 | llada_topp0.9_no_remask_guided_kstar_K12_a0.1 | llada_topp0.9_no_remask_guided_kstar_K14_a0.1 | llada_topp0.9_no_remask_guided_kstar_K16_a0.1 | llada_topp0.9_no_remask_guided_kstar_K20_a0.1 | llada_topp0.9_no_remask_guided_kstar_K24_a0.1 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| acc_edge@t=0.1 | 0.7924 | -- | -- | -- | -- | -- | -- | -- | -- |
| acc_edge@t=0.3 | 0.6728 | -- | -- | -- | -- | -- | -- | -- | -- |
| acc_edge@t=0.5 | 0.5419 | -- | -- | -- | -- | -- | -- | -- | -- |
| acc_edge@t=0.7 | 0.3961 | -- | -- | -- | -- | -- | -- | -- | -- |
| acc_edge@t=0.9 | 0.2762 | -- | -- | -- | -- | -- | -- | -- | -- |
| acc_node@t=0.1 | 0.8253 | -- | -- | -- | -- | -- | -- | -- | -- |
| acc_node@t=0.3 | 0.7042 | -- | -- | -- | -- | -- | -- | -- | -- |
| acc_node@t=0.5 | 0.5660 | -- | -- | -- | -- | -- | -- | -- | -- |
| acc_node@t=0.7 | 0.4160 | -- | -- | -- | -- | -- | -- | -- | -- |
| acc_node@t=0.9 | 0.2816 | -- | -- | -- | -- | -- | -- | -- | -- |
| ce_edge@t=0.1 | 0.5952 | -- | -- | -- | -- | -- | -- | -- | -- |
| ce_edge@t=0.3 | 0.9435 | -- | -- | -- | -- | -- | -- | -- | -- |
| ce_edge@t=0.5 | 1.3254 | -- | -- | -- | -- | -- | -- | -- | -- |
| ce_edge@t=0.7 | 1.7861 | -- | -- | -- | -- | -- | -- | -- | -- |
| ce_edge@t=0.9 | 2.1794 | -- | -- | -- | -- | -- | -- | -- | -- |
| ce_node@t=0.1 | 0.5415 | -- | -- | -- | -- | -- | -- | -- | -- |
| ce_node@t=0.3 | 0.8277 | -- | -- | -- | -- | -- | -- | -- | -- |
| ce_node@t=0.5 | 1.1675 | -- | -- | -- | -- | -- | -- | -- | -- |
| ce_node@t=0.7 | 1.4838 | -- | -- | -- | -- | -- | -- | -- | -- |
| ce_node@t=0.9 | 1.7839 | -- | -- | -- | -- | -- | -- | -- | -- |

---
*Auto-generated by `scripts/compare.py`. Values shown as mean +/- std (population) over N seeds.*
*JS/TV/W1 are the primary distance measures. KL metrics marked "(diag.)" are diagnostic only.*
