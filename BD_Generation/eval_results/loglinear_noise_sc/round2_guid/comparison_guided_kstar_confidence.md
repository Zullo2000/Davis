# Evaluation Comparison

**Generated**: 2026-03-02 19:49
**Methods**: llada_topp0.9_remdm_confidence_tsw1.0, llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K4_a0.1, llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K8_a0.1, llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K10_a0.1, llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K12_a0.1, llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K14_a0.1, llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K16_a0.1, llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K20_a0.1, llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K24_a0.1

## Configuration

| Parameter | llada_topp0.9_remdm_confidence_tsw1.0 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K4_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K8_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K10_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K12_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K14_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K16_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K20_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K24_a0.1 |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Seeds | [42, 123, 456, 789, 1337] | [42, 123] | [42, 123] | [42, 123] | [42, 123] | [42, 123] | [42, 123] | [42, 123] | [42, 123] |
| Num samples | 1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Sampling steps | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Temperature | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Top-p | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 |
| Unmasking mode | llada | llada | llada | llada | llada | llada | llada | llada | llada |
| Remasking | True | True | True | True | True | True | True | True | True |
| Remasking strategy | confidence | -- | -- | -- | -- | -- | -- | -- | -- |
| Remasking eta | 0.0 | -- | -- | -- | -- | -- | -- | -- | -- |
| Remasking t_switch | 1.0 | -- | -- | -- | -- | -- | -- | -- | -- |
| Checkpoint | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt |
| Guidance K | -- | 4 | 8 | 10 | 12 | 14 | 16 | 20 | 24 |
| Guidance alpha | -- | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| Reward mode | -- | soft | soft | soft | soft | soft | soft | soft | soft |
| Phi function | -- | linear | linear | linear | linear | linear | linear | linear | linear |
| Num constraints | -- | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |

## Validity

| Metric | llada_topp0.9_remdm_confidence_tsw1.0 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K4_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K8_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K10_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K12_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K14_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K16_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K20_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K24_a0.1 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Validity rate | 99.7 +/- 0.1% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Connected rate | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Valid types rate | 99.7 +/- 0.1% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| No MASK rate | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Inside validity | 93.3 +/- 0.5% | 95.5 +/- 1.5% | 95.5 +/- 1.5% | 96.0 +/- 3.0% | 96.0 +/- 1.0% | 93.0% | 97.0 +/- 1.0% | 96.5 +/- 0.5% | 96.5 +/- 1.5% |

## Coverage

| Metric | llada_topp0.9_remdm_confidence_tsw1.0 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K4_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K8_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K10_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K12_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K14_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K16_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K20_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K24_a0.1 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Diversity | 0.9824 +/- 0.0048 | 0.9900 | 0.9950 +/- 0.0050 | 0.9950 +/- 0.0050 | 0.9850 +/- 0.0050 | 0.9850 +/- 0.0150 | 1 | 0.9900 +/- 0.0100 | 1 |
| Novelty | 0.9988 +/- 0.0007 | 1 | 1 | 1 | 1 | 1 | 0.9950 +/- 0.0050 | 1 | 1 |
| Mode coverage (unweighted) | 8.8 +/- 0.4% | 3.1 +/- 0.1% | 3.0 +/- 0.2% | 2.9 +/- 0.1% | 2.9 +/- 0.1% | 2.7 +/- 0.3% | 3.1 +/- 0.1% | 2.9 +/- 0.3% | 2.8 +/- 0.3% |
| Mode coverage (weighted) | 73.3 +/- 2.3% | 45.9 +/- 3.9% | 52.2 +/- 9.2% | 40.5 +/- 0.1% | 49.2 +/- 0.1% | 37.0 +/- 2.5% | 45.4 +/- 4.4% | 55.0 +/- 5.5% | 48.9 +/- 0.5% |
| Unique archetypes | 120.8000 +/- 4.7497 | 29.5000 +/- 1.5000 | 27.5000 +/- 1.5000 | 31.5000 +/- 0.5000 | 26 | 26 +/- 3 | 26 +/- 1 | 25.5000 +/- 2.5000 | 25 +/- 2 |

## Priority Metrics

| Metric | llada_topp0.9_remdm_confidence_tsw1.0 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K4_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K8_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K10_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K12_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K14_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K16_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K20_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K24_a0.1 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Mode coverage (weighted) | 73.3 +/- 2.3% | 45.9 +/- 3.9% | 52.2 +/- 9.2% | 40.5 +/- 0.1% | 49.2 +/- 0.1% | 37.0 +/- 2.5% | 45.4 +/- 4.4% | 55.0 +/- 5.5% | 48.9 +/- 0.5% |
| Spatial transitivity | 98.7 +/- 0.4% | 98.0% | 99.0% | 98.5 +/- 0.5% | 100.0% | 100.0% | 99.0% | 100.0% | 98.5 +/- 1.5% |
| Cond. edge TV (weighted) | 0.5707 +/- 0.0081 | 0.6471 +/- 0.0055 | 0.6212 +/- 0.0075 | 0.6777 +/- 0.0175 | 0.6162 +/- 0.0117 | 0.6575 +/- 0.0068 | 0.6448 +/- 0.0020 | 0.6413 +/- 0.0171 | 0.6317 +/- 0.0003 |
| Type-cond. degree TV (weighted) | 0.1691 +/- 0.0052 | 0.1859 +/- 0.0014 | 0.1437 +/- 0.0119 | 0.1944 +/- 0.0275 | 0.1762 +/- 0.0365 | 0.1543 +/- 0.0212 | 0.1700 +/- 0.0168 | 0.1623 +/- 0.0125 | 0.1546 +/- 0.0162 |
| Node TV | 0.1988 +/- 0.0052 | 0.2135 +/- 0.0021 | 0.2125 +/- 0.0005 | 0.2369 +/- 0.0027 | 0.2311 +/- 0.0076 | 0.2169 +/- 0.0025 | 0.2180 +/- 0.0024 | 0.2236 +/- 0.0032 | 0.2270 +/- 0.0010 |

## Constraint Satisfaction

| Metric | llada_topp0.9_remdm_confidence_tsw1.0 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K4_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K8_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K10_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K12_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K14_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K16_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K20_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K24_a0.1 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Satisfaction (all)** | 16.7 +/- 1.1% | 32.5 +/- 2.5% | 39.0 +/- 4.0% | 48.0 +/- 3.0% | 49.0 +/- 1.0% | 53.0 +/- 4.0% | 55.5 +/- 2.5% | 56.0 +/- 4.0% | 61.0 +/- 2.0% |
| Satisfaction: between_2_and_3_bathrooms | 58.9 +/- 1.8% | 67.0 +/- 3.0% | 68.0 +/- 3.0% | 81.5 +/- 3.5% | 77.0 +/- 1.0% | 79.0 +/- 2.0% | 81.5 +/- 0.5% | 79.5 +/- 0.5% | 79.5 +/- 3.5% |
| Satisfaction: kitchen_near_living | 92.0 +/- 1.1% | 97.0 +/- 2.0% | 98.5 +/- 0.5% | 96.5 +/- 0.5% | 99.0% | 96.5 +/- 0.5% | 99.5 +/- 0.5% | 99.0 +/- 1.0% | 99.5 +/- 0.5% |
| Satisfaction: no_bath_kitchen | 46.6 +/- 1.2% | 62.5 +/- 1.5% | 68.0 +/- 3.0% | 66.5 +/- 0.5% | 68.5 +/- 2.5% | 75.0 +/- 4.0% | 74.0 +/- 4.0% | 76.0 +/- 5.0% | 79.5 +/- 1.5% |
| Satisfaction: one_kitchen | 81.3 +/- 1.0% | 96.0 +/- 2.0% | 98.0 +/- 1.0% | 95.5 +/- 0.5% | 98.5 +/- 1.5% | 96.0% | 98.5 +/- 0.5% | 99.5 +/- 0.5% | 99.5 +/- 0.5% |
| Mean violation: between_2_and_3_bathrooms | 0.4158 +/- 0.0183 | 0.3400 +/- 0.0400 | 0.3200 +/- 0.0300 | 0.1900 +/- 0.0300 | 0.2300 +/- 0.0100 | 0.2100 +/- 0.0200 | 0.1850 +/- 0.0050 | 0.2050 +/- 0.0050 | 0.2050 +/- 0.0350 |
| Mean violation: kitchen_near_living | 0.0802 +/- 0.0108 | 0.0300 +/- 0.0200 | 0.0150 +/- 0.0050 | 0.0350 +/- 0.0050 | 0.0100 | 0.0350 +/- 0.0050 | 0.005000 +/- 0.005000 | 0.0100 +/- 0.0100 | 0.005000 +/- 0.005000 |
| Mean violation: no_bath_kitchen | 0.7274 +/- 0.0171 | 0.4150 +/- 0.0050 | 0.3550 +/- 0.0250 | 0.3650 +/- 0.0050 | 0.3450 +/- 0.0150 | 0.2800 +/- 0.0500 | 0.2650 +/- 0.0350 | 0.2600 +/- 0.0400 | 0.2200 +/- 0.0200 |
| Mean violation: one_kitchen | 0.1942 +/- 0.0116 | 0.0400 +/- 0.0200 | 0.0200 +/- 0.0100 | 0.0450 +/- 0.0050 | 0.0150 +/- 0.0150 | 0.0400 | 0.0150 +/- 0.0050 | 0.005000 +/- 0.005000 | 0.005000 +/- 0.005000 |
| Mean viol. (failed): between_2_and_3_bathrooms | 1.0112 +/- 0.0043 | 1.0278 +/- 0.0278 | 1 | 1.0333 +/- 0.0333 | 1 | 1 | 1 | 1 | 1 |
| Mean viol. (failed): kitchen_near_living | 1 | 1 | 1 | 1 | 1 | 1 | 0.5000 +/- 0.5000 | 0.5000 +/- 0.5000 | 0.5000 +/- 0.5000 |
| Mean viol. (failed): no_bath_kitchen | 1.3620 +/- 0.0297 | 1.1090 +/- 0.0577 | 1.1118 +/- 0.0261 | 1.0900 +/- 0.0312 | 1.0984 +/- 0.0396 | 1.1166 +/- 0.0213 | 1.0227 +/- 0.0227 | 1.0962 +/- 0.0617 | 1.0718 +/- 0.0191 |
| Mean viol. (failed): one_kitchen | 1.0393 +/- 0.0097 | 1 | 1 | 1 | 0.5000 +/- 0.5000 | 1 | 1 | 0.5000 +/- 0.5000 | 0.5000 +/- 0.5000 |

## Denoising (Model Quality, Seed-Independent)

| Metric | llada_topp0.9_remdm_confidence_tsw1.0 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K4_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K8_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K10_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K12_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K14_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K16_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K20_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_kstar_K24_a0.1 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| acc_edge@t=0.1 | 0.8011 | -- | -- | -- | -- | -- | -- | -- | -- |
| acc_edge@t=0.3 | 0.6739 | -- | -- | -- | -- | -- | -- | -- | -- |
| acc_edge@t=0.5 | 0.5386 | -- | -- | -- | -- | -- | -- | -- | -- |
| acc_edge@t=0.7 | 0.3982 | -- | -- | -- | -- | -- | -- | -- | -- |
| acc_edge@t=0.9 | 0.2732 | -- | -- | -- | -- | -- | -- | -- | -- |
| acc_node@t=0.1 | 0.8322 | -- | -- | -- | -- | -- | -- | -- | -- |
| acc_node@t=0.3 | 0.6883 | -- | -- | -- | -- | -- | -- | -- | -- |
| acc_node@t=0.5 | 0.5558 | -- | -- | -- | -- | -- | -- | -- | -- |
| acc_node@t=0.7 | 0.4223 | -- | -- | -- | -- | -- | -- | -- | -- |
| acc_node@t=0.9 | 0.2757 | -- | -- | -- | -- | -- | -- | -- | -- |
| ce_edge@t=0.1 | 0.5788 | -- | -- | -- | -- | -- | -- | -- | -- |
| ce_edge@t=0.3 | 0.9407 | -- | -- | -- | -- | -- | -- | -- | -- |
| ce_edge@t=0.5 | 1.3409 | -- | -- | -- | -- | -- | -- | -- | -- |
| ce_edge@t=0.7 | 1.7756 | -- | -- | -- | -- | -- | -- | -- | -- |
| ce_edge@t=0.9 | 2.1861 | -- | -- | -- | -- | -- | -- | -- | -- |
| ce_node@t=0.1 | 0.5313 | -- | -- | -- | -- | -- | -- | -- | -- |
| ce_node@t=0.3 | 0.8650 | -- | -- | -- | -- | -- | -- | -- | -- |
| ce_node@t=0.5 | 1.1742 | -- | -- | -- | -- | -- | -- | -- | -- |
| ce_node@t=0.7 | 1.4736 | -- | -- | -- | -- | -- | -- | -- | -- |
| ce_node@t=0.9 | 1.7952 | -- | -- | -- | -- | -- | -- | -- | -- |

---
*Auto-generated by `scripts/compare.py`. Values shown as mean +/- std (population) over N seeds.*
*JS/TV/W1 are the primary distance measures. KL metrics marked "(diag.)" are diagnostic only.*
