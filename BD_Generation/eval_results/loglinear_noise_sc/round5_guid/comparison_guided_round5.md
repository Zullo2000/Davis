# Evaluation Comparison

**Generated**: 2026-03-06 11:14
**Methods**: llada_topp0.9_no_remask, llada_topp0.9_remdm_confidence_tsw1.0, llada_topp0.9_no_remask_guided_r4soft_K16_a0.01, llada_topp0.9_remdm_confidence_tsw1.0_guided_r4soft_K16_a0.01, llada_topp0.9_remdm_confidence_tsw1.0_guided_r5optA_K16_a0.01, llada_topp0.9_remdm_confidence_tsw1.0_guided_r5optB_K16_a0.01

## Configuration

| Parameter | llada_topp0.9_no_remask | llada_topp0.9_remdm_confidence_tsw1.0 | llada_topp0.9_no_remask_guided_r4soft_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r4soft_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r5optA_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r5optB_K16_a0.01 |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| Seeds | [42, 123, 456, 789, 1337] | [42, 123, 456, 789, 1337] | [42, 123, 456] | [42, 123, 456] | [42, 123, 456] | [42, 123, 456] |
| Num samples | 1000 | 1000 | 200 | 200 | 200 | 200 |
| Sampling steps | 100 | 100 | 100 | 100 | 100 | 100 |
| Temperature | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Top-p | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 |
| Unmasking mode | llada | llada | llada | llada | llada | llada |
| Remasking | False | True | False | True | True | True |
| Remasking strategy | -- | confidence | -- | -- | -- | -- |
| Remasking eta | -- | 0.0 | -- | -- | -- | -- |
| Remasking t_switch | -- | 1.0 | -- | -- | -- | -- |
| Checkpoint | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt |
| Guidance K | -- | -- | 16 | 16 | 16 | 16 |
| Guidance alpha | -- | -- | 0.01 | 0.01 | 0.01 | 0.01 |
| Reward mode | -- | -- | soft | soft | soft | soft |
| Phi function | -- | -- | linear | linear | linear | linear |
| Num constraints | -- | -- | 4 | 4 | 4 | 4 |

## Validity

| Metric | llada_topp0.9_no_remask | llada_topp0.9_remdm_confidence_tsw1.0 | llada_topp0.9_no_remask_guided_r4soft_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r4soft_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r5optA_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r5optB_K16_a0.01 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| Validity rate | 100.0% | 99.7 +/- 0.1% | 100.0% | 100.0% | 99.5 +/- 0.4% | 100.0% |
| Connected rate | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Valid types rate | 100.0% | 99.7 +/- 0.1% | 100.0% | 100.0% | 99.5 +/- 0.4% | 100.0% |
| No MASK rate | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Inside validity | 99.4 +/- 0.3% | 93.3 +/- 0.5% | 99.8 +/- 0.2% | 95.2 +/- 0.8% | 95.3 +/- 0.2% | 97.8 +/- 1.3% |

## Coverage

| Metric | llada_topp0.9_no_remask | llada_topp0.9_remdm_confidence_tsw1.0 | llada_topp0.9_no_remask_guided_r4soft_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r4soft_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r5optA_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r5optB_K16_a0.01 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| Diversity | 0.9454 +/- 0.0047 | 0.9824 +/- 0.0048 | 0.9050 +/- 0.0163 | 0.9917 +/- 0.0062 | 0.9933 +/- 0.0047 | 0.9900 +/- 0.0000 |
| Novelty | 0.9748 +/- 0.0027 | 0.9988 +/- 0.0007 | 0.9950 | 1 | 1 | 1 |
| Mode coverage (unweighted) | 3.7 +/- 0.2% | 8.8 +/- 0.4% | 2.1 +/- 0.2% | 3.9 +/- 0.1% | 3.5 +/- 0.2% | 3.2 +/- 0.3% |
| Mode coverage (weighted) | 69.6 +/- 1.2% | 73.3 +/- 2.3% | 41.2 +/- 1.6% | 50.4 +/- 0.1% | 53.6 +/- 5.2% | 50.4 +/- 8.2% |
| Unique archetypes | 28.6000 +/- 1.2000 | 120.8000 +/- 4.7497 | 15.0000 +/- 1.6330 | 37.0000 +/- 2.9439 | 36.3333 +/- 2.8674 | 30.0000 +/- 2.9439 |

## Priority Metrics

| Metric | llada_topp0.9_no_remask | llada_topp0.9_remdm_confidence_tsw1.0 | llada_topp0.9_no_remask_guided_r4soft_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r4soft_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r5optA_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r5optB_K16_a0.01 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| Mode coverage (weighted) | 69.6 +/- 1.2% | 73.3 +/- 2.3% | 41.2 +/- 1.6% | 50.4 +/- 0.1% | 53.6 +/- 5.2% | 50.4 +/- 8.2% |
| Spatial transitivity | 99.9 +/- 0.0% | 98.7 +/- 0.4% | 100.0% | 98.7 +/- 0.6% | 98.7 +/- 1.0% | 99.3 +/- 0.5% |
| Cond. edge TV (weighted) | 0.4719 +/- 0.0080 | 0.5707 +/- 0.0081 | 0.6261 +/- 0.0016 | 0.6070 +/- 0.0142 | 0.6003 +/- 0.0094 | 0.6043 +/- 0.0025 |
| Type-cond. degree TV (weighted) | 0.1594 +/- 0.0088 | 0.1691 +/- 0.0052 | 0.2034 +/- 0.0116 | 0.1515 +/- 0.0069 | 0.1627 +/- 0.0080 | 0.1553 +/- 0.0068 |
| Node TV | 0.1186 +/- 0.0027 | 0.1988 +/- 0.0052 | 0.1604 +/- 0.0033 | 0.2115 +/- 0.0088 | 0.2203 +/- 0.0034 | 0.2114 +/- 0.0088 |

## Constraint Satisfaction

| Metric | llada_topp0.9_no_remask | llada_topp0.9_remdm_confidence_tsw1.0 | llada_topp0.9_no_remask_guided_r4soft_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r4soft_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r5optA_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r5optB_K16_a0.01 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Satisfaction (all)** | 13.3 +/- 1.2% | 16.7 +/- 1.1% | 69.0 +/- 4.1% | 56.0 +/- 2.3% | 55.0 +/- 1.4% | 56.0 +/- 1.8% |
| Satisfaction: between_2_and_3_bathrooms | 49.2 +/- 1.7% | 58.9 +/- 1.8% | 93.8 +/- 0.2% | 76.5 +/- 2.5% | 79.5 +/- 1.1% | 77.3 +/- 2.3% |
| Satisfaction: kitchen_near_living | 91.3 +/- 1.1% | 92.0 +/- 1.1% | 100.0% | 98.7 +/- 0.5% | 99.5 +/- 0.7% | 99.7 +/- 0.5% |
| Satisfaction: no_bath_kitchen | 52.0 +/- 1.1% | 46.6 +/- 1.2% | 74.7 +/- 4.1% | 79.0% | 75.2 +/- 0.5% | 76.3 +/- 0.8% |
| Satisfaction: one_kitchen | 91.3 +/- 1.0% | 81.3 +/- 1.0% | 100.0% | 97.3 +/- 0.6% | 98.5 +/- 1.8% | 99.5 +/- 0.4% |
| Mean violation: between_2_and_3_bathrooms | 0.5084 +/- 0.0165 | 0.4158 +/- 0.0183 | 0.0617 +/- 0.0024 | 0.2350 +/- 0.0255 | 0.2050 +/- 0.0108 | 0.2267 +/- 0.0232 |
| Mean violation: kitchen_near_living | 0.0868 +/- 0.0106 | 0.0802 +/- 0.0108 | 0 | 0.0133 +/- 0.0047 | 0.005000 +/- 0.007071 | 0.003333 +/- 0.004714 |
| Mean violation: no_bath_kitchen | 0.5414 +/- 0.0052 | 0.7274 +/- 0.0171 | 0.2633 +/- 0.0450 | 0.2283 +/- 0.0118 | 0.2633 +/- 0.0118 | 0.2450 +/- 0.0147 |
| Mean violation: one_kitchen | 0.0866 +/- 0.0104 | 0.1942 +/- 0.0116 | 0 | 0.0267 +/- 0.0062 | 0.0150 +/- 0.0178 | 0.005000 +/- 0.004082 |
| Mean viol. (failed): between_2_and_3_bathrooms | 1 | 1.0112 +/- 0.0043 | 1 | 1 | 1 | 1 |
| Mean viol. (failed): kitchen_near_living | 1 | 1 | 0 | 1 | 0.3333 +/- 0.4714 | 0.3333 +/- 0.4714 |
| Mean viol. (failed): no_bath_kitchen | 1.1278 +/- 0.0176 | 1.3620 +/- 0.0297 | 1.0379 +/- 0.0102 | 1.0873 +/- 0.0561 | 1.0599 +/- 0.0270 | 1.0343 +/- 0.0255 |
| Mean viol. (failed): one_kitchen | 1 | 1.0393 +/- 0.0097 | 0 | 1 | 0.6667 +/- 0.4714 | 0.6667 +/- 0.4714 |

## Denoising (Model Quality, Seed-Independent)

| Metric | llada_topp0.9_no_remask | llada_topp0.9_remdm_confidence_tsw1.0 | llada_topp0.9_no_remask_guided_r4soft_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r4soft_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r5optA_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r5optB_K16_a0.01 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| acc_edge@t=0.1 | 0.7924 | 0.8011 | -- | -- | -- | -- |
| acc_edge@t=0.3 | 0.6728 | 0.6739 | -- | -- | -- | -- |
| acc_edge@t=0.5 | 0.5419 | 0.5386 | -- | -- | -- | -- |
| acc_edge@t=0.7 | 0.3961 | 0.3982 | -- | -- | -- | -- |
| acc_edge@t=0.9 | 0.2762 | 0.2732 | -- | -- | -- | -- |
| acc_node@t=0.1 | 0.8253 | 0.8322 | -- | -- | -- | -- |
| acc_node@t=0.3 | 0.7042 | 0.6883 | -- | -- | -- | -- |
| acc_node@t=0.5 | 0.5660 | 0.5558 | -- | -- | -- | -- |
| acc_node@t=0.7 | 0.4160 | 0.4223 | -- | -- | -- | -- |
| acc_node@t=0.9 | 0.2816 | 0.2757 | -- | -- | -- | -- |
| ce_edge@t=0.1 | 0.5952 | 0.5788 | -- | -- | -- | -- |
| ce_edge@t=0.3 | 0.9435 | 0.9407 | -- | -- | -- | -- |
| ce_edge@t=0.5 | 1.3254 | 1.3409 | -- | -- | -- | -- |
| ce_edge@t=0.7 | 1.7861 | 1.7756 | -- | -- | -- | -- |
| ce_edge@t=0.9 | 2.1794 | 2.1861 | -- | -- | -- | -- |
| ce_node@t=0.1 | 0.5415 | 0.5313 | -- | -- | -- | -- |
| ce_node@t=0.3 | 0.8277 | 0.8650 | -- | -- | -- | -- |
| ce_node@t=0.5 | 1.1675 | 1.1742 | -- | -- | -- | -- |
| ce_node@t=0.7 | 1.4838 | 1.4736 | -- | -- | -- | -- |
| ce_node@t=0.9 | 1.7839 | 1.7952 | -- | -- | -- | -- |

---
*Auto-generated by `scripts/compare.py`. Values shown as mean +/- std (population) over N seeds.*
*JS/TV/W1 are the primary distance measures. KL metrics marked "(diag.)" are diagnostic only.*
