# Evaluation Comparison

**Generated**: 2026-03-03 00:39
**Methods**: llada_topp0.9_no_remask, llada_topp0.9_no_remask_guided_alpha_K16_a0.01, llada_topp0.9_no_remask_guided_alpha_K16_a0.05, llada_topp0.9_no_remask_guided_alpha_K16_a0.1, llada_topp0.9_no_remask_guided_alpha_K16_a0.15, llada_topp0.9_no_remask_guided_alpha_K16_a0.3, llada_topp0.9_no_remask_guided_alpha_K16_a0.5

## Configuration

| Parameter | llada_topp0.9_no_remask | llada_topp0.9_no_remask_guided_alpha_K16_a0.01 | llada_topp0.9_no_remask_guided_alpha_K16_a0.05 | llada_topp0.9_no_remask_guided_alpha_K16_a0.1 | llada_topp0.9_no_remask_guided_alpha_K16_a0.15 | llada_topp0.9_no_remask_guided_alpha_K16_a0.3 | llada_topp0.9_no_remask_guided_alpha_K16_a0.5 |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Seeds | [42, 123, 456, 789, 1337] | [42, 123, 456] | [42, 123, 456] | [42, 123, 456] | [42, 123, 456] | [42, 123, 456] | [42, 123, 456] |
| Num samples | 1000 | 200 | 200 | 200 | 200 | 200 | 200 |
| Sampling steps | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Temperature | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Top-p | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 |
| Unmasking mode | llada | llada | llada | llada | llada | llada | llada |
| Remasking | False | False | False | False | False | False | False |
| Remasking strategy | -- | -- | -- | -- | -- | -- | -- |
| Remasking eta | -- | -- | -- | -- | -- | -- | -- |
| Remasking t_switch | -- | -- | -- | -- | -- | -- | -- |
| Checkpoint | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt |
| Guidance K | -- | 16 | 16 | 16 | 16 | 16 | 16 |
| Guidance alpha | -- | 0.01 | 0.05 | 0.1 | 0.15 | 0.3 | 0.5 |
| Reward mode | -- | soft | soft | soft | soft | soft | soft |
| Phi function | -- | linear | linear | linear | linear | linear | linear |
| Num constraints | -- | 4 | 4 | 4 | 4 | 4 | 4 |

## Validity

| Metric | llada_topp0.9_no_remask | llada_topp0.9_no_remask_guided_alpha_K16_a0.01 | llada_topp0.9_no_remask_guided_alpha_K16_a0.05 | llada_topp0.9_no_remask_guided_alpha_K16_a0.1 | llada_topp0.9_no_remask_guided_alpha_K16_a0.15 | llada_topp0.9_no_remask_guided_alpha_K16_a0.3 | llada_topp0.9_no_remask_guided_alpha_K16_a0.5 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Validity rate | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Connected rate | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Valid types rate | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| No MASK rate | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Inside validity | 99.4 +/- 0.3% | 99.8 +/- 0.2% | 99.2 +/- 0.2% | 99.3 +/- 0.2% | 99.7 +/- 0.2% | 99.3 +/- 0.9% | 99.8 +/- 0.2% |

## Coverage

| Metric | llada_topp0.9_no_remask | llada_topp0.9_no_remask_guided_alpha_K16_a0.01 | llada_topp0.9_no_remask_guided_alpha_K16_a0.05 | llada_topp0.9_no_remask_guided_alpha_K16_a0.1 | llada_topp0.9_no_remask_guided_alpha_K16_a0.15 | llada_topp0.9_no_remask_guided_alpha_K16_a0.3 | llada_topp0.9_no_remask_guided_alpha_K16_a0.5 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Diversity | 0.9454 +/- 0.0047 | 0.9050 +/- 0.0163 | 0.9317 +/- 0.0024 | 0.9533 +/- 0.0125 | 0.9717 +/- 0.0047 | 0.9750 +/- 0.0041 | 0.9817 +/- 0.0085 |
| Novelty | 0.9748 +/- 0.0027 | 0.9950 | 0.9950 | 0.9967 +/- 0.0024 | 0.9917 +/- 0.0024 | 0.9883 +/- 0.0024 | 0.9817 +/- 0.0062 |
| Mode coverage (unweighted) | 3.7 +/- 0.2% | 2.1 +/- 0.2% | 1.9 +/- 0.3% | 2.0 +/- 0.1% | 2.0 +/- 0.1% | 2.0 +/- 0.1% | 2.1 +/- 0.1% |
| Mode coverage (weighted) | 69.6 +/- 1.2% | 41.2 +/- 1.6% | 42.7 +/- 2.2% | 60.6 +/- 0.2% | 61.0 +/- 0.3% | 61.7 +/- 0.9% | 61.9 +/- 1.9% |
| Unique archetypes | 28.6000 +/- 1.2000 | 15.0000 +/- 1.6330 | 13.6667 +/- 2.3570 | 14.0000 +/- 0.8165 | 14.6667 +/- 0.4714 | 14.6667 +/- 0.4714 | 15.0000 +/- 0.8165 |

## Priority Metrics

| Metric | llada_topp0.9_no_remask | llada_topp0.9_no_remask_guided_alpha_K16_a0.01 | llada_topp0.9_no_remask_guided_alpha_K16_a0.05 | llada_topp0.9_no_remask_guided_alpha_K16_a0.1 | llada_topp0.9_no_remask_guided_alpha_K16_a0.15 | llada_topp0.9_no_remask_guided_alpha_K16_a0.3 | llada_topp0.9_no_remask_guided_alpha_K16_a0.5 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Mode coverage (weighted) | 69.6 +/- 1.2% | 41.2 +/- 1.6% | 42.7 +/- 2.2% | 60.6 +/- 0.2% | 61.0 +/- 0.3% | 61.7 +/- 0.9% | 61.9 +/- 1.9% |
| Spatial transitivity | 99.9 +/- 0.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.7 +/- 0.2% | 99.8 +/- 0.2% |
| Cond. edge TV (weighted) | 0.4719 +/- 0.0080 | 0.6261 +/- 0.0016 | 0.6043 +/- 0.0068 | 0.5731 +/- 0.0081 | 0.5511 +/- 0.0090 | 0.5078 +/- 0.0033 | 0.4835 +/- 0.0085 |
| Type-cond. degree TV (weighted) | 0.1594 +/- 0.0088 | 0.2034 +/- 0.0116 | 0.2144 +/- 0.0096 | 0.2152 +/- 0.0083 | 0.2091 +/- 0.0083 | 0.1693 +/- 0.0048 | 0.1668 +/- 0.0046 |
| Node TV | 0.1186 +/- 0.0027 | 0.1604 +/- 0.0033 | 0.1606 +/- 0.0032 | 0.1601 +/- 0.0028 | 0.1525 +/- 0.0041 | 0.1335 +/- 0.0041 | 0.1274 +/- 0.0089 |

## Constraint Satisfaction

| Metric | llada_topp0.9_no_remask | llada_topp0.9_no_remask_guided_alpha_K16_a0.01 | llada_topp0.9_no_remask_guided_alpha_K16_a0.05 | llada_topp0.9_no_remask_guided_alpha_K16_a0.1 | llada_topp0.9_no_remask_guided_alpha_K16_a0.15 | llada_topp0.9_no_remask_guided_alpha_K16_a0.3 | llada_topp0.9_no_remask_guided_alpha_K16_a0.5 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Satisfaction (all)** | 13.3 +/- 1.2% | 69.0 +/- 4.1% | 62.2 +/- 4.1% | 53.5 +/- 2.9% | 50.2 +/- 1.6% | 32.7 +/- 2.3% | 23.5 +/- 1.4% |
| Satisfaction: between_2_and_3_bathrooms | 49.2 +/- 1.7% | 93.8 +/- 0.2% | 89.2 +/- 0.6% | 84.5 +/- 1.1% | 80.8 +/- 2.3% | 68.0 +/- 2.9% | 60.8 +/- 4.3% |
| Satisfaction: kitchen_near_living | 91.3 +/- 1.1% | 100.0% | 100.0% | 99.5 +/- 0.4% | 99.2 +/- 0.5% | 97.3 +/- 0.9% | 96.3 +/- 0.5% |
| Satisfaction: no_bath_kitchen | 52.0 +/- 1.1% | 74.7 +/- 4.1% | 71.2 +/- 4.2% | 65.5 +/- 1.6% | 65.2 +/- 1.2% | 57.5 +/- 2.3% | 52.7 +/- 2.7% |
| Satisfaction: one_kitchen | 91.3 +/- 1.0% | 100.0% | 100.0% | 99.5 +/- 0.4% | 99.2 +/- 0.5% | 97.3 +/- 0.9% | 96.3 +/- 0.5% |
| Mean violation: between_2_and_3_bathrooms | 0.5084 +/- 0.0165 | 0.0617 +/- 0.0024 | 0.1083 +/- 0.0062 | 0.1550 +/- 0.0108 | 0.1917 +/- 0.0232 | 0.3200 +/- 0.0294 | 0.3917 +/- 0.0429 |
| Mean violation: kitchen_near_living | 0.0868 +/- 0.0106 | 0 | 0 | 0.005000 +/- 0.004082 | 0.008333 +/- 0.004714 | 0.0267 +/- 0.0094 | 0.0367 +/- 0.0047 |
| Mean violation: no_bath_kitchen | 0.5414 +/- 0.0052 | 0.2633 +/- 0.0450 | 0.3067 +/- 0.0504 | 0.3667 +/- 0.0155 | 0.3700 +/- 0.0178 | 0.4617 +/- 0.0165 | 0.5200 +/- 0.0389 |
| Mean violation: one_kitchen | 0.0866 +/- 0.0104 | 0 | 0 | 0.005000 +/- 0.004082 | 0.008333 +/- 0.004714 | 0.0267 +/- 0.0094 | 0.0367 +/- 0.0047 |
| Mean viol. (failed): between_2_and_3_bathrooms | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| Mean viol. (failed): kitchen_near_living | 1 | 0 | 0 | 0.6667 +/- 0.4714 | 1 | 1 | 1 |
| Mean viol. (failed): no_bath_kitchen | 1.1278 +/- 0.0176 | 1.0379 +/- 0.0102 | 1.0608 +/- 0.0193 | 1.0632 +/- 0.0188 | 1.0620 +/- 0.0297 | 1.0878 +/- 0.0385 | 1.0974 +/- 0.0201 |
| Mean viol. (failed): one_kitchen | 1 | 0 | 0 | 0.6667 +/- 0.4714 | 1 | 1 | 1 |

## Denoising (Model Quality, Seed-Independent)

| Metric | llada_topp0.9_no_remask | llada_topp0.9_no_remask_guided_alpha_K16_a0.01 | llada_topp0.9_no_remask_guided_alpha_K16_a0.05 | llada_topp0.9_no_remask_guided_alpha_K16_a0.1 | llada_topp0.9_no_remask_guided_alpha_K16_a0.15 | llada_topp0.9_no_remask_guided_alpha_K16_a0.3 | llada_topp0.9_no_remask_guided_alpha_K16_a0.5 |
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
