# Evaluation Comparison

**Generated**: 2026-03-06 17:16
**Methods**: llada_topp0.9_no_remask, llada_topp0.9_remdm_confidence_tsw1.0, llada_topp0.9_remdm_confidence_tsw1.0_guided_r7decayA_K16_a0.01, llada_topp0.9_remdm_confidence_tsw1.0_guided_r7decayB_K16_a0.01

## Configuration

| Parameter | llada_topp0.9_no_remask | llada_topp0.9_remdm_confidence_tsw1.0 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r7decayA_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r7decayB_K16_a0.01 |
|-----------|:---:|:---:|:---:|:---:|
| Seeds | [42, 123, 456, 789, 1337] | [42, 123, 456, 789, 1337] | [42, 123, 456] | [42, 123, 456] |
| Num samples | 1000 | 1000 | 200 | 200 |
| Sampling steps | 100 | 100 | 100 | 100 |
| Temperature | 0.0 | 0.0 | 0.0 | 0.0 |
| Top-p | 0.9 | 0.9 | 0.9 | 0.9 |
| Unmasking mode | llada | llada | llada | llada |
| Remasking | False | True | True | True |
| Remasking strategy | -- | confidence | -- | -- |
| Remasking eta | -- | 0.0 | -- | -- |
| Remasking t_switch | -- | 1.0 | -- | -- |
| Checkpoint | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt |
| Guidance K | -- | -- | 16 | 16 |
| Guidance alpha | -- | -- | 0.01 | 0.01 |
| Reward mode | -- | -- | soft | soft |
| Phi function | -- | -- | linear | linear |
| Num constraints | -- | -- | 4 | 4 |

## Validity

| Metric | llada_topp0.9_no_remask | llada_topp0.9_remdm_confidence_tsw1.0 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r7decayA_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r7decayB_K16_a0.01 |
|--------|:---:|:---:|:---:|:---:|
| Validity rate | 100.0% | 99.7 +/- 0.1% | 100.0% | 100.0% |
| Connected rate | 100.0% | 100.0% | 100.0% | 100.0% |
| Valid types rate | 100.0% | 99.7 +/- 0.1% | 100.0% | 100.0% |
| No MASK rate | 100.0% | 100.0% | 100.0% | 100.0% |
| Inside validity | 99.4 +/- 0.3% | 93.3 +/- 0.5% | 99.8 +/- 0.2% | 100.0% |

## Coverage

| Metric | llada_topp0.9_no_remask | llada_topp0.9_remdm_confidence_tsw1.0 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r7decayA_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r7decayB_K16_a0.01 |
|--------|:---:|:---:|:---:|:---:|
| Diversity | 0.9454 +/- 0.0047 | 0.9824 +/- 0.0048 | 0.6850 +/- 0.0187 | 0.7233 +/- 0.0125 |
| Novelty | 0.9748 +/- 0.0027 | 0.9988 +/- 0.0007 | 0.9967 +/- 0.0024 | 0.9950 |
| Mode coverage (unweighted) | 3.7 +/- 0.2% | 8.8 +/- 0.4% | 2.0 +/- 0.1% | 2.5% |
| Mode coverage (weighted) | 69.6 +/- 1.2% | 73.3 +/- 2.3% | 41.3 +/- 0.4% | 42.1 +/- 1.0% |
| Unique archetypes | 28.6000 +/- 1.2000 | 120.8000 +/- 4.7497 | 14.6667 +/- 1.2472 | 18.3333 +/- 0.4714 |

## Priority Metrics

| Metric | llada_topp0.9_no_remask | llada_topp0.9_remdm_confidence_tsw1.0 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r7decayA_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r7decayB_K16_a0.01 |
|--------|:---:|:---:|:---:|:---:|
| Mode coverage (weighted) | 69.6 +/- 1.2% | 73.3 +/- 2.3% | 41.3 +/- 0.4% | 42.1 +/- 1.0% |
| Spatial transitivity | 99.9 +/- 0.0% | 98.7 +/- 0.4% | 100.0% | 100.0% |
| Cond. edge TV (weighted) | 0.4719 +/- 0.0080 | 0.5707 +/- 0.0081 | 0.6907 +/- 0.0228 | 0.6771 +/- 0.0086 |
| Type-cond. degree TV (weighted) | 0.1594 +/- 0.0088 | 0.1691 +/- 0.0052 | 0.2229 +/- 0.0233 | 0.2186 +/- 0.0154 |
| Node TV | 0.1186 +/- 0.0027 | 0.1988 +/- 0.0052 | 0.1856 +/- 0.0021 | 0.1834 +/- 0.0033 |

## Constraint Satisfaction

| Metric | llada_topp0.9_no_remask | llada_topp0.9_remdm_confidence_tsw1.0 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r7decayA_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r7decayB_K16_a0.01 |
|--------|:---:|:---:|:---:|:---:|
| **Satisfaction (all)** | 13.3 +/- 1.2% | 16.7 +/- 1.1% | 54.2 +/- 2.2% | 58.2 +/- 2.6% |
| Satisfaction: between_2_and_3_bathrooms | 49.2 +/- 1.7% | 58.9 +/- 1.8% | 89.0 +/- 1.6% | 88.5 +/- 0.7% |
| Satisfaction: kitchen_near_living | 91.3 +/- 1.1% | 92.0 +/- 1.1% | 100.0% | 100.0% |
| Satisfaction: no_bath_kitchen | 52.0 +/- 1.1% | 46.6 +/- 1.2% | 64.2 +/- 3.5% | 69.0 +/- 2.3% |
| Satisfaction: one_kitchen | 91.3 +/- 1.0% | 81.3 +/- 1.0% | 100.0% | 100.0% |
| Mean violation: between_2_and_3_bathrooms | 0.5084 +/- 0.0165 | 0.4158 +/- 0.0183 | 0.1100 +/- 0.0163 | 0.1150 +/- 0.0071 |
| Mean violation: kitchen_near_living | 0.0868 +/- 0.0106 | 0.0802 +/- 0.0108 | 0 | 0 |
| Mean violation: no_bath_kitchen | 0.5414 +/- 0.0052 | 0.7274 +/- 0.0171 | 0.3767 +/- 0.0452 | 0.3300 +/- 0.0216 |
| Mean violation: one_kitchen | 0.0866 +/- 0.0104 | 0.1942 +/- 0.0116 | 0 | 0 |
| Mean viol. (failed): between_2_and_3_bathrooms | 1 | 1.0112 +/- 0.0043 | 1 | 1 |
| Mean viol. (failed): kitchen_near_living | 1 | 1 | 0 | 0 |
| Mean viol. (failed): no_bath_kitchen | 1.1278 +/- 0.0176 | 1.3620 +/- 0.0297 | 1.0490 +/- 0.0273 | 1.0656 +/- 0.0360 |
| Mean viol. (failed): one_kitchen | 1 | 1.0393 +/- 0.0097 | 0 | 0 |

## Denoising (Model Quality, Seed-Independent)

| Metric | llada_topp0.9_no_remask | llada_topp0.9_remdm_confidence_tsw1.0 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r7decayA_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_r7decayB_K16_a0.01 |
|--------|:---:|:---:|:---:|:---:|
| acc_edge@t=0.1 | 0.7924 | 0.8011 | -- | -- |
| acc_edge@t=0.3 | 0.6728 | 0.6739 | -- | -- |
| acc_edge@t=0.5 | 0.5419 | 0.5386 | -- | -- |
| acc_edge@t=0.7 | 0.3961 | 0.3982 | -- | -- |
| acc_edge@t=0.9 | 0.2762 | 0.2732 | -- | -- |
| acc_node@t=0.1 | 0.8253 | 0.8322 | -- | -- |
| acc_node@t=0.3 | 0.7042 | 0.6883 | -- | -- |
| acc_node@t=0.5 | 0.5660 | 0.5558 | -- | -- |
| acc_node@t=0.7 | 0.4160 | 0.4223 | -- | -- |
| acc_node@t=0.9 | 0.2816 | 0.2757 | -- | -- |
| ce_edge@t=0.1 | 0.5952 | 0.5788 | -- | -- |
| ce_edge@t=0.3 | 0.9435 | 0.9407 | -- | -- |
| ce_edge@t=0.5 | 1.3254 | 1.3409 | -- | -- |
| ce_edge@t=0.7 | 1.7861 | 1.7756 | -- | -- |
| ce_edge@t=0.9 | 2.1794 | 2.1861 | -- | -- |
| ce_node@t=0.1 | 0.5415 | 0.5313 | -- | -- |
| ce_node@t=0.3 | 0.8277 | 0.8650 | -- | -- |
| ce_node@t=0.5 | 1.1675 | 1.1742 | -- | -- |
| ce_node@t=0.7 | 1.4838 | 1.4736 | -- | -- |
| ce_node@t=0.9 | 1.7839 | 1.7952 | -- | -- |

---
*Auto-generated by `scripts/compare.py`. Values shown as mean +/- std (population) over N seeds.*
*JS/TV/W1 are the primary distance measures. KL metrics marked "(diag.)" are diagnostic only.*
