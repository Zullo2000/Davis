# Evaluation Comparison

**Generated**: 2026-03-03 02:52
**Methods**: llada_topp0.9_remdm_confidence_tsw1.0, llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.01, llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.05, llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.1, llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.15, llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.3, llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.5

## Configuration

| Parameter | llada_topp0.9_remdm_confidence_tsw1.0 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.05 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.15 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.3 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.5 |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Seeds | [42, 123, 456, 789, 1337] | [42, 123, 456] | [42, 123, 456] | [42, 123, 456] | [42, 123, 456] | [42, 123, 456] | [42, 123, 456] |
| Num samples | 1000 | 200 | 200 | 200 | 200 | 200 | 200 |
| Sampling steps | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Temperature | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Top-p | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 |
| Unmasking mode | llada | llada | llada | llada | llada | llada | llada |
| Remasking | True | True | True | True | True | True | True |
| Remasking strategy | confidence | -- | -- | -- | -- | -- | -- |
| Remasking eta | 0.0 | -- | -- | -- | -- | -- | -- |
| Remasking t_switch | 1.0 | -- | -- | -- | -- | -- | -- |
| Checkpoint | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt | checkpoint_final.pt |
| Guidance K | -- | 16 | 16 | 16 | 16 | 16 | 16 |
| Guidance alpha | -- | 0.01 | 0.05 | 0.1 | 0.15 | 0.3 | 0.5 |
| Reward mode | -- | soft | soft | soft | soft | soft | soft |
| Phi function | -- | linear | linear | linear | linear | linear | linear |
| Num constraints | -- | 4 | 4 | 4 | 4 | 4 | 4 |

## Validity

| Metric | llada_topp0.9_remdm_confidence_tsw1.0 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.05 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.15 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.3 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.5 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Validity rate | 99.7 +/- 0.1% | 99.8 +/- 0.2% | 100.0% | 99.8 +/- 0.2% | 99.8 +/- 0.2% | 99.8 +/- 0.2% | 99.8 +/- 0.2% |
| Connected rate | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Valid types rate | 99.7 +/- 0.1% | 99.8 +/- 0.2% | 100.0% | 99.8 +/- 0.2% | 99.8 +/- 0.2% | 99.8 +/- 0.2% | 99.8 +/- 0.2% |
| No MASK rate | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Inside validity | 93.3 +/- 0.5% | 94.8 +/- 1.5% | 96.3 +/- 1.0% | 96.3 +/- 1.0% | 95.0 +/- 4.3% | 92.3 +/- 1.3% | 92.2 +/- 1.2% |

## Coverage

| Metric | llada_topp0.9_remdm_confidence_tsw1.0 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.05 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.15 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.3 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.5 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Diversity | 0.9824 +/- 0.0048 | 0.9883 +/- 0.0062 | 0.9883 +/- 0.0062 | 0.9850 +/- 0.0071 | 0.9833 +/- 0.0094 | 0.9900 +/- 0.0082 | 0.9950 |
| Novelty | 0.9988 +/- 0.0007 | 0.9983 +/- 0.0024 | 1 | 1 | 1 | 1 | 1 |
| Mode coverage (unweighted) | 8.8 +/- 0.4% | 3.3 +/- 0.2% | 3.6 +/- 0.2% | 3.7 +/- 0.2% | 4.0 +/- 0.4% | 4.2 +/- 0.1% | 4.2 +/- 0.3% |
| Mode coverage (weighted) | 73.3 +/- 2.3% | 49.8 +/- 0.1% | 54.1 +/- 8.3% | 51.2 +/- 1.1% | 49.9 +/- 0.8% | 58.9 +/- 6.2% | 52.0 +/- 7.6% |
| Unique archetypes | 120.8000 +/- 4.7497 | 33.6667 +/- 4.4969 | 36.6667 +/- 0.9428 | 34.3333 +/- 0.4714 | 37.6667 +/- 4.4969 | 41.6667 +/- 2.0548 | 41.3333 +/- 1.6997 |

## Priority Metrics

| Metric | llada_topp0.9_remdm_confidence_tsw1.0 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.05 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.15 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.3 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.5 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Mode coverage (weighted) | 73.3 +/- 2.3% | 49.8 +/- 0.1% | 54.1 +/- 8.3% | 51.2 +/- 1.1% | 49.9 +/- 0.8% | 58.9 +/- 6.2% | 52.0 +/- 7.6% |
| Spatial transitivity | 98.7 +/- 0.4% | 99.0 +/- 0.7% | 99.2 +/- 0.5% | 99.0 +/- 0.4% | 99.5% | 98.7 +/- 0.2% | 99.2 +/- 0.2% |
| Cond. edge TV (weighted) | 0.5707 +/- 0.0081 | 0.6074 +/- 0.0145 | 0.6188 +/- 0.0087 | 0.6201 +/- 0.0061 | 0.6346 +/- 0.0063 | 0.6286 +/- 0.0241 | 0.6246 +/- 0.0023 |
| Type-cond. degree TV (weighted) | 0.1691 +/- 0.0052 | 0.1718 +/- 0.0072 | 0.1544 +/- 0.0019 | 0.1501 +/- 0.0119 | 0.1525 +/- 0.0095 | 0.1942 +/- 0.0094 | 0.1617 +/- 0.0158 |
| Node TV | 0.1988 +/- 0.0052 | 0.2126 +/- 0.0030 | 0.2259 +/- 0.0047 | 0.2263 +/- 0.0059 | 0.2221 +/- 0.0031 | 0.2160 +/- 0.0019 | 0.2113 +/- 0.0044 |

## Constraint Satisfaction

| Metric | llada_topp0.9_remdm_confidence_tsw1.0 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.05 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.15 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.3 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.5 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Satisfaction (all)** | 16.7 +/- 1.1% | 55.3 +/- 1.6% | 56.5 +/- 3.3% | 57.3 +/- 4.3% | 55.7 +/- 3.1% | 38.3 +/- 3.0% | 30.8 +/- 2.1% |
| Satisfaction: between_2_and_3_bathrooms | 58.9 +/- 1.8% | 78.2 +/- 2.3% | 81.0 +/- 1.5% | 81.3 +/- 1.5% | 82.0 +/- 0.8% | 74.5 +/- 3.2% | 69.2 +/- 1.6% |
| Satisfaction: kitchen_near_living | 92.0 +/- 1.1% | 98.7 +/- 0.5% | 99.3 +/- 0.2% | 99.2 +/- 0.8% | 98.2 +/- 1.9% | 96.7 +/- 0.6% | 96.7 +/- 0.6% |
| Satisfaction: no_bath_kitchen | 46.6 +/- 1.2% | 77.3 +/- 2.5% | 74.3 +/- 3.5% | 73.5 +/- 2.9% | 73.5 +/- 2.9% | 60.8 +/- 4.5% | 53.5 +/- 1.4% |
| Satisfaction: one_kitchen | 81.3 +/- 1.0% | 97.7 +/- 1.2% | 97.7 +/- 0.8% | 99.0 +/- 0.4% | 97.2 +/- 1.5% | 94.7 +/- 0.8% | 92.7 +/- 0.9% |
| Mean violation: between_2_and_3_bathrooms | 0.4158 +/- 0.0183 | 0.2183 +/- 0.0232 | 0.1900 +/- 0.0147 | 0.1867 +/- 0.0155 | 0.1800 +/- 0.0082 | 0.2550 +/- 0.0319 | 0.3117 +/- 0.0143 |
| Mean violation: kitchen_near_living | 0.0802 +/- 0.0108 | 0.0133 +/- 0.0047 | 0.006667 +/- 0.002357 | 0.008333 +/- 0.008498 | 0.0183 +/- 0.0189 | 0.0333 +/- 0.0062 | 0.0333 +/- 0.0062 |
| Mean violation: no_bath_kitchen | 0.7274 +/- 0.0171 | 0.2467 +/- 0.0301 | 0.2767 +/- 0.0368 | 0.2867 +/- 0.0370 | 0.2883 +/- 0.0425 | 0.4583 +/- 0.0473 | 0.5767 +/- 0.0306 |
| Mean violation: one_kitchen | 0.1942 +/- 0.0116 | 0.0233 +/- 0.0125 | 0.0233 +/- 0.0085 | 0.0100 +/- 0.0041 | 0.0283 +/- 0.0155 | 0.0533 +/- 0.0085 | 0.0750 +/- 0.0108 |
| Mean viol. (failed): between_2_and_3_bathrooms | 1.0112 +/- 0.0043 | 1 | 1 | 1 | 1 | 1 | 1.0112 +/- 0.0079 |
| Mean viol. (failed): kitchen_near_living | 1 | 1 | 1 | 0.6667 +/- 0.4714 | 1 | 1 | 1 |
| Mean viol. (failed): no_bath_kitchen | 1.3620 +/- 0.0297 | 1.0867 +/- 0.0328 | 1.0789 +/- 0.0163 | 1.0795 +/- 0.0265 | 1.0854 +/- 0.0812 | 1.1722 +/- 0.0429 | 1.2393 +/- 0.0276 |
| Mean viol. (failed): one_kitchen | 1.0393 +/- 0.0097 | 1 | 1 | 1 | 1 | 1 | 1.0208 +/- 0.0295 |

## Denoising (Model Quality, Seed-Independent)

| Metric | llada_topp0.9_remdm_confidence_tsw1.0 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.01 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.05 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.1 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.15 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.3 | llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.5 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| acc_edge@t=0.1 | 0.8011 | -- | -- | -- | -- | -- | -- |
| acc_edge@t=0.3 | 0.6739 | -- | -- | -- | -- | -- | -- |
| acc_edge@t=0.5 | 0.5386 | -- | -- | -- | -- | -- | -- |
| acc_edge@t=0.7 | 0.3982 | -- | -- | -- | -- | -- | -- |
| acc_edge@t=0.9 | 0.2732 | -- | -- | -- | -- | -- | -- |
| acc_node@t=0.1 | 0.8322 | -- | -- | -- | -- | -- | -- |
| acc_node@t=0.3 | 0.6883 | -- | -- | -- | -- | -- | -- |
| acc_node@t=0.5 | 0.5558 | -- | -- | -- | -- | -- | -- |
| acc_node@t=0.7 | 0.4223 | -- | -- | -- | -- | -- | -- |
| acc_node@t=0.9 | 0.2757 | -- | -- | -- | -- | -- | -- |
| ce_edge@t=0.1 | 0.5788 | -- | -- | -- | -- | -- | -- |
| ce_edge@t=0.3 | 0.9407 | -- | -- | -- | -- | -- | -- |
| ce_edge@t=0.5 | 1.3409 | -- | -- | -- | -- | -- | -- |
| ce_edge@t=0.7 | 1.7756 | -- | -- | -- | -- | -- | -- |
| ce_edge@t=0.9 | 2.1861 | -- | -- | -- | -- | -- | -- |
| ce_node@t=0.1 | 0.5313 | -- | -- | -- | -- | -- | -- |
| ce_node@t=0.3 | 0.8650 | -- | -- | -- | -- | -- | -- |
| ce_node@t=0.5 | 1.1742 | -- | -- | -- | -- | -- | -- |
| ce_node@t=0.7 | 1.4736 | -- | -- | -- | -- | -- | -- |
| ce_node@t=0.9 | 1.7952 | -- | -- | -- | -- | -- | -- |

---
*Auto-generated by `scripts/compare.py`. Values shown as mean +/- std (population) over N seeds.*
*JS/TV/W1 are the primary distance measures. KL metrics marked "(diag.)" are diagnostic only.*
