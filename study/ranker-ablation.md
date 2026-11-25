# Region ranker ablation

Each variant uses the same frozen candidate boxes, detector and CLIP scores, and point readout. The learned variants use the original 237 labeled training groups, 174 of which contain both positive and negative candidates. Development uses the same 256 groups with a positive candidate. The split image identities and training group counts match both archived checkpoints. Epoch selection uses development target overlap. Each learned variant runs with seeds 17 and 29.

| Ranking on held-out specified-point inputs | Correct marker | Certified answers correct | All four correct |
| --- | ---: | ---: | ---: |
| Detector score | 88/128 | 162/231 | 24/44 |
| Equal detector and CLIP fusion | 92/128 | 165/231 | 21/44 |
| Linear residual, seeds 17 / 29 | 109/128 · 109/128 | 198/231 · 198/231 | 33/44 · 33/44 |
| MLP without explicit interaction, seeds 17 / 29 | 110/128 · 111/128 | 200/231 · 203/231 | 33/44 · 35/44 |
| Full residual MLP, seeds 17 / 29 | 110/128 · 110/128 | 200/231 · 200/231 | 34/44 · 34/44 |

The linear residual sees the same features as the full MLP. The MLP without explicit interaction sees the region and text embeddings with the same hidden width, optimization and training budget as the full MLP. The full MLP adds their elementwise product and has more input parameters. The original detector figure of 171/231 uses a center-region rule. All rows here use the marker pixel so the ranking comparison shares a readout.

The learned residual provides most of the accuracy difference. The MLP without explicit interaction is tied on seed 17 and three answers ahead on seed 29. Its layout-paired 95% intervals against the full MLP are [−2.55, +2.56] and [−0.88, +4.02] percentage points. On the independent upstream-recipe target identity check, the full and interaction-free MLPs both score 90/128 and 89/128, the linear ranker scores 89/128 and 88/128, and the frozen fusion scores 81/128. The evidence supports a simple learned ranker on these splits. It gives little basis for attributing an improvement to the explicit feature product.

The cached pool contains a box that recovers the designated marker for all 128 independent image–target queries. The 18 misses from the archived full MLP are selections of the other marker. A hard-negative comparison can therefore hold the detector fixed and use exact target identity as its endpoint. Select any margin on development data, then score an independent object/layout split. The marked-point scores here serve as a fixed retrospective comparison.

[Ablation metrics](../results/grounding-study/reproduction/ranker-ablation.json) · [Per-question predictions](../results/grounding-study/reproduction/ranker-ablation-predictions.json) · [Split hashes](../results/grounding-study/reproduction/split-manifest.json)
