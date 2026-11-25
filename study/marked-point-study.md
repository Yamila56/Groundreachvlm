# Specified-point reachability

This evaluation asks whether a precisely specified visible point is within the Panda tool link's position workspace. Each candidate object has an identical magenta cross in RGB. The question names one object, and the map stays unmarked. The cross locates the target point without revealing which object is asked about. Grounding DINO and the frozen CLIP-based residual rankers use the original RGB and the target phrase. Rules can recover the marker inside their proposed box without simulator annotations.

## Held-out scenes

The generator uses seed 563891 to create 32 physical layouts. For each layout it renders two object orders and uses two workspace configurations per image. The selected surface point can vary across each object's visible region. The 256 questions contain 93 joint-limited IK/FK witnesses, 138 conservative URDF exclusions, and 25 unresolved cases. Independent certificate replay verified all positives and negatives. The four-question analysis includes 44 of the 64 views whose four labels were certified. The map-switch analysis includes 93 certified pairs with different labels.

| Model or readout | Certified questions | Four answers correct | Correct map-switch pairs |
| --- | ---: | ---: | ---: |
| Frozen global PhysVLM | 93/231 | 0/44 | 0/93 |
| Detector boxes, center rule | 171/231 | 11/44 | 48/93 |
| Residual ranking, center rule | 198/231 | 34/44 | 80/93 |
| Residual ranking, marker pixel | 200/231 | 34/44 | 80/93 |
| Residual ranking, Logistic | 190/231 | 34/44 | 80/93 |
| Residual ranking, two-layer MLP | 192/231 | 34/44 | 80/93 |
| Residual ranking, local projector, seeds 17 and 29 | 187/231 · 122/231 | 29/44 · 1/44 | 75/93 · 6/93 |
| Simulator box, center rule | 230/231 | 44/44 | 93/93 |
| Simulator marker pixel | 231/231 | 44/44 | 93/93 |

Both ranking seeds select the correct marker for 110/128 unique image–target queries. The raw detector selects 88/128. Among certified questions, every error from the ranked marker-pixel rule occurs at a wrong marker. The two ranking seeds yield identical decisions for the rule on this set. Relative to the same-box center rule, the marker rule gains 0.87 percentage points with a layout-paired 95% interval of [0, 2.19]. The Logistic and MLP readouts use historical development for selection and both trail the center rule here.

Giving the two projectors annotated boxes brings them to 214/231 and 139/231. The second model gives contradictory or unrecognized Yes/No text on 85 certified questions with annotated boxes and 89 with automatic boxes. Scoring the first Yes/No occurrence raises its results to 222/231 and 194/231, respectively. This sensitivity analysis separates some answer-format failures from local readout errors. The primary scores require an unambiguous answer.

At the specified pixels, the sampled upstream workspace agrees with all 231 certified labels. Twenty-three of the 25 unresolved points lie within the sampled workspace threshold. Those points retain their unknown status. This comparison covers the generated Panda scenes with certified outcomes and leaves uncertainty at the selected boundary cases.

## Reproduce

Generate the scenes in the CPU simulator configured in the repository README.

```bash
"$cpu_python" study/generate_grounding_study.py "$root/marked-point" \
    --point-check --seed 563891 --count 32
```

Use the unmarked `manifest.json` for `detect_grounding_targets.py`, `score_region_semantics.py --confidence .12 --top-k 5 --features ...`, and both frozen checkpoints from the ranking study. Select raw detector order with `rerank_grounding_targets.py semantic --weight 0`.

Run `evaluate_marked_points.py prepare` with the manifest and the three outputs passed as `--boxes detector=... ranked17=... ranked29=...`. Run `evaluate_target_alignment.py predict --free-only` on `global.json` for frozen full-image inference. Run it with the matching local projector on the two ranked manifests, and with each projector on the oracle manifest. The scoring command takes `regions.json`, the historical `source-dev` and `source-test` manifests, and the frozen predictions. Pass the four projector prediction files with `--projector ranked17-projector=... ranked29-projector=... oracle-projector17=... oracle-projector29=...`.

[Complete scores](../results/grounding-study/marked-point.json) · [Certificate replay](../results/grounding-study/marked-point-certificate-audit.json)
