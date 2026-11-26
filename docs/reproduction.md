# Reproduction

## CPU replay

Use Python 3.12, install the checkout with `python -m pip install -e .`, and run `groundreach-reproduce`. It recomputes certified counts and layout-paired intervals for both seeds on both confirmation sets and for the 8-pixel shift. Each computed report is checked against its saved `paired.json`. The legacy single-run entry point remains `python study/summarize_physical_tokens.py --help`.

## Model inference and training

Install `.[vlm]` with CUDA PyTorch and a compatible torchvision build. Use local Qwen3-VL-2B-Instruct weights matching the SHA in the reference protocol. Use `configs/point_token_a.json` for seed 17/set A and `configs/point_token_b.json` for seed 29/set B. The manifest paths must point to directories containing their rendered referenced images. The localization files passed to the runner are the full `evaluate_relation_holdout.py` predictions. The archived compact `auto-points.json` files support inspection and use a different schema.

```bash
groundreach-point-token --config configs/point_token_a.json --model models/qwen3-vl-2b \
  --train path/to/train/manifest.json --test path/to/confirm/manifest.json \
  --train-predictions path/to/train/localization/predictions.json \
  --test-predictions path/to/confirm/localization/predictions.json \
  --output runs/point-a-recheck \
  --load-heads results/grounding-study/point-token-pilot/point-token-automatic
```

The cross-check configurations in [configs](../configs) specify the alternate set for each seed. Pass the recorded heads directory with `--load-heads`. The same command trains the two residual heads when `--load-heads` is omitted. Both branches keep the frozen forward and weight-reload checks. The earlier script syntax is available via `python study/train_physical_token.py --help`. [Published heads and prediction manifests](../results/grounding-study/point-token-pilot)

## S-P Map and localization preparation

Use the PhysVLM simulator checkout at `37e25fd83d8b9f0bdb425661e2dcdffa674a79c7` and set `PHYSVLM_UPSTREAM` to its directory. The Linux simulator environment is documented by [setup_cpu_env.sh](../study/setup_cpu_env.sh). The earlier PhysVLM inference rollout has its own environment as described in [run_marked_point_study.sh](../study/run_marked_point_study.sh). Keep that environment separate from the Qwen3-VL environment.

Render train and test splits with `study/generate_relation_holdout.py --anchor-switch --balanced-points`. Train uses 104 layouts and seed 911361. Set A uses seed 911367 and 32 layouts with `--physical-probe`. Set B uses 911369 and 32 layouts with `--physical-probe --vary-pattern-direction`. Use the resulting detector inputs with [DINO candidate generation](../study/detect_grounding_targets.py), [CLIP scoring](../study/score_region_semantics.py) and [relation grounding](../study/evaluate_relation_holdout.py) in that order. The simulator supplies independent certificates and records source image hashes. [Label semantics](benchmark.md)

## Figures

Install `.[figures]` and run `python scripts/figures/render.py`. The original examples used by the figures are in [assets/figure-inputs](../assets/figure-inputs).