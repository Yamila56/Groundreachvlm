# Reproduction

## CPU replay

Use Python 3.12, install the checkout with `python -m pip install -e .`, and run `groundreach-reproduce`. It recomputes certified counts and layout-paired intervals for both seeds on both confirmation sets and for the 8-pixel shift. Each computed report is checked against its saved `paired.json`. The legacy single-run entry point remains `python -m experiments.counterfactual.summarize_physical_tokens --help`.

## Model inference and training

Install `.[vlm]` with CUDA PyTorch and a compatible torchvision build. Use the full [Qwen3-VL-2B-Instruct snapshot](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/e2378df056d88153dc44616229fa371fcb87e236), including its processor and tokenizer files. Before loading the model, the runner verifies the SHA-256 of the single `model.safetensors` file against the selected protocol. Use `configs/point_token_a.json` for seed 17/set A and `configs/point_token_b.json` for seed 29/set B. The manifest paths must point to directories containing their rendered referenced images. The localization files passed to the runner are the full predictions from `groundreach-bind`. The archived compact `auto-points.json` files support inspection and use a different schema.

```bash
groundreach-point-token --config configs/point_token_a.json --model models/qwen3-vl-2b \
  --train path/to/train/manifest.json --test path/to/confirm/manifest.json \
  --train-predictions path/to/train/localization/predictions.json \
  --test-predictions path/to/confirm/localization/predictions.json \
  --output runs/point-a-recheck \
  --load-heads results/grounding-study/point-token-pilot/point-token-automatic
```

The cross-check configurations in [configs](../configs) specify the alternate set for each seed. Pass the recorded heads directory with `--load-heads`. Use `groundreach-infer` with the same flags to require a saved head directory. The `groundreach-point-token` command trains the two residual heads when `--load-heads` is omitted. Both branches keep the frozen forward and weight-reload checks. The earlier script syntax is available via `python -m experiments.point_adaptation.train_physical_token --help`. [Published heads and prediction manifests](../results/grounding-study/point-token-pilot)

## S-P Map and localization preparation

Use the PhysVLM simulator checkout at `37e25fd83d8b9f0bdb425661e2dcdffa674a79c7` and set `PHYSVLM_UPSTREAM` to its directory. The Linux simulator environment is documented by [setup_cpu_env.sh](../experiments/physvlm_reproduction/setup_cpu_env.sh). The earlier PhysVLM inference rollout has its own environment as described in [run_marked_point_study.sh](../experiments/relational_grounding/run_marked_point_study.sh). Keep that environment separate from the Qwen3-VL environment.

Install the package in the simulation environment with `python -m pip install -e .` and use the following commands from the checkout root. The grounding environment requires CUDA and the `.[grounding]` extra. Use the recorded [Grounding DINO snapshot](../results/grounding-study/relation-study/dino-snapshot-sha256.json) and the [CLIP ViT-B/32 checkpoint](https://huggingface.co/openai/clip-vit-base-patch32/tree/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268).

```bash
groundreach-prepare data/train --count 104 --seed 911361 --anchor-switch --balanced-points
groundreach-prepare data/set-a --count 32 --seed 911367 --anchor-switch --balanced-points --physical-probe
groundreach-prepare data/set-b --count 32 --seed 911369 --anchor-switch --balanced-points --physical-probe --vary-pattern-direction

hf download IDEA-Research/grounding-dino-tiny --revision a2bb814dd30d776dcf7e30523b00659f4f141c71 --local-dir models/grounding-dino
hf download openai/clip-vit-base-patch32 --revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 --include '*.json' '*.txt' 'pytorch_model.bin' --local-dir models/clip

# Repeat the following for train, set-a and set-b.
split=data/set-a
groundreach-detect "$split/detector-inputs.json" models/grounding-dino "$split/dino.json"
groundreach-score-regions "$split/detector-inputs.json" "$split/dino.json" models/clip "$split/clip.json"
groundreach-bind "$split/manifest.json" "$split/clip.json" "$split/localization"
```

The simulator writes independent certificates alongside each manifest and records image hashes. The localization output contains the full `predictions.json` required by the point adapter. [Label semantics](benchmark.md)

The study runners live in [experiments](../experiments) and are invoked as Python modules from the repository root. The directory groups cover PhysVLM reproduction, geometry audits, relational grounding, point adaptation and counterfactual evaluation. The simulator checkout and historical PhysVLM inference environment remain separate dependencies.

## Third-party components

The pinned PhysVLM checkout provides the simulator and assets. Its per-asset license notices remain with that checkout. A license covering the project requires a separate review of the rights to each component. [Pinned simulator source](https://github.com/jetteezhou/PhysVLM/tree/37e25fd83d8b9f0bdb425661e2dcdffa674a79c7/EQA-phys-simulator)

## Figures

Install `.[figures]` and run `python scripts/figures/render.py`. The original examples used by the figures are in [assets/figure-inputs](../assets/figure-inputs).