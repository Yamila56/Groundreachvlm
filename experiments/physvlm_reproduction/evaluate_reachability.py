"""Run the public two-stream PhysVLM checkpoint on fixed-image reachability pairs."""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
END_MARKERS = re.compile(r"^(?:\s*<\|(?:endoftext|im_end)\|>)+\s*|(?:\s*<\|(?:endoftext|im_end)\|>)+\s*$")
UPSTREAM = Path(os.environ.get("PHYSVLM_UPSTREAM", str(ROOT))).resolve()
sys.path.insert(0, str(UPSTREAM / "physvlm-main"))
from physvlm.constants import DEPTH_TOKEN_INDEX, IMAGE_TOKEN_INDEX
from physvlm.conversation import conv_templates
from physvlm.mm_utils import process_images, tokenizer_image_token
from physvlm.model.language_model.llava_qwen import LlavaConfig, LlavaQwen2ForCausalLM
from physvlm.model.multimodal_encoder.siglip_encoder import SigLipVisionModel, SigLipVisionTower


def load_model(path, device):
    config = LlavaConfig.from_pretrained(path, local_files_only=True)
    config.mm_vision_tower = config.mm_depth_tower = str(path.resolve())
    weights = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    prefixes = ("model.vision_tower.", "model.depth_tower.", "model.mm_projector.", "model.mm_depth_projector.")
    if not all(any(key.startswith(prefix) for key in weights) for prefix in prefixes):
        raise ValueError("Checkpoint index does not include both visual branches and projectors.")

    def initialize_tower(tower, device_map=None):
        # Both visual towers are already present in the composite checkpoint.
        tower.vision_tower = SigLipVisionModel(tower.config)
        tower.vision_tower.requires_grad_(False)
        tower.is_loaded = True

    with patch.object(SigLipVisionTower, "load_model", initialize_tower):
        model, loading = LlavaQwen2ForCausalLM.from_pretrained(
            path, config=config, torch_dtype=torch.bfloat16, device_map={"": device},
            local_files_only=True, output_loading_info=True,
        )
    parameters = dict(model.named_parameters(remove_duplicate=False))
    missing_parameters = set(loading["missing_keys"]) & parameters.keys()
    if config.tie_word_embeddings and "model.embed_tokens.weight" not in missing_parameters:
        missing_parameters.discard("lm_head.weight")
    unexpected_vision = [key for key in loading["unexpected_keys"] if key.startswith(prefixes)]
    if missing_parameters or unexpected_vision or loading["mismatched_keys"] or loading["error_msgs"]:
        raise RuntimeError(json.dumps(loading, default=str))
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, local_files_only=True)
    print(json.dumps({"event": "checkpoint", "missing": loading["missing_keys"],
                      "unexpected": loading["unexpected_keys"]}), flush=True)
    return model.eval(), tokenizer


def parse_label(text):
    text = END_MARKERS.sub("", text).strip()
    values = re.findall(r"\b(?:yes|no)\b", text, re.IGNORECASE)
    match = re.match(r"\s*(yes|no)\b", text, re.IGNORECASE)
    if not match or len(set(word.lower() for word in values)) > 1:
        return "U"
    return match[1][0].upper()


def image(path):
    with Image.open(path) as source:
        return source.convert("RGB")


def infer(model, tokenizer, rgb, spatial_map, question, probe):
    conv = conv_templates["qwen2"].copy()
    conv.append_message(conv.roles[0], f"<image>\n<depth>\n{question}\nAnswer Yes or No.")
    conv.append_message(conv.roles[1], None)
    tokens = tokenizer_image_token(conv.get_prompt(), tokenizer, return_tensors="pt").unsqueeze(0).to(model.device)
    if any((tokens == value).sum().item() != 1 for value in (IMAGE_TOKEN_INDEX, DEPTH_TOKEN_INDEX)):
        raise ValueError("Both modalities must appear exactly once in the input.")
    rgb_pixels = process_images([rgb], model.get_vision_tower().image_processor, model.config)
    map_pixels = process_images([spatial_map], model.get_depth_tower().image_processor, model.config)
    diagnostics = {}
    hooks = []
    if probe:
        def capture(name):
            def hook(module, args, output):
                vector = output.detach().float().cpu().contiguous()
                pixels = args[0]
                diagnostics[name] = {
                    "input_shape": list(pixels.shape), "input_dtype": str(pixels.dtype),
                    "input_min": float(pixels.min()), "input_max": float(pixels.max()),
                    "feature_shape": list(vector.shape),
                    "feature_sha256": hashlib.sha256(vector.numpy().tobytes()).hexdigest(),
                }
            return hook
        for name, tower in (("rgb", model.get_vision_tower()), ("map", model.get_depth_tower())):
            hooks.append(tower.register_forward_hook(capture(name)))
    try:
        output = model.generate(
            tokens, images=rgb_pixels.to(model.device, model.dtype),
            depth_images=map_pixels.to(model.device, model.dtype), image_sizes=[rgb.size],
            do_sample=False, max_new_tokens=16, output_scores=probe,
            return_dict_in_generate=True, temperature=None, top_p=None, use_cache=True,
        )
    finally:
        for hook in hooks:
            hook.remove()
    first_scores = output.scores[0].float().cpu() if probe else None
    if probe and (set(diagnostics) != {"rgb", "map"} or not first_scores.isfinite().all()):
        raise RuntimeError("The two visual towers and finite first-token logits are required.")
    response = tokenizer.decode(output.sequences[0], skip_special_tokens=True).strip()
    return {"answer": parse_label(response), "response": response, "towers": diagnostics}, first_scores


def evaluate(model, tokenizer, fixtures, output, probe):
    with output.open("x", encoding="utf-8") as stream:
        for path in fixtures:
            fixture = json.loads(path.read_text(encoding="utf-8"))
            rgb_path = path.parent / fixture["rgb"]
            if hashlib.sha256(rgb_path.read_bytes()).hexdigest() != fixture["rgb_sha256"]:
                raise ValueError(f"RGB hash mismatch at {path}")
            rgb = image(rgb_path)
            maps = {key: image(path.parent / conf["sp_map"]) for key, conf in fixture["configurations"].items()}
            for name, truth in fixture["pairs"].items():
                question = f"Is the {name} in the robot's reachable space?"
                results, scores = {}, {}
                for key in ("A", "B"):
                    results[key], scores[key] = infer(model, tokenizer, rgb, maps[key], question, probe)
                row = {"scene": fixture.get("scene_id", path.parent.name), "target": name,
                       "question": question, "truth": truth, "predictions": results}
                if probe:
                    row["map_logits_max_abs_delta"] = float((scores["A"] - scores["B"]).abs().max())
                    row["same_rgb_features"] = (
                        results["A"]["towers"]["rgb"]["feature_sha256"] == results["B"]["towers"]["rgb"]["feature_sha256"]
                    )
                    row["different_map_features"] = (
                        results["A"]["towers"]["map"]["feature_sha256"] != results["B"]["towers"]["map"]["feature_sha256"]
                    )
                    repeated, same_scores = infer(model, tokenizer, rgb, maps["A"], question, True)
                    row["repeat_matches"] = results["A"] == repeated and torch.equal(scores["A"], same_scores)
                    if not (row["same_rgb_features"] and row["different_map_features"] and
                            row["map_logits_max_abs_delta"] > 0 and row["repeat_matches"]):
                        raise RuntimeError(f"Dual-input wiring probe failed: {row}")
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(json.dumps({"event": "pair", "scene": row["scene"], "target": name, "truth": truth,
                                  "predicted": "".join(results[key]["answer"] for key in ("A", "B"))}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures", nargs="+", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.perf_counter()
    model, tokenizer = load_model(args.model, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evaluate(model, tokenizer, args.fixtures, args.output, args.probe)
    print(json.dumps({"event": "complete", "seconds": time.perf_counter() - started,
                      "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}), flush=True)


if __name__ == "__main__":
    main()
