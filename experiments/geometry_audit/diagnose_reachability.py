"""Compare free generation and candidate answer scores for PhysVLM."""
import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import torch

from experiments.physvlm_reproduction.evaluate_reachability import (
    conv_templates, image, load_model, parse_label, process_images, tokenizer_image_token,
)


def inputs(model, tokenizer, rgb, spatial_map, question, explicit):
    conv = conv_templates["qwen2"].copy()
    suffix = "\nAnswer Yes or No." if explicit else ""
    conv.append_message(conv.roles[0], f"<image>\n<depth>\n{question}{suffix}")
    conv.append_message(conv.roles[1], None)
    prefix = conv.get_prompt()
    ids = tokenizer_image_token(prefix, tokenizer, return_tensors="pt").unsqueeze(0).to(model.device)
    rgb_pixels = process_images([rgb], model.get_vision_tower().image_processor, model.config)
    map_pixels = process_images([spatial_map], model.get_depth_tower().image_processor, model.config)
    args = dict(images=rgb_pixels.to(model.device, model.dtype),
                depths=map_pixels.to(model.device, model.dtype), image_sizes=[rgb.size])
    return prefix, ids, args


def candidates(tokenizer, prefix, ids, marker=""):
    ids = ids[0].tolist()
    result = {}
    for label, answer in (("Y", "Yes"), ("N", "No")):
        tokens = tokenizer_image_token(prefix + marker + answer, tokenizer)
        if tokens[:len(ids)] != ids or not tokens[len(ids):]:
            raise RuntimeError(f"Assistant/candidate tokenization boundary changed for {answer}")
        result[label] = tokens[len(ids):]
    return result


def scores(model, ids, visual, sequences, prompt_features=None):
    result = {}
    for label, parts in sequences.items():
        suffix = torch.tensor([parts], device=ids.device)
        if prompt_features is None:
            tokens = torch.cat((ids, suffix), dim=1)
            logits = model(input_ids=tokens, use_cache=False, **visual).logits
            answer_logits = logits[0, -len(parts)-1:-1].float()
        else:
            embedded = torch.cat((prompt_features, model.model.embed_tokens(suffix)), dim=1)
            hidden = model.model(inputs_embeds=embedded, use_cache=False).last_hidden_state
            answer_logits = model.lm_head(hidden[0, -len(parts)-1:-1]).float()
        values = answer_logits.log_softmax(dim=-1).gather(1, suffix.reshape(-1, 1)).squeeze(1)
        result[label] = values.tolist()
    return result


@torch.inference_mode()
def predict(model, tokenizer, rgb, spatial_map, question, explicit, reuse_features=False, free_only=False):
    prefix, ids, visual = inputs(model, tokenizer, rgb, spatial_map, question, explicit)
    generated = model.generate(ids, images=visual["images"], depth_images=visual["depths"],
        image_sizes=visual["image_sizes"], do_sample=False, max_new_tokens=16,
        output_scores=True, return_dict_in_generate=True, use_cache=True)
    raw_ids = generated.sequences[0].tolist()
    answer = tokenizer.decode(raw_ids, skip_special_tokens=True).strip()
    if free_only:
        return dict(free=parse_label(answer), response=answer, generated_ids=raw_ids)
    sequences = candidates(tokenizer, prefix, ids)
    features = None
    if reuse_features:
        _, _, _, _, features, _ = model.prepare_inputs_labels_for_multimodal(
            ids, None, None, None, None, visual["images"], visual["depths"], visual["image_sizes"])
    values = scores(model, ids, visual, sequences, features)
    first = generated.scores[0][0].float().log_softmax(0)
    error = max(abs(float(first[seq[0]]) - values[label][0]) for label, seq in sequences.items())
    if error > 0.2:
        raise RuntimeError(f"Generation and forced first-token log P differ by {error:.4f}")
    conditioned = candidates(tokenizer, prefix, ids, "<|endoftext|> ")
    marker_values = scores(model, ids, visual, conditioned, features)
    common = 0
    for y, n in zip(conditioned["Y"], conditioned["N"]):
        if y != n:
            break
        common += 1
    if not common or min(map(len, conditioned.values())) <= common:
        raise RuntimeError("Expected common marker followed by distinct answer tokens")
    marker_values = {key: value[common:] for key, value in marker_values.items()}
    return dict(free=parse_label(answer), response=answer, generated_ids=raw_ids,
                margin=sum(values["Y"]) - sum(values["N"]), candidate_ids=sequences, token_logp=values,
                marker_candidate_ids=conditioned, marker_token_logp=marker_values,
                marker_margin=sum(marker_values["Y"]) - sum(marker_values["N"]), first_token_error=error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    source_arg = parser.add_mutually_exclusive_group(required=True)
    source_arg.add_argument("--manifest", type=Path)
    source_arg.add_argument("--pairs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    source = args.manifest or args.pairs
    content = json.loads(source.read_text(encoding="utf-8"))
    examples = content["rows"] if args.manifest else paired_examples(source, content)
    start = time.perf_counter()
    model, tokenizer = load_model(args.model, "cuda:0")
    print(json.dumps({"event": "loaded", "seconds": time.perf_counter() - start}), flush=True)
    rows = []
    with args.output.open("x", encoding="utf-8") as stream:
        for row in examples[:args.limit]:
            rgb_path = source.parent / row["image"]
            map_path = source.parent / row["sp_map"]
            for path, field in ((rgb_path, "image_sha256"), (map_path, "map_sha256")):
                if hashlib.sha256(path.read_bytes()).hexdigest() != row[field]:
                    raise ValueError(f"Input hash mismatch: {path}")
            rgb, spatial_map = image(rgb_path), image(map_path)
            result = {**row, "original": predict(model, tokenizer, rgb, spatial_map, row["question"], False),
                      "explicit": predict(model, tokenizer, rgb, spatial_map, row["question"], True)}
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            rows.append(result)
    for mode in ("original", "explicit"):
        counts = Counter(f"{r['truth']}-{r[mode]['free']}" for r in rows)
        print(json.dumps(dict(mode=mode, free_confusion=counts, n=len(rows),
            candidate_correct=sum((r[mode]["margin"] > 0) == (r["truth"] == "Y") for r in rows),
            marker_candidate_correct=sum((r[mode]["marker_margin"] > 0) == (r["truth"] == "Y") for r in rows))))
    print(json.dumps(dict(event="complete", seconds=time.perf_counter() - start,
                          peak_mib=torch.cuda.max_memory_allocated()/2**20)), flush=True)


def paired_examples(selection, content):
    rows = []
    for group in ("YN", "NY", "YY", "NN"):
        chosen = [row for row in content["selection"] if row["label"] == group][:4]
        for row in chosen:
            path = selection.parent / row["fixture"]
            fixture = json.loads(path.read_text())
            target = next(iter(fixture["pairs"]))
            for side, label in zip(("A", "B"), group, strict=True):
                rgb = path.parent / fixture["rgb"]
                sp_map = path.parent / fixture["configurations"][side]["sp_map"]
                rows.append(dict(id=f"{row['scene']}-{side}", truth=label, side=side,
                    question=f"Is the {target} in the robot's reachable space?",
                    image=str(rgb.relative_to(selection.parent)), image_sha256=fixture["rgb_sha256"],
                    sp_map=str(sp_map.relative_to(selection.parent)),
                    map_sha256=hashlib.sha256(sp_map.read_bytes()).hexdigest()))
    return rows


if __name__ == "__main__":
    main()
