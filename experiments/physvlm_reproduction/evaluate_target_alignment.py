"""Evaluate a fixed PhysVLM checkpoint on crossed targets and the source recipe."""
import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from experiments.physvlm_reproduction.score_official_control import summarize as summarize_official


def load_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def predict_manifest(model, tokenizer, source, output, official, free_only=False):
    import torch
    from experiments.geometry_audit.diagnose_reachability import predict
    from experiments.physvlm_reproduction.evaluate_reachability import image

    manifest = json.loads(source.read_text(encoding="utf-8"))
    rows = manifest["rows"]
    hashes = {}
    for row in rows:
        for field, digest in (("image", "image_sha256"), ("sp_map", "map_sha256")):
            path = source.parent / row[field]
            if path not in hashes:
                hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
            if hashes[path] != row[digest]:
                raise ValueError(f"Unexpected input hash: {path}")
    start = time.perf_counter()
    with output.open("x", encoding="utf-8") as stream:
        total = len(rows)
        for index, row in enumerate(rows, 1):
            rgb = image(source.parent / row["image"])
            sp_map = image(source.parent / row["sp_map"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = {"explicit": predict(model, tokenizer, rgb, sp_map, row["question"], True, True, free_only=free_only)}
                if official:
                    result["original"] = predict(model, tokenizer, rgb, sp_map, row["question"], False, True, free_only=free_only)
            stream.write(json.dumps({**row, **result}) + "\n")
            if index % 32 == 0 or index == total:
                stream.flush()
                print(json.dumps(dict(file=output.name, rows=index, total=total,
                                      seconds=round(time.perf_counter()-start, 2))), flush=True)


def evaluate(args):
    import torch
    from experiments.physvlm_reproduction.evaluate_reachability import load_model
    from experiments.physvlm_reproduction.tune_reachability import configure

    args.output.mkdir(parents=True, exist_ok=False)
    model, tokenizer = load_model(args.model, "cuda:0")
    if args.checkpoint:
        state = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
        if state["mode"] not in {"projector", "lora"}:
            raise ValueError(f"Unrecognized checkpoint mode: {state['mode']}")
        params = configure(model, state["mode"], lora_rank=state.get("lora_rank", 8))
        if state["params"].keys() != params.keys():
            raise ValueError("Checkpoint trainable parameters did not match")
        with torch.no_grad():
            for name, tensor in state["params"].items():
                params[name].copy_(tensor)
    else:
        configure(model, "frozen")
    model.eval()
    if args.alignment:
        predict_manifest(model, tokenizer, args.alignment, args.output / "alignment.jsonl", False,
                         free_only=args.free_only)
    if args.official:
        predict_manifest(model, tokenizer, args.official, args.output / "official.jsonl", True,
                         free_only=args.free_only)
        write_json(args.output / "official.json", summarize_official(load_rows(args.output / "official.jsonl")))
    write_json(args.output / "complete.json", dict(peak_mib=torch.cuda.max_memory_allocated()/2**20))


def group_score(rows, output):
    grouped = {
        "target_switch": defaultdict(list),
        "map_switch": defaultdict(list),
        "joint_four": defaultdict(list),
    }
    confusion = Counter()
    for row in rows:
        prediction = output(row)
        confusion[f"{row['truth']}-{prediction}"] += 1
        grouped["target_switch"][row["scene"], row["side"]].append((row["truth"], prediction))
        grouped["map_switch"][row["scene"], row["target"]].append((row["truth"], prediction))
        grouped["joint_four"][row["scene"]].append((row["truth"], prediction))
    metrics = {"query": dict(n=len(rows), correct=sum(confusion[x] for x in ("Y-Y", "N-N"))),
               "confusion": dict(confusion)}
    for name, groups in grouped.items():
        expected = 4 if name == "joint_four" else 2
        if any(len(group) != expected or sum(x[0] == "Y" for x in group) * 2 != expected
               for group in groups.values()):
            raise ValueError(f"Malformed balanced group: {name}")
        metrics[name] = dict(n=len(groups),
                             correct=sum(all(gold == pred for gold, pred in group) for group in groups.values()))
        if expected == 2:
            transitions = Counter()
            for group in groups.values():
                predictions = [pred for _, pred in group]
                if "U" in predictions:
                    transitions["invalid"] += 1
                elif len(set(predictions)) == 1:
                    transitions["same"] += 1
                elif all(gold != pred for gold, pred in group):
                    transitions["reversed"] += 1
            metrics[name].update({key: transitions[key] for key in ("same", "reversed", "invalid")})
    return metrics


def margin_interaction(rows, field):
    views = defaultdict(list)
    for row in rows:
        views[row["scene"]].append(row)
    contrasts = [
        mean(row["explicit"][field] for row in group if row["truth"] == "Y")
        - mean(row["explicit"][field] for row in group if row["truth"] == "N")
        for group in views.values()
    ]
    return dict(n=len(contrasts), positive=sum(value > 0 for value in contrasts),
                zero=sum(value == 0 for value in contrasts), mean_log_odds_contrast=mean(contrasts))


def margin_label(value):
    return "Y" if value > 0 else "N" if value < 0 else "U"


def free_answer(row):
    return row["explicit"]["free"]


def paired_intervals(method_rows):
    if "frozen" not in method_rows:
        return {}
    grouped = {}
    for name, rows in method_rows.items():
        layouts = defaultdict(list)
        for row in rows:
            layouts[row["layout"]].append(row)
        grouped[name] = {key: group_score(group, free_answer) for key, group in layouts.items()}
    base = grouped["frozen"]
    intervals = {}
    for name, values in grouped.items():
        if name == "frozen":
            continue
        if base.keys() != values.keys():
            raise ValueError(f"Physical layout IDs differ for {name}")
        comparisons = {}
        for metric in ("query", "target_switch", "map_switch", "joint_four"):
            deltas = []
            for key in base:
                reference, adapted = base[key][metric], values[key][metric]
                if reference["n"] != adapted["n"]:
                    raise ValueError(f"Physical layout counts differ for {name}/{key}")
                deltas.append((adapted["correct"] - reference["correct"]) / reference["n"])
            rng = random.Random(20261125)
            draws = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(10000))
            comparisons[metric] = dict(delta=mean(deltas), ci95=[draws[250], draws[9749]])
        intervals[name] = dict(n_layouts=len(base), metrics=comparisons)
    return intervals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    run = actions.add_parser("predict")
    for field in ("model", "output"):
        run.add_argument(f"--{field}", required=True, type=Path)
    run.add_argument("--checkpoint", type=Path)
    run.add_argument("--official", type=Path)
    run.add_argument("--alignment", type=Path)
    run.add_argument("--free-only", action="store_true")
    score = actions.add_parser("score")
    score.add_argument("manifest", type=Path)
    score.add_argument("output", type=Path)
    score.add_argument("runs", nargs="*", help="name=prediction-directory")
    args = parser.parse_args()
    if args.action == "predict":
        if not (args.alignment or args.official):
            run.error("Pass an alignment or official-recipe manifest")
        evaluate(args)
        return

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_rows = manifest["rows"]
    threshold = manifest["geometry_threshold"]
    report = {}
    official = {}
    method_rows = {}
    methods = {"target_region_chroma": (manifest_rows, lambda row: row["geometry"]),
               "workspace_point_cloud": (manifest_rows, lambda row: row["workspace"]),
               "global_map_chroma": (manifest_rows, lambda row: "Y" if row["whole_map_median_chroma"] >= threshold else "N")}
    for entry in args.runs:
        name, directory = entry.split("=", 1)
        directory = Path(directory)
        rows = load_rows(directory / "alignment.jsonl")
        if [(r["id"], r["truth"]) for r in rows] != [(r["id"], r["truth"]) for r in manifest_rows]:
            raise ValueError(f"Unexpected prediction IDs: {name}")
        methods[name] = (rows, free_answer)
        method_rows[name] = rows
        if (directory / "official.jsonl").is_file():
            official[name] = summarize_official(load_rows(directory / "official.jsonl"))
    for name, (rows, method) in methods.items():
        report[name] = {"all": group_score(rows, method)}
        for yaw in sorted({row["camera_yaw"] for row in rows}):
            report[name][f"yaw_{yaw}"] = group_score([row for row in rows if row["camera_yaw"] == yaw], method)
    for name, rows in method_rows.items():
        if "margin" not in rows[0]["explicit"]:
            continue
        for field in ("margin", "marker_margin"):
            def candidate(row, margin_key=field):
                return margin_label(row["explicit"][margin_key])

            yaw_groups = {f"yaw_{yaw}": [row for row in rows if row["camera_yaw"] == yaw]
                          for yaw in sorted({row["camera_yaw"] for row in rows})}
            strata = {"all": rows, **yaw_groups}
            report[name][f"candidate_{field}"] = {
                label: group_score(group, candidate) for label, group in strata.items()
            }
            report[name][f"interaction_{field}"] = {
                label: margin_interaction(group, field) for label, group in strata.items()
            }
    write_json(args.output, dict(alignment=report, alignment_vs_frozen=paired_intervals(method_rows), official=official))


if __name__ == "__main__":
    main()
