#!/usr/bin/env bash
# Recompute the reported counts or rebuild the specified-point experiment.
set -euo pipefail

project=$(cd -- "$(dirname -- "$0")/.." && pwd)
repro="$project/results/grounding-study/reproduction"
published="$project/results/grounding-study/marked-point.json"

verify_artifacts() {
  "${REPLAY_PYTHON:-python3}" - "$repro" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = json.loads((root / "sha256.json").read_text(encoding="utf-8"))
for name, sha256 in expected.items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != sha256:
        raise SystemExit(f"Archive checksum mismatch: {name}")
PY
}

case "${1:-}" in
  replay)
    shift
    verify_artifacts
    exec "${REPLAY_PYTHON:-python3}" "$project/study/replay_marked_points.py" \
      "$repro/marked-point-predictions.json" --expected "$published" "$@"
    ;;
  run)
    if [[ $# -ne 5 ]]; then
      echo "Usage: $0 run OUTPUT DETECTOR_CHECKPOINT CLIP_CHECKPOINT PHYSVLM_CHECKPOINT" >&2
      exit 2
    fi
    ;;
  *)
    echo "Usage: $0 replay [--output PATH] or $0 run OUTPUT DETECTOR_CHECKPOINT CLIP_CHECKPOINT PHYSVLM_CHECKPOINT" >&2
    exit 2
    ;;
esac

output=$(realpath -m -- "$2")
detector=$(realpath -- "$3")
clip=$(realpath -- "$4")
physvlm=$(realpath -- "$5")
pilot_env=${PHYSVLM_PILOT_ENV:-$HOME/.local/share/physvlm-reachability}
cpu=${CPU_PYTHON:-$pilot_env/venv/bin/python}
grounding=${GROUNDING_PYTHON:-python3}
vlm=${VLM_PYTHON:-python3}
export LD_LIBRARY_PATH="$pilot_env/native/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
: "${PHYSVLM_UPSTREAM:?Set PHYSVLM_UPSTREAM to the pinned upstream checkout}"
expected=37e25fd83d8b9f0bdb425661e2dcdffa674a79c7
test "$(git -C "$PHYSVLM_UPSTREAM" rev-parse HEAD)" = "$expected"
verify_artifacts

cd "$project"
mkdir -p -- "$(dirname -- "$output")"
mkdir -- "$output"
"$cpu" study/generate_grounding_study.py "$output/data" --point-check --seed 563891 --count 32
input="$output/data/manifest.json"
"${REPLAY_PYTHON:-python3}" - "$repro/split-manifest.json" "$output/data" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

source = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["marked_point"]
folder = Path(sys.argv[2])
for name, sha256 in (("manifest.json", source["manifest_sha256"]),
                     ("certificate-audit.json", source["certificate_audit_sha256"])):
    if hashlib.sha256((folder / name).read_bytes()).hexdigest() != sha256:
        raise SystemExit(f"Regenerated {name} differs from the frozen split")
PY
"$grounding" study/detect_grounding_targets.py "$input" "$detector" "$output/raw-boxes.json"
"$grounding" study/score_region_semantics.py "$input" "$output/raw-boxes.json" "$clip" "$output/candidates.json" \
  --confidence .12 --top-k 5 --features "$output/features.pt"
"$grounding" study/rerank_grounding_targets.py semantic "$output/candidates.json" "$output/detector.json" --weight 0
boxes=("detector=$output/detector.json")
for seed in 17 29; do
  "$grounding" study/train_region_reranker.py apply "$repro/ranker-$seed" \
    "$output/features.pt" "$output/ranked$seed.json"
  boxes+=("ranked$seed=$output/ranked$seed.json")
done
"$vlm" study/evaluate_marked_points.py prepare "$input" "$output/readouts" --regions-only --boxes "${boxes[@]}"
"$vlm" study/evaluate_target_alignment.py predict --free-only --model "$physvlm" \
  --alignment "$output/readouts/global.json" --output "$output/frozen"
"$vlm" study/evaluate_marked_points.py score-rules "$output/readouts/regions.json" \
  "$output/frozen/alignment.jsonl" "$output/metrics.json" --export "$output/predictions.json"
