import json

import torch
from PIL import Image

from groundreach.adapters.residual import ResidualProjection
from groundreach.evaluation.counterfactual import paired_bootstrap
from groundreach.evaluation.inference import evaluate, load_examples
from groundreach.evaluation.metrics import counts
from groundreach.vision.point_selection import select_features
from groundreach.vision.spatial_map import shared_point_crop, workspace_mask


def test_aligned_crop_and_workspace_threshold():
    assert shared_point_crop([(10, 20), (100, 140)], 120, 160) == (0, 0, 120, 160)
    image = Image.new("RGB", (2, 1))
    image.putdata([(83, 0, 0), (82, 0, 0)])
    mask = workspace_mask(image)
    assert [mask.getpixel((x, 0)) for x in range(2)] == [255, 0]


def test_merged_patch_and_matched_controls():
    grid = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    features, index = select_features(grid, 8, 3, 12, 4)
    assert index == (1, 2)
    torch.testing.assert_close(features["point"], grid[1, 2], rtol=0, atol=0)
    torch.testing.assert_close(features["mean"], grid.mean((0, 1)), rtol=0, atol=0)
    for _ in ("point", "mean"):
        projection = ResidualProjection(2048)
        assert sum(p.numel() for p in projection.parameters() if p.requires_grad) == 133152
        token = torch.arange(2048, dtype=torch.float32)
        torch.testing.assert_close(projection(token), token, rtol=0, atol=0)


def test_missing_point_is_unanswered(tmp_path):
    row = {"id": "q", "point": None, "scene": "s", "target": "a", "side": "A", "truth": "Y"}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"rows": [row]}), encoding="utf-8")
    examples = load_examples(manifest, engine=None)
    predictions = evaluate(None, examples)
    assert predictions == {"q": {"answer": "U"}}
    assert counts([row], {"q": "U"})["unanswered"] == 1


def test_certified_pair_counts_and_layout_bootstrap():
    rows = [dict(id=str(i), scene="a", side="A" if i < 2 else "B",
                 target=i % 2, truth=truth)
            for i, truth in enumerate(("Y", "N", "N", "Y"))]
    outcome = counts(rows, {row["id"]: row["truth"] for row in rows})
    assert outcome["target_flip"]["correct"] == outcome["workspace_flip"]["correct"] == 2
    assert outcome["four"]["correct"] == 1
    assert paired_bootstrap({"a": [1, 1], "b": [0, 0]}) == {
        "n": 4, "layouts": 2, "difference": .5, "interval95": [0., 1.],
    }
