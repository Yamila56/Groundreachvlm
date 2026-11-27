import json
from types import SimpleNamespace

import pytest

import torch
from PIL import Image

from groundreach.adapters.residual import ResidualProjection
from groundreach.evaluation.counterfactual import paired_bootstrap
from groundreach.evaluation.inference import evaluate, load_examples
from groundreach.evaluation.metrics import counts
from groundreach.io import sha256_file, verify_model_weights
from groundreach.models.frozen_vlm import FrozenVLM
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

def test_weight_hash_checks_the_safetensors_file(tmp_path):
    weights = tmp_path / "model.safetensors"
    with pytest.raises(FileNotFoundError):
        verify_model_weights(tmp_path, "0" * 64)
    weights.write_bytes(b"published weights")
    expected = sha256_file(weights)
    verify_model_weights(tmp_path, expected)
    weights.write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_model_weights(tmp_path, expected)


@pytest.mark.parametrize("fail_at", ["registration", "forward"])
def test_cache_and_embedding_are_restored_after_error(monkeypatch, fail_at):
    class Visual(torch.nn.Module):
        def forward(self, pixels):
            return pixels, []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = Visual()
            self.embeddings = torch.nn.Embedding(8, 2)
            self.fail = False

        def get_input_embeddings(self):
            return self.embeddings

        def forward(self, input_ids, attention_mask, pixel_values, **kwargs):
            visual, _ = self.visual(pixel_values)
            values = self.embeddings(input_ids)
            if self.fail:
                raise RuntimeError("forward")
            return SimpleNamespace(logits=values + visual)

    def unavailable(*args, **kwargs):
        raise RuntimeError("registration")

    engine = FrozenVLM.__new__(FrozenVLM)
    engine.model = Model()
    engine.device = torch.device("cpu")
    engine.no, engine.yes, engine.slot = 0, 1, 2
    original_forward = engine.model.visual.forward
    example = {
        "inputs": {"input_ids": torch.tensor([[1]]), "attention_mask": torch.ones(1, 1)},
        "vision": (torch.ones(2), []),
        "pixels": torch.ones(2),
    }
    with monkeypatch.context() as patch:
        if fail_at == "registration":
            patch.setattr(engine.model.embeddings, "register_forward_hook", unavailable)
        else:
            engine.model.fail = True
        with pytest.raises(RuntimeError, match=fail_at):
            engine.logits(example, torch.zeros(2))
    engine.model.fail = False
    assert engine.model.visual.forward == original_forward
    assert not engine.model.embeddings._forward_hooks

    token = torch.tensor([4., 7.], requires_grad=True)
    torch.testing.assert_close(engine.logits(example, token), torch.tensor([5., 8.]))
    engine.logits(example, token).sum().backward()
    torch.testing.assert_close(token.grad, torch.ones(2))
    torch.testing.assert_close(engine.logits(example), engine.logits(example, replay=False))
    assert engine.model.visual.forward == original_forward
    assert not engine.model.embeddings._forward_hooks
