"""Frozen visual cache and point-token injection."""

from contextlib import ExitStack

import torch

from groundreach.models.prompts import LEGEND
from groundreach.vision.point_selection import select_features

QUESTION = "Is the specified surface point in the robot's reachable space?"


class FrozenVLM:
    def __init__(self, path):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.device = torch.device("cuda")
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, dtype=torch.bfloat16, local_files_only=True, use_safetensors=True,
        ).to(self.device).eval().requires_grad_(False)
        tokenizer = self.processor.tokenizer
        choices = [tokenizer.encode(word, add_special_tokens=False) for word in ("No", "Yes", ".")]
        if any(len(choice) != 1 for choice in choices):
            raise ValueError("Expected single tokens for No, Yes, and the injected slot")
        self.no, self.yes, self.slot = (item[0] for item in choices)

    @torch.no_grad()
    def prepare(self, rgb, spatial, x, y, keep_pixels=False):
        w, h = spatial.size
        text = (f"{LEGEND} The target surface point in image 1 is at (x={x}, y={y}) "
                f"in the {w} by {h} pixel image, from the upper left.\n\n{QUESTION}")
        messages = [{"role":"user","content":[{"type":"image","image":rgb},
                                                {"type":"image","image":spatial},{"type":"text","text":text}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                     return_dict=True, return_tensors="pt").to(self.device)
        vision = self.model.visual(inputs.pixel_values.to(self.model.visual.dtype), grid_thw=inputs.image_grid_thw)
        merge = self.model.visual.spatial_merge_size
        _, rows, cols = inputs.image_grid_thw[-1].tolist()
        rows, cols = rows//merge, cols//merge
        grid = vision[0][-rows*cols:].reshape(rows,cols,-1).float()
        features, (row, col) = select_features(grid, x, y, w, h)
        pixels = inputs.pop("pixel_values")
        example = {
            "inputs":inputs.to("cpu"),
            "vision":(vision[0].cpu(), [value.cpu() for value in vision[1]]),
            "features":features,
            "index":[row,col],
        }
        if keep_pixels:
            example["pixels"] = pixels.cpu()
        return example

    def logits(self, example, token=None, *, replay=True):
        inputs = {key:value.to(self.device) for key,value in example["inputs"].items()}
        if token is not None:
            original_ids = inputs["input_ids"]
            inputs["input_ids"] = torch.cat((original_ids, original_ids.new_full((1,1), self.slot)),dim=1)
            mask = inputs["attention_mask"]
            inputs["attention_mask"] = torch.cat((mask, mask.new_ones((1,1))),dim=1)
        inputs["pixel_values"] = torch.empty(0, device=self.device)
        if not replay:
            inputs["pixel_values"] = example["pixels"].to(self.device)
        with ExitStack() as cleanup:
            if replay:
                visual, deep = example["vision"]
                original_forward = self.model.visual.forward
                cleanup.callback(setattr, self.model.visual, "forward", original_forward)

                def replay_visual(*args, **kwargs):
                    return visual.to(self.device), [value.to(self.device) for value in deep]

                self.model.visual.forward = replay_visual
            if token is not None:
                def inject(_, ids, values):
                    if ids[0][0, -1].item() != self.slot:
                        raise ValueError("The physical token is outside the designated slot")
                    return torch.cat((values[:, :-1], token.to(values.dtype).view(1, 1, -1)), dim=1)

                handle = self.model.get_input_embeddings().register_forward_hook(inject)
                cleanup.callback(handle.remove)
            logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        return logits[[self.no, self.yes]]
