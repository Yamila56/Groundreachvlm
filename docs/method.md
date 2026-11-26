# Point-conditioned map readout

The spatial locator picks a marked point. All available query points in the same layout define a shared RGB and map crop. The frozen visual encoder computes merged map patches. The point readout selects the patch containing the query coordinate. The matched baseline uses the mean over the same cropped grid.

Both readouts use a width-32 residual MLP with a zero-initialized output projection. The adapted vector replaces an extra embedding slot, leaving the original vision inputs, positions and DeepStack features in the language-model forward pass. The head is trained with cross-entropy on the Yes/No candidate logits. The runner checks that the backbone receives no gradients, cached and native forwards agree, and reloaded weights preserve logits.

[Coordinate mapping](../groundreach/vision/point_selection.py) · [Residual head](../groundreach/adapters/residual.py) · [Frozen VLM](../groundreach/models/frozen_vlm.py) · [Matched training](../groundreach/training.py)

Earlier proxy analyses use a separate PhysVLM head, learned rankers, relation rules or linear probes. Those paths remain under [study](../study).