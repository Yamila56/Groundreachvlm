# Reachability labels and workspace maps

This study measures three geometric quantities under distinct protocols.

| Protocol | Point being classified | Reference |
| --- | --- | --- |
| Upstream Panda recipe | Object origin | Distance to the base, at most 0.905 m including tolerance |
| Sampled workspace | Visible surface point selected from the simulator instance mask | Distance to the closest point in the workspace cloud, at most 0.05 m |
| Controlled reachability | Visible point specified by the synthetic fixture | Joint-limited IK with FK verification for positives and a conservative URDF bound for negatives |

The upstream map renderer projects depth pixels into the robot base frame. It grays out pixels beyond the workspace-cloud threshold. The checked-in Panda cloud has 20,329 points. The pinned upstream generator samples one million joint configurations uniformly over ±180° per joint, runs DH forward kinematics, and stores 5 cm voxel centers. That recipe is an approximation. See the pinned [label rule](https://github.com/jetteezhou/PhysVLM/blob/37e25fd83d8b9f0bdb425661e2dcdffa674a79c7/EQA-phys-simulator/simulation_utils.py#L197-L203), [map renderer](https://github.com/jetteezhou/PhysVLM/blob/37e25fd83d8b9f0bdb425661e2dcdffa674a79c7/EQA-phys-simulator/generate_sp_map.py#L98-L120), and [workspace generator](https://github.com/jetteezhou/PhysVLM/blob/37e25fd83d8b9f0bdb425661e2dcdffa674a79c7/EQA-phys-simulator/offline_calculate_robot_ws.py).

## Replay audit

Replaying 70 Panda scenes yields **70/70 byte-identical S-P Map files**. The geometric workspace mask agrees with the map chroma threshold at **99.85%** of 21,504,000 image pixels and **99.60%** of 180,369 visible target pixels. Depth and transformed coordinates are checked before counting. All selected scenes have valid coverage in this replay.

The original radial labels differ from the sampled-workspace labels at **15/128** designated visible points. Three of the 64 radial positives have no visible point within the workspace-cloud threshold. The designated pixel is the valid instance pixel closest to the pixel-coordinate median. Its mask-based selection uses simulator annotations for evaluation.

An independent run uses 27 additional scenes and 48 questions. It reproduces all 27 map files byte for byte and reaches 99.85% pixel agreement. Two of its 48 designated points disagree with the radial labels. The size of this discrepancy varies across scene draws. [Independent replay](../results/grounding-study/source-map-geometry-replication.json)

Re-scoring the same saved answers against both definitions illustrates the effect of label choice. The models answered the original object-level question in every case.

| Readout | Original radial labels | Designated surface point in sampled workspace |
| --- | ---: | ---: |
| Frozen global PhysVLM | 94/128 | 81/128 |
| Annotated box and center rule | 115/128 | 126/128 |
| Annotated box and local projector, seeds 17 / 29 | 102/128 · 88/128 | 105/128 · 94/128 |

The retrospective point agreement quantifies how much the label definition affects this comparison. A point-specific benchmark can explicitly designate the queried point and use separate IK certificates.

[Replay data and PCD hash](../results/grounding-study/source-map-geometry.json) · [Replay script](../experiments/geometry_audit/audit_source_map_geometry.py) · [Readout results](../results/grounding-study/learned-reranker.json)
