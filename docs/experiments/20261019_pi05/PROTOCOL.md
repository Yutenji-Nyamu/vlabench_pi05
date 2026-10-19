# Protocol

| Component | Pin |
|---|---|
| VLABench | `OpenMOSS/VLABench@cf588fe60c0c7282174fe979f5913170cfe69017` |
| OpenPI | `Shiduo-zh/openpi@788ef1f20e655de5bb40869fb7b7fcdd49732387`, branch `pi05` |
| Model | `VLABench/pi05-primitive-10task@f658a8bec801aab960fe0ceab370a123b31c1837` |
| Asset mirror | `lerobot/vlabench-assets@08d7a4479a55c8fbb89ac0fd6fcf06e53d65b4a4` |

The [model card](https://huggingface.co/VLABench/pi05-primitive-10task) pairs this model with `pi05_ft_vlabench_primitive`. It predicts 10 actions, executes 5 before replanning, and uses the official 10-step flow sampler. Three policy images, state normalization, coordinate conversion, gripper threshold, inverse kinematics, task horizons, termination, intention and progress metrics remain upstream behavior.

Nominal coverage is 10 task families × 5 tracks × 50 trials. The pinned released JSONs contain **2460 unique configurations**: cross-category `insert_flower` has 10, not 50. We evaluate all available configurations without padding repeats. In the common-sense track the actual poker/mahjong task keys are `select_nth_largest_poker` and `select_unique_type_mahjong`.

Tracks: in-distribution, cross-category, common-sense, semantic-instruction and unseen-texture. No additional open cross-task suite, fine-tuning, expert seed filtering, U signal, or policy switching is included.

Execution starts with the official topology: one policy server and two independent simulation clients per GPU, on two allocated GPUs. The policy service serializes requests as implemented upstream; this is not batched model inference. The simulator is MuJoCo 3.2.2 / dm_control 1.0.22, not RoboDojo's Isaac simulator.

Recording additions are isolated in `tools/pi05_eval`: durable episode JSON, bounded task-level error skipping, progress timestamps, and the terminal RGB observation appended to the native four-camera mosaic video. Successful and failed episodes retain videos. Errors and skipped configurations are reported separately from valid policy failures and are never silently removed from the coverage denominator.

The upstream reference in-distribution mean is 40.6%; it is a model-card result, not a result from this deployment. Shared policy RNG and concurrent request ordering are recorded execution conditions, not a claim of bitwise reproduction.
