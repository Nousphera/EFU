# Implementation and experiment settings

EFU combines local unlearning objectives with decentralized encrypted aggregation. See [the paper](https://doi.org/10.1145/3746252.3761091) and [preprint methods](https://arxiv.org/abs/2508.07873) for the research formulation.

| Component | Code | Behavior |
|---|---|---|
| Local learning and requests | `src/efu/train.py`, `client.py` | Sample/class request selection, local training, scheduled unlearning and retained-only training |
| EFU objective, Eq. 3 | `client.py`, `utils/mas.py` | Negative forget CE, adversarial CE and elementwise inverted-MAS drift penalty |
| Offline targeted L2 APGD | `utils/apgd.py`, `prepare_unlearning` | Raw [0,1] inputs, model-internal normalization, cached incorrect-label attacks |
| K-means weight clustering | `utils/fe.py::cluster` | Explicit seed and four initializations |
| Coordinate FE | `coordinate_fe.py`, vendored PyMIFE | Client keys, functional shares and sample-weighted P-256 aggregation |
| Task unlearning | `task_data.py`, `task_unlearning.py` | Shared backbone, identity/gender heads, simultaneous forgetting/retention and separate evaluation |
| Local objective comparisons | `utils/baselines.py` | PGD, SGA-EWC and FedOSD options for sample/class requests |

## Learning and unlearning

`--method efu` uses encrypted aggregation. `fedavg` is a plaintext learning control; `retrain` starts from the same seeded initialization and excludes the forget set from its first round. `--forget-client`, `--forget-fraction` and `--forget-class` define a sample/class request. Every scheduled request round includes its requesting client; remaining participants are sampled uniformly. Following the unlearning window, training resumes on retained IDs.

The EFU objective combines negative cross-entropy on forgotten examples, cross-entropy on targeted adversarial examples with incorrect labels, and an inverted-MAS parameter drift penalty. Attack inputs, initial reference weights and importance values are cached for the request. `--attack-steps` controls APGD iterations. Reported `accuracy` evaluates the held-out dataset; `forget_train_accuracy` evaluates the designated forgotten training IDs.

For task unlearning, `--task identity-gender` applies an all-client request using explicit LFW annotations. It trains both tasks initially, optimizes gender CE and the identity unlearning objective together during the request, then trains only gender. Unit objective weights, Adam with weight decay 0.01, identity-head regularizer scope and one-based phase indexing are recorded in the run. [TASK_UNLEARNING.md](TASK_UNLEARNING.md) gives the full recipe.

## Experiment configuration

The CLI exposes clients, participation, local/unlearning epochs, learning rate, seed, model, partition concentration and quantization. `resnet18` and `vit_b_32` support explicit torchvision ImageNet-weight acquisition through `--pretrained`. Images are resized and converted to tensors; normalization is inside the model so adversarial inputs remain in [0,1]. FEMNIST also supports natural writer partitions.

Paper settings include Dirichlet concentration 0.1, 64 centroids, one learning epoch, five unlearning epochs and pretrained ResNet-18/ViT-B/32. Sample/class experiments use 10 clients with 20% participation. The identity/gender task uses 30 clients, full participation and 100 total rounds. Round numbering starts at one: `--unlearn-start 51 --unlearn-rounds 10` gives 50 initial learning rounds and a request window of rounds 51–60. Record supplied data splits, annotations, model weights and seeds with the run.

## Coordinate-labelled functional encryption

Aggregation uses PyMIFE's `FeDDHMultiClientDec` primitive over P-256: public parameters, independent client secrets, pairwise key exchange, client-generated functional shares and decryption of the sample-weighted sum. Each expanded coordinate has a unique session/round/coordinate label. The aggregator holds public parameters and shares; it checks participant IDs, function weights, labels and dimensions. Keys use operating-system entropy independently of training seeds.

Each client sends **d ciphertexts for d model coordinates**. The paper's compressed-centroid transport uses κ ciphertexts; the implemented coordinate transport has a different cost. Client-side expansion gives every coordinate its own encryption label even when several coordinates use the same centroid. BatchNorm running statistics are averaged in plaintext and integer counters use the maximum client value.

The executable is a local simulator: all parties run in one process and exchange messages directly. Each selected client must reply. Authenticated transport, client-process isolation, persistent key provisioning and dropout handling belong to a deployment layer. PyMIFE is a research cryptography library; application tests verify arithmetic and message invariants.

## Numerical representation

Clients agree on the active cohort, positive integer sample counts, model layout, fixed-point precision and public absolute weight bound. A weight is rounded after multiplication by `10**precision`; decryption returns the sample-weighted mean of those integers, scaled back to real values. The fixed-point rounding error is at most `0.5 / 10**precision` per coordinate relative to the reconstructed clustered mean; clustering adds its own approximation error. Nonfinite and out-of-bound values fail validation.

Larger precision, sample counts and model dimensions increase bounded discrete-log work. Coarse precision can erase small updates. Use the tiny model to check the complete path before allocating a full-model experiment, and record `kappa`, `precision` and `max_abs` with results. A changed active cohort receives a fresh decentralized setup.

## Data, randomness and saved results

[DATASETS.md](DATASETS.md) specifies preparation and consumed paths. `scripts/prepare_data.py` creates `ROOT/DATASET/manifest.json`; the trainer's `--data-root` is the same parent root. Run `--validate` before an experiment to hash every file. Training checks split sizes, class maps and label order, and records the manifest SHA256.

Training, partitioning and K-means use explicit seeds; cryptographic randomness stays fresh. The default synthetic configuration uses two clients, eight training vectors per client, three classes, a linear model, four centroids, precision 2, CPU and one local epoch. Imports, installation and tests require no dataset or pretrained-weight download.

JSON records configuration, exact client partitions, participants, sample counts, metrics and Python/platform/Torch/NumPy versions. The sibling `.pt` contains CPU model tensors, configuration and class information:

```python
import torch
checkpoint = torch.load("runs/experiment.pt", weights_only=True, map_location="cpu")
```

Existing output or checkpoint paths are rejected before training. Keep the prepared manifest, dependency constraints and seed with each result. Client keys and plaintext input records are excluded from reports.

## Validation

```bash
python -m pip check
python -m pytest -q
```

The suite covers seeded partitions, the primary trainer, prepared image inputs, signed fixed-point encrypted aggregation, sample weighting, participant/message validation and checkpoint output. `constraints.txt` records the dependency versions tested with Python 3.12 on macOS arm64.

Task tests additionally verify annotation alignment and integrity, live gender gradients during identity forgetting, repeated use of cached attacks/importance, phase routing, separate per-head metrics and actual ResNet18/ViT dual-head forward passes.
