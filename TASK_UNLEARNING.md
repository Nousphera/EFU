# Identity forgetting with gender retention

The maintained `--task identity-gender` route implements the task structure in
[EFU §5.2 and Table 2](https://arxiv.org/html/2508.07873v1#S5.SS2): a shared backbone,
two classification heads, joint initial learning, an all-client identity request,
and simultaneous retained-task and unlearning objectives. It uses the same
coordinate-labelled functional encryption as the other maintained EFU runs.
Published Table 2 results and the compressed-centroid protocol are not reproduced.

## Supply annotations explicitly

Acquire LFW images and independently supplied gender annotations under their
applicable terms. Create a UTF-8 CSV with exactly these columns:

```csv
path,gender
Person_A/Person_A_0001.jpg,provided-category-a
Person_A/Person_A_0002.jpg,provided-category-a
Person_B/Person_B_0001.jpg,provided-category-b
```

Paths are relative to the acquired LFW root and retain original filenames.
Every image selected by the identity/split preparation needs a nonempty annotation;
exactly two supplied categories must be represented across the selected images.
The strings are preserved in a sorted label map. Duplicate rows, missing labels,
malformed paths and incomplete coverage fail. Nothing predicts or infers a
person's gender from images, identity names or other attributes.

```bash
python scripts/prepare_data.py --dataset lfw --source /data/lfw --root data \
  --identities 30 --gender-annotations /data/annotations/lfw-gender.csv
python scripts/prepare_data.py --dataset lfw --root data --validate
```

Preparation copies selected images and writes `data/lfw/tasks.json`; the manifest
hashes it and records the original CSV hash, category map and selected counts.
Task training rechecks annotation integrity and exact train/test image coverage.
The selected identity names and seeded split remain the maintained conventions
in [DATASETS.md](DATASETS.md), rather than recovered historical choices.

## Execute

Start with a small encrypted run on acquired, prepared data:

```bash
efu-train --task identity-gender --dataset lfw --data-root data --model tiny \
  --clients 2 --alpha 10 --rounds 3 --unlearn-start 2 --unlearn-rounds 1 \
  --unlearn-epochs 1 --attack-steps 2 --precision 3 --kappa 16 --output runs/lfw-task-check.json
```

This covers one round each of joint learning, identity unlearning and gender-only
learning. For a completely offline execution check, omit the dataset/root flags;
the default synthetic vectors have artificial binary labels and represent no
human subjects. `vit_b_32` provides the shared ViT-B/32 architecture, and
`resnet18` is also supported. `--pretrained` explicitly downloads torchvision's
ImageNet weights; it does not claim the exact historical Hugging Face checkpoint.
Full-backbone coordinate-FE runs are expensive and unbenchmarked.

The task route requires `--participation 1`, acts on every client's complete local
partition, and supports EFU's own unlearning objective. Sample/client/class
selection flags and the single-task PGD/SGA-EWC/FedOSD variants do not define this
system-wide task request. Use the default `--unlearn-method efu`.

The task smoke uses `--precision 3 --kappa 16` so small updates survive coordinate-FE quantization. Coarse diagnostic settings can round small model changes to zero; choose precision and quantization levels deliberately for larger experiments.

The CLI uses **one-based round numbers**: `--unlearn-start 51 --unlearn-rounds 10`
means 50 joint rounds, identity unlearning in rounds 51–60, then gender-only
training. The paper refers to a request at round 50 and also says 50 initial
training rounds; this explicit indexing resolves that ambiguity for new runs.
The larger paper settings include 30 clients, full participation, 100 total
rounds, Dirichlet concentration 0.1, batch size 64, one learning/five unlearning
epochs, learning rate 0.001 and 64 centroids. They are configurable; the diagnostic
defaults are deliberately smaller. Empty client partitions fail explicitly.

## Objective and retained-only behavior

Before the request, optimize `CE(identity) + CE(gender)`. During the request,
optimize all four terms in one step on each batch:

```text
CE(gender) - CE(identity) + CE(identity on targeted adversarial inputs)
           + inverted-MAS parameter drift penalty
```

Targets for the offline L2 APGD attacks are randomly selected incorrect identity
labels. Attacks, importance weights and reference parameters are prepared once
per client at request start and reused across its unlearning epochs/rounds.
After the request, only gender cross-entropy supplies learning gradients.

Explicit maintained choices are unit weights for all four terms, Adam with
weight decay 0.01 (the optimizer in paper §5.1), and identity MAS/attack routing
through the shared backbone and identity head. The gender head is excluded from
the identity-specific regularizer. MAS scaling follows this release's recovered
per-parameter normalization. The exact task-specific normalization/head recipe
and pretrained checkpoint are not established by the paper text or recovered code.
These choices are recorded in each result's `task_recipe`.

`--method retrain` starts from the same seeded initialization and trains only
gender from round one. `--method fedavg` is a joint-learning plaintext control.
The identity head remains in both controls for held-out evaluation; neither
control establishes successful erasure by itself.

## Evidence and outputs

Each round records its phase, participants, requesting clients, local objective
components, and separate held-out identity/gender accuracy and loss. JSON stores
annotation provenance, partitions, recipe and manifest hash. The matching `.pt`
contains the shared backbone, both heads and class counts.

Tests exercise generated annotated images through preparation and all three
training phases with real FE, verify live gender gradients during identity
forgetting, fixed offline attacks, retained-only head routing, annotation changes,
and actual unpretrained ResNet/ViT two-head forward passes. They verify execution
and loss routing, not near-random identity accuracy, retained-task benchmark
performance, a privacy guarantee or the paper's historical split.
