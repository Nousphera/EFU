# EFU

**EFU: Enforcing Federated Unlearning via Functional Encryption**

[Paper](https://doi.org/10.1145/3746252.3761091) · [Preprint](https://arxiv.org/abs/2508.07873) · [Project](https://nousphera.github.io/EFU/) · [Citation](CITATION.cff)

## Abstract

Federated unlearning (FU) algorithms allow clients in federated settings to exercise their “right to be forgotten” by removing the influence of their data from a collaboratively trained model. Existing FU methods maintain data privacy by performing unlearning locally on the client-side and sending targeted updates to the server without exposing forgotten data; yet they often rely on server-side cooperation, revealing the client's intent and identity without enforcement guarantees - compromising autonomy and unlearning privacy. In this work, we propose EFU (Enforced Federated Unlearning), a cryptographically enforced FU framework that enables clients to initiate unlearning while concealing its occurrence from the server. Specifically, EFU leverages functional encryption to bind encrypted updates to specific aggregation functions, ensuring the server can neither perform unauthorized computations nor detect or skip unlearning requests. To further mask behavioral and parameter shifts in the aggregated model, we incorporate auxiliary unlearning losses based on adversarial examples and parameter importance regularization. Extensive experiments show that EFU achieves near-random accuracy on forgotten data while maintaining performance comparable to full retraining across datasets and neural architectures - all while concealing unlearning intent from the server. Furthermore, we demonstrate that EFU is agnostic to the underlying unlearning algorithm, enabling secure, function-hiding, and verifiable unlearning for any client-side FU mechanism that issues targeted updates.

*The abstract describes the paper. This maintained research implementation uses decentralized coordinate-labelled FE. It does not reproduce the paper's compressed-ciphertext costs or establish its security claims.* See [implementation scope](REPRODUCIBILITY.md).

## Setup

Python 3.12 is tested; package metadata permits 3.11–3.13. Use a dedicated virtual environment. Linux needs a C/C++ toolchain, Python development headers, GMP and libffi development libraries (for example, `build-essential python3-dev libgmp-dev libffi-dev` on Debian/Ubuntu). macOS needs Python 3.12 and the standalone Command Line Tools; install GMP/libffi if native cryptographic wheels are unavailable. On Windows, use WSL2 with the Linux prerequisites; native Windows is untested.

```bash
git clone https://github.com/Nousphera/EFU.git
cd EFU
bash scripts/setup.sh .venv
source .venv/bin/activate
python -m pytest -q
```

The setup script creates the venv, installs `requirements.txt` with the tested `constraints.txt`, and runs `pip check`. It selects installed macOS Command Line Tools without accepting an Xcode license. Equivalent manual setup:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# macOS with standalone Command Line Tools:
# export DEVELOPER_DIR=/Library/Developer/CommandLineTools
python -m pip install -c constraints.txt -r requirements.txt
```

Requirements install the vendored PyMIFE source and native cryptographic dependencies. Installation and tests do not download datasets or model weights. See [source attribution](PROVENANCE.md).

## Datasets

Prepare once, validate, and pass the same parent root to training:

```bash
python scripts/prepare_data.py --dataset cifar10 --root data --download
python scripts/prepare_data.py --dataset cifar10 --root data --validate
```

This fetches the official [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) Python data through torchvision's checksum-verified loader, preserving its 50,000/10,000 train/test split. Use `cifar100` for CIFAR-100. Output lives under `data/cifar10/`, including `manifest.json` with file SHA256 hashes, class mapping and split sizes. To import an existing torchvision CIFAR root, use `--source /path/to/extracted-cifar-root` instead of `--download`. Existing prepared roots are never overwritten.

For EFU's additional datasets, import acquired LFW identity folders or LEAF FEMNIST JSON:

```bash
python scripts/prepare_data.py --dataset lfw --source /path/to/lfw --root data --identities 30 --split-seed 2025
python scripts/prepare_data.py --dataset femnist --source /path/to/leaf/data/femnist/data --root data
python scripts/prepare_data.py --dataset femnist --root data --validate
```

LFW uses a documented, seeded identity-classification split; FEMNIST retains writer IDs and all 62 label indices. These imports feed the same learning/unlearning trainer. See [dataset acquisition and split conventions](DATASETS.md); the historical split and two-head identity/gender task-forgetting experiment are not reconstructed.

For an explicit image dataset split, use `--dataset imagefolder --source /path/to/splits`, where the source contains `train/CLASS/*` and `test/CLASS/*`. Validation checks class agreement, readable images, exact duplicate image content across splits and file changes. [DATASETS.md](DATASETS.md) gives the complete layout and limits; `python scripts/prepare_data.py --help` lists all options.

## Run

A two-round offline check trains real client models and executes real P-256 functional encryption:

```bash
efu-train --output runs/smoke.json
```

Run the prepared CIFAR data through the same primary trainer:

```bash
efu-train --dataset cifar10 --data-root data --model tiny \
  --clients 2 --rounds 2 --alpha 10 --precision 1 --output runs/cifar10.json
efu-train --help
```

`--method fedavg|retrain` provides plaintext controls. Define a request with `--forget-client`, `--forget-fraction` or `--forget-class`, `--unlearn-start`, `--unlearn-rounds` and `--unlearn-epochs`. Scheduled unlearning always includes the requester. `--attack-steps 100` selects the recovered APGD iteration count; the diagnostic default is 2. Models `resnet18` and `vit_b_32` support explicit `--pretrained` weight downloads. `--partition writer` preserves FEMNIST writers as clients and requires `--clients` to equal the manifest's training-writer count. The default is seeded Dirichlet label partitioning.

The maintained runner is a local federated simulation, with all parties in one process. Its input and aggregation paths are described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Outputs and reproducibility

Each named run saves JSON containing configuration, data-manifest hash, exact client partitions, participant/sample counts, metrics and environment versions. A sibling `.pt` stores the final CPU state dictionary and configuration, loadable with `torch.load(..., weights_only=True)`. Existing JSON/checkpoint paths are protected; choose a new output name to rerun. Save the input manifest and run `--validate` before a research experiment; training checks label/split identity, while full file hashing is an explicit preparation check.

**The release encrypts expanded coordinates (d ciphertexts), not kappa centroids.** It retains the local learning/unlearning operations while avoiding mask reuse in the recovered compressed wrapper. It does not establish paper benchmark results, cryptographic security, network isolation or dropout recovery. Full-model encrypted runs are expensive and unbenchmarked. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) separates paper settings, implemented behavior and remaining gaps.

## Citation

```bibtex
@inproceedings{mohammadi2025efu,
  title = {{EFU}: Enforcing Federated Unlearning via Functional Encryption},
  author = {Mohammadi, Samaneh and Tsouvalas, Vasileios and Symeonidis, Iraklis and Balador, Ali and Ozcelebi, Tanir and Flammini, Francesco and Meratnia, Nirvana},
  booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
  year = {2025},
  pages = {2148--2158},
  publisher = {Association for Computing Machinery},
  doi = {10.1145/3746252.3761091},
  url = {https://doi.org/10.1145/3746252.3761091}
}
```

## License and attribution

See [LICENSE](LICENSE), [NOTICE](NOTICE), and [PROVENANCE.md](PROVENANCE.md) for the first-party and third-party terms and exact source commits. Imported code retains its original license notices; these may differ from the repository's root license.
