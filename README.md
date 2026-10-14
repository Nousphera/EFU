# EFU

**Enforcing Federated Unlearning via Functional Encryption**

[Paper](https://doi.org/10.1145/3746252.3761091) · [Project](https://nousphera.github.io/EFU/) · [Citation](CITATION.cff) · [Provenance](PROVENANCE.md)

Client-side unlearning with negative cross-entropy, offline targeted adversarial examples, inverted MAS regularization and encrypted weighted aggregation.

This repository recovers the authors' research code and provides a tested local
simulation. The default encryption path uses decentralized, coordinate-labelled
FE to avoid defects in the recovered compressed wrapper. **It does not reproduce
the paper's kappa-ciphertext cost or establish its security claims.** See
[code-to-paper mapping and limits](REPRODUCIBILITY.md) before using results.

## Install and run

Use Python 3.11–3.13 in a dedicated environment (tested: 3.12).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints.txt -r requirements.txt
efu-train --output runs/smoke.json
python -m pytest -q
```

The default is a small, two-round CPU experiment with generated data. It trains
real client models and performs real P-256 FE operations. The output JSON records
configuration, partition indices and measured metrics. No dataset is downloaded.

For options and a CIFAR experiment with the compact model:

```bash
efu-train --help
efu-train --dataset cifar10 --download --model tiny --clients 2 \
  --rounds 2 --alpha 10 --precision 1 --output runs/cifar10.json
```

Use `--unlearn-start`, `--unlearn-rounds`, `--unlearn-epochs`,
`--forget-client`, and either `--forget-fraction` or `--forget-class` to define a
request. Offline attacks are prepared once and retained across unlearning rounds;
`--attack-steps 100` selects the full recovered APGD iteration count. The small
default uses 2 steps only for the smoke test. `--method fedavg` and `--method
retrain` provide plaintext controls. `--unlearn-method pgd|sga_ewc|fedosd` selects
recovered alternative local objectives; see their scope limits in the mapping.
Image models `resnet18` and `vit_b_32` are supported; coordinate FE for full models
is expensive and has not been benchmarked here.

## Attribution

The existing project website and citation files are preserved. Original research
code, vendored cryptographic primitives and native filters keep their source
attribution and license texts. [PROVENANCE.md](PROVENANCE.md) records exact source
commits, changes and remaining licensing/reproduction review items.
