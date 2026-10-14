# Code-to-paper mapping and reproduction limits

Reference audited: arXiv:2508.07873v1, Sections 3–5, Algorithms 1–2 and
Appendices A–C. The DOI links the CIKM 2025 publication; no equivalence between
versions is assumed.

| Paper component | Implementation | Status |
|---|---|---|
| Negative forget CE + adversarial CE + inverted MAS, Eq. 3 | `src/efu/client.py`, `utils/mas.py` | Real gradients; scalar-broadcast MAS bug corrected; cached reference/attacks across a request |
| Offline targeted L2 adversarial preparation | `utils/apgd.py`, `prepare_unlearning` | Recovered APGD; raw [0,1] inputs, normalization within image model; configured iterations |
| K-means compression, App. A | `utils/fe.py::cluster` | Recovered K-means, seed explicit, four initializations |
| Independent client keys and partial decryption shares | `coordinate_fe.py`, vendored PyMIFE decentralized DDH | Executed with P-256; no server master secret |
| Encrypted centroid expansion, Algorithms 1–2 | Coordinate-labelled expanded model | Deliberately different; kappa-ciphertext construction remains unresolved |
| Sample/class unlearning, retained learning resumes | `train.py` | Explicit forget IDs; client either learns or unlearns; training resumes on retained IDs |
| PGD, SGA-EWC, FedOSD comparisons | `utils/baselines.py` | Recovered local objective variants; not independently validated against original baseline papers |
| FEMNIST and LFW identity datasets | `dataset_io.py`, `femnist.py`, `train.py` | Maintained imports with explicit split/writer conventions; historical benchmark splits not recovered |
| Two-head identity/gender task forgetting | Missing annotations, second head and task-specific training/evaluation in recovered source | Not implemented or reproduced |

The paper uses Dir(0.1), 64 centroids, 1 learning/5 unlearning epochs, pretrained
ResNet-18 and ViT-B/32, unlearning rounds 50–59, 10 clients at 20% participation
for sample/class experiments, and three-run means. The new harness exposes these
controls where supported but defaults to a tiny diagnostic workload. The
recovered source had class-exclusive partitions, per-batch online attacks,
centralized key generation and different datasets/models; these were not silently
presented as the paper configuration. CIFAR image transforms here are resize and
ToTensor; they are not claimed to reproduce all original augmentation details.

`accuracy` is held-out accuracy on the selected dataset. `forget_train_accuracy` evaluates
the explicitly forgotten training IDs, not forgotten-class test accuracy or an
erasure guarantee. No privacy detector, membership-inference audit, task-wise
forgetting result or counterfactual prediction-gap result has been reproduced.
The `retrain` control starts from the same seeded initialization and excludes the
forget set from the first round; complete paper comparisons remain unexecuted.

## Protocol boundary

The recovered compressed wrappers encrypted every centroid with the same scalar
key and round label. In that construction, subtracting two ciphertexts cancels
the shared mask and exposes a bounded discrete-log encoding of the centroid
difference. The wrappers also held a central master key; the original EncCluster
Flower path sent plaintext centroids and mappings to the server before encryption.
Those paths are **not shipped as runnable options**.

The maintained path uses PyMIFE's existing `FeDDHMultiClientDec` primitive:
public P-256 parameters, independent client secrets, pairwise key exchange,
client-generated functional shares, and server decryption with all shares.
Each expanded model coordinate has a unique session/round/coordinate label.
The server object retains no client private keys. It checks the participant set,
function weights, labels and dimensions. Cryptographic keys use OS entropy;
training seeds do not seed keys. The local simulator holds all parties in one
process and supplies their messages directly; authenticated transport, process
isolation, persistent keys, dropout recovery and distributed key provisioning
are not implemented. Application input checks are not a cryptographic proof.

This conservative mode sends **d ciphertexts per client**, not kappa encrypted
centroids. It therefore cannot substantiate the paper's encryption-time or
communication claims. Reintroducing centroid reuse requires a compatible
multi-message/label construction and a security argument covering substitution;
changing only the tag to a centroid index is insufficient when different clients
select different centroid indices at the same model position.

PyMIFE is a research library whose authors explicitly state that it is not
peer-reviewed. Tests establish arithmetic and application invariants, not a
security certification. The current release is a research implementation with
these stated differences, not a complete reproduction of the paper protocol.

## Numeric contract

Clients agree on the active cohort, positive integer sample counts, model layout,
fixed-point precision and public absolute weight bound before setup. Each scalar
is rounded to the nearest integer after multiplication by `10**precision`.
Decryption returns the sample-weighted mean of those fixed-point values.
Quantization alone introduces at most `0.5 / 10**precision` absolute error per
coordinate relative to the reconstructed clustered mean. K-means/filter error
is additional. Oversized/nonfinite values are rejected, not silently clipped.
The bounded discrete-log search can be expensive for large sample counts,
precision or models. This execution mode is intended for bounded experiments.
BatchNorm running statistics are aggregated in plaintext; integer counters
retain the source convention. These buffers are outside the encrypted guarantee.

## Reproducibility

Python 3.12 on Apple Silicon was used for release checks. Supported metadata is
Python 3.11–3.13; other Python/platform combinations have not been executed here.
`constraints.txt` records the tested dependency closure. Model/training,
partition and clustering randomness is seeded; crypto randomness remains fresh.
The JSON output records CLI settings, exact partition indices, forget indices
when applicable, participant IDs, sample counts, losses and accuracies. It does
not contain input records, weights, private keys or claims of benchmark parity.

The default two-round synthetic test uses two clients, eight training samples per
client, three classes, a 4-to-3 linear model, four K-means centroids, precision 2,
CPU and one local epoch. No data/model download occurs on import, installation
or test. `scripts/prepare_data.py --download` explicitly fetches CIFAR; `--source` imports acquired data. `--pretrained` explicitly allows
weights to be fetched for supported EFU image models. No paper checkpoints or
paper-result logs are included. No GPU run or full paper benchmark was executed.

The CLI supports Dirichlet label partitioning and optional FEMNIST writer partitions, explicit
participation, local epochs, learning rate, seed, model, precision, public bound
and output path. It rejects empty clients and cohorts with fewer than two active
participants. A changed cohort gets a fresh decentralized setup. Setup cost is
included in the current simulator and is not comparable to the paper timings.

## Release artifact protection

The maintained CLI rejects an existing output JSON or sibling checkpoint before training. Final `.pt` files contain CPU state dictionaries, configuration and class count, loadable with `torch.load(path, weights_only=True, map_location="cpu")`. The JSON includes Python, platform, torch and NumPy versions. Save the dependency constraints and input-data version alongside the run. Encryption key generation remains random; seeded training does not imply deterministic ciphertext bytes.

Partial-participation unlearning explicitly includes the requesting client in every scheduled request round and samples the remaining cohort uniformly. The metrics record `unlearning_client`; no request round is silently skipped because of ordinary client subsampling. This scheduler is a documented release choice, not a recovered original experiment schedule.

## Dataset setup follow-up

[DATASETS.md](DATASETS.md) defines the maintained preparation contract. Training now consumes `ROOT/DATASET` roots created by `scripts/prepare_data.py`; older direct CIFAR roots can be imported with `--source`. The preparation manifest identifies class maps, split sizes, label order and file hashes. Reports include its SHA256. Generated source-format fixtures exercise import, validation and actual training; their metrics are correctness diagnostics. No newly implemented ingestion path implies recovered historical splits or reproduced benchmark results.
