# Dataset preparation contract

Run `python scripts/prepare_data.py --help` after installation. The script writes `ROOT/DATASET/manifest.json` together with the files actually opened by training. `--root` and the trainer's `--data-root` always refer to the same **parent** root. Datasets are kept out of Git.

Preparation takes exactly one of `--download`, `--source` and `--validate`. A source is an already extracted directory; user archives are not extracted. Symlinks, existing output directories, class mismatches and malformed inputs are rejected. Imports copy only the selected dataset's consumed directories/files, excluding unrelated data from the source root. They are staged and published only after validation succeeds, leaving source files unchanged. Public downloads use torchvision's official dataset implementations and checksums, without falling back to arbitrary mirrors.

The manifest records the source, preparation convention, split sizes, label-sequence hashes, class-to-index map and SHA256 of every input file. Validation rehashes all files and reopens the splits. Training verifies split/label/class identity and saves the manifest hash; it does not rehash every image in each training round. Run `--validate` immediately before experiments and keep the manifest with results. Exact byte-identical images across ImageFolder train/test splits are rejected; perceptually identical recompressed images are not detected.

## CIFAR-10 and CIFAR-100

Acquire from the [official CIFAR page](https://www.cs.toronto.edu/~kriz/cifar.html), or let preparation download it:

```bash
python scripts/prepare_data.py --dataset cifar10 --root data --download
python scripts/prepare_data.py --dataset cifar10 --root data --validate
```

The expected output is `data/cifar10/cifar-10-batches-py/{data_batch_1,...,data_batch_5,test_batch,batches.meta}` plus the manifest. CIFAR-100 uses `data/cifar100/cifar-100-python/{train,test,meta}`. Official train/test splits are preserved. For an existing extracted download, `--source` points to the directory **containing** `cifar-10-batches-py` or `cifar-100-python`, not to that batch directory itself. The torchvision loader checks official file hashes before reading its Python batch format.

## Explicit image splits

Import existing class folders with:

```bash
python scripts/prepare_data.py --dataset imagefolder --source /path/to/splits --root data
python scripts/prepare_data.py --dataset imagefolder --root data --validate
```

The source layout is:

```text
splits/
  train/class-a/image-001.jpg
  train/class-b/image-002.jpg
  test/class-a/image-101.jpg
  test/class-b/image-102.jpg
```

Every class must have at least one image in both splits. The sorted class directory names define label indices. This adapter supports supplied fixed splits; it does not claim to recover any particular benchmark split or create labels. Sources and labels must be acquired under their applicable terms.

## LFW identity classification

Acquire and extract LFW images from the [dataset provider](https://vis-www.cs.umass.edu/lfw/) or an existing authorized copy. The provider endpoint may be unavailable; this script deliberately imports a local copy instead of selecting an unofficial mirror. Pass the root containing identity directories such as `George_W_Bush/*.jpg`:

```bash
python scripts/prepare_data.py --dataset lfw --source /path/to/lfw --root data \
  --identities 30 --split-seed 2025 --test-fraction 0.2
python scripts/prepare_data.py --dataset lfw --root data --validate
efu-train --dataset lfw --data-root data --model tiny --clients 2 \
  --alpha 10 --rounds 2 --precision 1 --output runs/lfw.json
```

**Maintained convention:** select the requested number of identities with the most images (at least two each; alphabetical ties). For each selected identity, permute sorted filenames with one NumPy RNG seeded by `--split-seed`, hold out the rounded test fraction, and retain at least one example on either side. The manifest records this convention and the selected identity map. An already fixed historical split can instead be supplied with `--dataset imagefolder`.

This enables sample/class unlearning on LFW identities. It does not recover the paper's exact 30 identities or historical split. The recovered source only provides identity labels, a single classifier head and scalar classification/unlearning losses. It contains no supplied gender annotations, second head, task-specific loss routing or separate identity/gender evaluation. The paper's identity-forgetting/gender-retention experiment therefore requires those missing artifacts and a separately validated task-forgetting implementation. Gender labels are not inferred from images or names.

## FEMNIST through LEAF

Use the [LEAF FEMNIST preprocessing instructions](https://github.com/TalwalkarLab/leaf/blob/master/data/femnist/README.md) to obtain the corpus and create explicit JSON splits. A documented upstream example is `./preprocess.sh -s niid --sf 1.0 -k 0 -t sample`; set and record `--smplseed` and `--spltseed` for your run. The upstream corpus/preprocessing is large and is not run during this package's installation. Import its `data` directory:

```bash
python scripts/prepare_data.py --dataset femnist \
  --source /path/to/leaf/data/femnist/data --root data
python scripts/prepare_data.py --dataset femnist --root data --validate
efu-train --dataset femnist --data-root data --model tiny --clients 2 \
  --alpha 10 --rounds 2 --precision 1 --output runs/femnist.json
```

The source must contain `train/*.json` and `test/*.json`. Each shard has `users`, optional `num_samples`, and `user_data[writer] = {"x": [[784 pixel values], ...], "y": [integer label, ...]}`. Pixels must be finite in [0,1], labels in 0–61, sample counts must match, and writer IDs must be unique within each split. All 62 label indices remain fixed, even if a small subset lacks some classes. Pixels are rounded to 8-bit grayscale then converted to RGB for the supported image models.

Default runs repartition the retained training records by seeded label Dirichlet sampling. To preserve natural writer clients, add `--partition writer --clients N`, where N is the number of writers in `manifest.json` under `splits.train.users`. Writers are ordered lexicographically and no writer is dropped. The complete imported corpus and each split's writer IDs remain in the manifest. JSON supplies no original sample IDs, so distinct train/test sample identity cannot be independently established beyond the source split: retain the LEAF preprocessing provenance. A writer occurring in both splits is expected for LEAF's sample-based split.

These are functional maintained ingestion paths, not a recovery of the paper's original FEMNIST shards, LFW identity list, full augmentation pipeline or benchmark numbers.
