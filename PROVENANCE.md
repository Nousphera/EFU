# Provenance and licensing

## Recovery sources

- Authors' public implementation: [SamaneMohammadi/EFU](https://github.com/SamaneMohammadi/EFU/tree/e8837be2eed617d9d18c9222ebd7aa3480a264b4), commit `e8837be2eed617d9d18c9222ebd7aa3480a264b4`.
- Maintained decentralized primitive: [MechFroG88/PyMIFE](https://github.com/MechFroG88/PyMIFE/tree/69938282ed17fd2f1ddb11102c1480ad6dc5c04b), commit `69938282ed17fd2f1ddb11102c1480ad6dc5c04b`, package version 0.0.14. Its complete source snapshot and original MIT license are in `modules/mife`.
- The existing Nousphera history, `docs/` project website and paper citation are preserved. Full recovery snapshots remain separately archived during preparation; this release contains the selected executable source, not generated build artifacts, binaries, checkpoints or data.

`SOURCE_MANIFEST.json` records upstream paths, original hashes and released hashes.
Source moved into the `efu` Python namespace. Modified imported files carry
notices; new harness/protocol/test/documentation files are maintenance additions.

## Changes from the recovered implementation

- Replaced central-master-key / repeated centroid-label wrappers with the existing decentralized PyMIFE primitive and unique coordinate labels.
- Added an explicit numeric/shape/participant contract, sample-weighted verification and a no-download seeded training path.
- Added Python package metadata, constraints, tests, provenance and reproducibility documentation.
- Removed nonfunctional launchers and machine-specific data paths from the released tree; source commits above preserve their history.
- Corrected elementwise inverted MAS and offline attack preparation/cache; model normalization stays inside the attacked model.

## License records

The repository and recovered EFU root `LICENSE` contain Apache-2.0; the recovered
README's MIT statement was inconsistent with its license file and is not reused.
`licenses/UPSTREAM-LICENSE` preserves that source text. The older bundled mife
snapshot supplied Apache text while declaring MIT metadata; it is replaced by
canonical PyMIFE with its actual MIT copyright notice, and the old text is
retained in `licenses/MIFE-SNAPSHOT-LICENSE` for provenance.

`utils/apgd.py` identifies its adaptation from [fra31/auto-attack](https://github.com/fra31/auto-attack).
The original MIT notice (copyright 2020 Francesco Croce) is included as
`licenses/AUTOATTACK-LICENSE`. PyMIFE retains copyright 2023 Tan Kel Zin.

## Dataset/setup follow-up (2026-10-15)

The maintained release adds `scripts/setup.sh`, `scripts/prepare_data.py`, namespaced dataset preparation/validation, manifests and generated-source integration tests. These modules are new integration code rather than recovered historical experiment scripts. Official dataset loading remains delegated to torchvision; fixed-split imports remain user-supplied. See `DATASETS.md` for split conventions and `REPRODUCIBILITY.md` for method/benchmark limits.
