# Attribution and licenses

## Source attribution

- EFU implementation by Samaneh Mohammadi and the EFU authors: [SamaneMohammadi/EFU](https://github.com/SamaneMohammadi/EFU/tree/e8837be2eed617d9d18c9222ebd7aa3480a264b4), commit `e8837be2eed617d9d18c9222ebd7aa3480a264b4`.
- Decentralized FE primitive: [MechFroG88/PyMIFE](https://github.com/MechFroG88/PyMIFE/tree/69938282ed17fd2f1ddb11102c1480ad6dc5c04b), commit `69938282ed17fd2f1ddb11102c1480ad6dc5c04b`, package 0.0.14. Its source and MIT license are included in `modules/mife`.
- APGD adapts [fra31/auto-attack](https://github.com/fra31/auto-attack), copyright 2020 Francesco Croce, under the MIT license in `licenses/AUTOATTACK-LICENSE`.

[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) identifies upstream file paths and hashes. Imported files retain attribution and modification notices. The `efu` package includes the training CLI, coordinate-labelled protocol, dataset preparation and dual-task integration; [implementation details](REPRODUCIBILITY.md) describe these components.

## License records

The repository's [LICENSE](LICENSE) is Apache-2.0. The EFU source license is also preserved in [licenses/UPSTREAM-LICENSE](licenses/UPSTREAM-LICENSE). PyMIFE retains its MIT license and copyright 2023 Tan Kel Zin. An earlier MIFE snapshot's Apache license text is retained in `licenses/MIFE-SNAPSHOT-LICENSE`; the bundled canonical PyMIFE component is governed by its own MIT notice.

[NOTICE](NOTICE) collects component attribution. Dataset and pretrained-model access terms are separate from software licenses.
