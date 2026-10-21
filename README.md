# RR-QPSO for Quantum Molecular Generation — arXiv v1

Public research archive for **Rank-Refined Quantum-Behaved Particle Swarm Optimization for Quantum Molecular Generation**,
by Sing-Yun Wu, Sheng Yun Wu, I-Min Chiang and Tai-Yue Li.

[arXiv:2607.10284v1](https://arxiv.org/abs/2607.10284v1) · [Paper PDF](paper/RR-QPSO_arXiv-2607.10284v1.pdf)

`main` contains the published v1 code, original figures and data, plus this publication index.
The original scientific snapshot is preserved without modification at annotated tag **`v1-preprint`**,
commit **`a5719115efb1a782fa66d879f8155c7ccd39d9ca`**. This snapshot includes data/figure packaging committed on July 16 after the July 11 preprint.
Subsequent manuscript development is maintained separately and is not part of this public archive.

| Material | Location |
|---|---|
| Published manuscript | [paper/](paper/) |
| Original README, methods, usage and figures | [docs/ARXIV_V1_README.md](docs/ARXIV_V1_README.md) or `git show v1-preprint:README.md` |
| Original optimizer and unconditional runner | `qpso_optimizer_ae.py`, `run_qpso_qmg_cudaq.py` |
| Property-guided runner | `run_qpso_qmg_cudaq_hbahbd_multiobj.py` |
| QMG circuit and decoder | `qmg/` |
| Published figure assets | `figures/` |
| Tracked particle-sweep data | `results/fig3_particle_sweep/`, `results_v8/` |
| Tracked multi-objective data | `results_hbahbd_multiobj/` |
| Complete original single-run data and PDF | [arxiv-v1 release](https://github.com/SiriusW823/sqmg_project-cudaq/releases/tag/arxiv-v1) |
| Data filenames, sizes and SHA-256 hashes | [docs/ARXIV_V1_DATA_MANIFEST.csv](docs/ARXIV_V1_DATA_MANIFEST.csv) |

## Reproduce the historical version

```bash
git clone https://github.com/SiriusW823/sqmg_project-cudaq.git
cd sqmg_project-cudaq
git checkout v1-preprint
```

Use the original CUDA-Q 0.7.1 / NumPy <2 / RDKit environment and NVIDIA V100 GPUs.
Original commands and configurations are in the preserved README at the tag. Source files and numerical results were not rewritten during organization.
The historical README is retained as a historical record; its original scientific claims should not be treated as validation of later work.
The single-run comparisons are descriptive and do not establish equal-budget optimizer superiority.

## Citation

Sing-Yun Wu, Sheng Yun Wu, I-Min Chiang, Tai-Yue Li. *Rank-Refined Quantum-Behaved Particle Swarm Optimization for Quantum Molecular Generation*.
arXiv:2607.10284v1 (2026). [DOI: 10.48550/arXiv.2607.10284](https://doi.org/10.48550/arXiv.2607.10284).
