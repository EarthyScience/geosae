# Scripts of *Decomposing Earth Embeddings with Sparse Autoencoders* (NeurIPS 2026)

One script per figure or table of the paper, plus `build_store.py`, which builds the paper's data
store. Each figure script draws either from fresh `geosae` stage outputs or, with
`--paper-data <dir>`, from the camera-ready inputs on Zenodo
([doi:10.5281/zenodo.24046835](https://doi.org/10.5281/zenodo.24046835)).

The commands for every figure and table are on the documentation page
[Reproducing the figures and tables](https://earthyscience.github.io/geosae/neurips2026/reproduce-figures/)
(source: [`docs/neurips2026/reproduce-figures.md`](../../docs/neurips2026/reproduce-figures.md)):
the training cells, inference, evaluation stages and plotting command of each, and the
`--paper-data` alternative. The data store, checkpoints and training sweep are on
[Reproducing: data, checkpoints and training](https://earthyscience.github.io/geosae/neurips2026/reproduce-data/).

Every script takes `--out <dir>` and writes a PDF and a PNG there, plus a JSON with the numbers the
paper quotes; the table scripts also write a LaTeX tabular. Run
`python scripts/neurips2026/<script>.py -h` for its arguments.
