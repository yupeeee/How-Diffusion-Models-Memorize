# How Diffusion Models Memorize

Experiments on diffusion-model memorization: proximity, Theorem 1, Lemma 2,
Corollary 3, and forward-corrupted versus generated states.

## Run

From the repository root, in your Python environment:

```bash
python -m pip install -r requirements.txt
./run_all.sh --download
```

This prepares the data and runs the pipeline. Defaults: **20 seeds, 50 steps,
guidance 7.5, GMM selection**. Runs SDv1/DDIM, SDv1/DDPM, SDv2/DDIM, and
RealVis/DDIM, each with zero and model-implied-mean centering.

```bash
# Run again, reusing compatible caches
./run_all.sh

# Run just SDv1/DDIM with zero centering
./run_all.sh --model sdv1 --scheduler ddim --no-mu

# Regenerate figures only; matching saved results must already exist
./run_all.sh --plot

# All options
./run_all.sh --help
```

Use `--use-mu` or `--no-mu` to run only one centering mode.
Change defaults with `--N`, `--T`, and `--g`.
All visible CUDA GPUs are used automatically; restrict them with
`CUDA_VISIBLE_DEVICES=0`. Use `--overwrite` only to force recomputation.

## Outputs

- `outputs/`: figures, result tables, and example images, grouped by run.
- `logs/`: reusable trajectories, predictions, scores, and computation caches.
- `data/webster/`: recovered benchmark data and frozen prompt selections.
