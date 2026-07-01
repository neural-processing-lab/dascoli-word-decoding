# LibriBrain100 Sentence Decoding Baseline (d'Ascoli et al. 2025)

This repository adapts the sentence-decoding code from d'Ascoli et al.,
"Towards decoding individual words from non-invasive brain recordings"
(Nature Neuroscience, 2025), for the 2026 PNPL LibriBrain100 competition.

The default experiment is intentionally simple: train a
brain-to-text-embedding model on the first ten Sherlock sessions, validate on
the next Sherlock session, and test on a held-out Sherlock session.

## What It Does

- Loads LibriBrain100 through the `pnpl` package.
- Converts PNPL LibriBrain100 records into `neuralset` timelines.
- Trains the existing sentence-decoding pipeline with the ConvNet plus Transformer.
- Evaluates word retrieval from predicted text embeddings.
- Reports the PNPL-style metric: top-10 balanced accuracy on the 50-word
  LibriBrain100 evaluation vocabulary.

## Repository Map

```text
sentence_decoding/              Training loop, metrics, grids
sentence_decoding/grids/        Runnable experiment configs
neuralset/                      Dataset loading and feature extraction
neuraltrain/                    Model, optimizer, loss, and metric utilities
runs/                           Local caches and results, if SAVEPATH points here
```

The main LibriBrain100 entry point is:

```text
sentence_decoding/grids/libribrain100.py
```

## Quick Start

Create an environment:

```bash
conda create -n neuralset python=3.11
conda activate neuralset
pip install -U torch==2.3.1 torchvision==0.18.1
```

Install the local packages:

```bash
cd neuralset
pip install --config-settings editable_mode=strict -e '.[dev]'
cd ..

cd neuraltrain
pip install --config-settings editable_mode=strict -e .
cd ..
```

Install the experiment dependencies:

```bash
pip install pnpl lightning wandb x_transformers kenlm transformers sentencepiece
```

Set paths for data, caches, and results:

```bash
export SAVEPATH=/Users/hans/dascoli/runs
export DATAPATH=/Users/hans/dascoli/data
export LIBRIBRAIN100_PATH=/Users/hans/dascoli/libribrain100

mkdir -p "$SAVEPATH/cache" "$SAVEPATH/results" "$DATAPATH" "$LIBRIBRAIN100_PATH"
```

Run the LibriBrain100 baseline:

```bash
python -m sentence_decoding.grids.libribrain100
```

The first run may download LibriBrain100 files and compute text embeddings, so it
can spend a while preparing data before training begins.

## Default Experiment

By default, `sentence_decoding.grids.libribrain100` runs:

| Split | Subject | Task | Session | Run |
| --- | --- | --- | --- | --- |
| Train | sub-0 | Sherlock1 | 1-10 | 1 |
| Validation | sub-0 | Sherlock1 | 11 | 2 |
| Test | sub-0 | Sherlock1 | 12 | 2 |

The model uses:

- neural input: MEG windows around each word
- target: `t5-large` text embeddings
- loss: SigLIP-style contrastive loss
- architecture: SimpleConvTimeAgg plus Transformer
- metric: embedding retrieval

## Main Metric

For LibriBrain100, the key metric is:

```text
val_retrieval_acc10_vocab=libribrain50_macro_0
test_retrieval_acc10_vocab=libribrain50_macro_0
```

This is top-10 balanced accuracy, macro-averaged over the observed words from
the 50-word LibriBrain100 evaluation vocabulary:

```text
is the a to it i not was we be he that have this they of there and are in but
will so all my for she were any really at out our am its had him an very has do
can time think good always new people as on
```

The older 250-most-frequent-word retrieval metric is not used by the
LibriBrain100 grid.

## Changing Splits

The default split is deliberately focused. To change it, edit:

```text
sentence_decoding/grids/libribrain100.py
```

The important knob is `data.query`. For example:

```python
"data.query": "subject == 'LibriBrain100/0' and corpus == 'sherlock'"
```

That uses all deep-subject Sherlock records selected by the query, while keeping
the train/validation/test split annotations provided by PNPL.

For all available sub-0 LibriBrain100 records:

```python
"data.query": "subject == 'LibriBrain100/0'"
```

## Useful Knobs

Disable Weights & Biases:

```python
"use_wandb": False
```

Change the LibriBrain100 data directory:

```bash
export LIBRIBRAIN100_PATH=/path/to/libribrain100
```

Change where caches and results are written:

```bash
export SAVEPATH=/path/to/runs
export DATAPATH=/path/to/data
```

## Outputs

Results are written under:

```text
$SAVEPATH/results/sentence_decoding/libribrain100/
```

Useful artifacts include:

- `config.yaml`: the exact config used for the run
- `best.ckpt` and `last.ckpt`: model checkpoints
- `retrieval_outputs/`: saved retrieval predictions and targets
- `decoded_sentences/`: decoded text snapshots from validation/test

## Notes

- This is a research baseline, not a polished competition submission script.
- The grid runs locally by default for LibriBrain100.
- PNPL handles LibriBrain100 file resolution and download.
- If a local run prints `Done.` too quickly, inspect the saved job metadata in
  the corresponding results folder; the local task runner can cache failed jobs.

## Original Code

The original README from the sentence-decoding codebase is preserved as
`README_ORIGINAL.md`.
