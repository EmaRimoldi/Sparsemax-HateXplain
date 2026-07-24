# Sparsemax-supervised attention on HateXplain

This is an independent, cleaned research repository for comparing two
human-rationale supervision objectives on BERT:

- `softmax`: the historical baseline, which applies cross-entropy to final-layer,
  post-softmax `[CLS]` attention;
- `sparsemax`: sparsemax targets and the sparsemax Fenchel--Young loss applied to
  raw final-layer `[CLS]`-to-token scores.

In both variants BERT's internal self-attention remains softmax. Six heads in the
last encoder layer are supervised, while classification uses class-balanced
cross-entropy. The refactored protocol uses the same data split, seed, epoch count,
and optimizer settings for both variants.

> **Content warning:** `data/dataset.json` contains hateful and offensive language.
> It is retained only to make the experiment reproducible.

## Status

The source code and official HateXplain split are present. The historical
checkpoints and LIME JSONL outputs were not preserved, so the numerical tables in
the accompanying paper have **not** been reproduced. See [AUDIT.md](AUDIT.md) for
the evidence, corrections, and remaining limitations.

## Setup

Python 3.10--3.13 is supported.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

## Dataset check

```bash
hatexplain-stats \
  --dataset data/dataset.json \
  --splits data/post_id_divisions.json
```

Expected split sizes are 15,383 train, 1,922 validation, and 1,924 test posts.
Posts without a two-of-three label majority are absent from the official split.

## Training

```bash
hatexplain-train --config configs/softmax.json
hatexplain-train --config configs/sparsemax.json
```

The best validation checkpoint and JSON metrics are written beneath `runs/`, which
is intentionally ignored by Git. Run metadata includes the full configuration and
the exact split counts. Class weights are computed from the **training** split.

## Comparing paired LIME outputs

After generating ERASER-style JSONL outputs for the same post IDs:

```bash
hatexplain-compare-lime \
  --dataset data/dataset.json \
  --splits data/post_id_divisions.json \
  --softmax lime_results/softmax.jsonl \
  --sparsemax lime_results/sparsemax.jsonl
```

The comparison aligns records by `annotation_id`; mismatched ID sets fail fast
instead of silently comparing rows in a different order.

## Provenance and attribution

The dataset, split, and parts of the experimental design derive from
[HateXplain](https://github.com/punyajoy/HateXplain), released with the MIT license
included here. Please cite:

> Binny Mathew, Punyajoy Saha, Seid Muhie Yimam, Chris Biemann, Pawan Goyal, and
> Animesh Mukherjee. “HateXplain: A Benchmark Dataset for Explainable Hate Speech
> Detection.” AAAI 2021.

This repository was initialized with new Git history and is not a GitHub fork.
