# Experimental programme

This directory separates the archived course-project reproduction from the new
causal evaluation. The primary question is no longer whether sparsemax "looks
interpretable", but which of three factors causes any measured improvement:

1. **target geometry**: softmax or sparsemax normalization of the annotation mask;
2. **loss geometry**: softmax Fenchel--Young (cross-entropy) or sparsemax
   Fenchel--Young loss;
3. **semantic content**: human or length-matched random rationales.

All core cells supervise the same raw `[CLS]` attention scores. This removes the
historical confound between post-softmax attention values and pre-normalization
scores.

## Directory map

| ID | Purpose | Training matrix |
|---|---|---:|
| `00_archived_reproduction` | Reproduce the two historical implementations without treating them as a controlled comparison | 2 conditions x 5 seeds |
| `01_core_factorial` | Estimate target, loss, content, and interaction effects | 2 x 2 x 2 x 5 seeds |
| `02_lambda_frontier` | Trace utility--plausibility response curves on validation data | 2 x 2 x 5 lambda values x 1 pilot seed |
| `03_controls` | Unsupervised-attention and SRA-style MSE reference conditions | 2 conditions x 5 seeds |
| `04_faithfulness` | Independent attribution, perturbation, randomization, and NAOPC evaluation | Evaluation-only |
| `05_robustness` | Second encoder/dataset, ID/OOD, and synthetic spurious-correlation tests | Gated follow-up |
| `06_annotators` | Mean, majority, union, and per-annotator sensitivity | Staged follow-up |

The pilot lambda sweep selects a small, fixed set of values for the confirmatory
multi-seed run; test data must not be used for this choice. `protocol.md` defines
outcomes, statistical comparisons, and stop/go gates.

## Materialize configurations

Install the package in editable mode, then generate every currently executable
matrix:

```bash
python experiments/materialize.py --all
```

Or generate one matrix:

```bash
python experiments/materialize.py experiments/01_core_factorial/matrix.json
```

Generated JSON files are written to `experiments/generated/` and are ignored by
Git. Each file is accepted by the standard trainer:

```bash
hatexplain-train \
  --config experiments/generated/01_core_factorial/01_core_factorial__target_normalization-sparsemax__attention_loss-sparsemax__rationale_source-human__seed-42.json
```

The checked-in manifests, base configuration, source revision, environment lock,
and saved metrics are the reproducibility record. Model checkpoints and generated
run directories remain local.
