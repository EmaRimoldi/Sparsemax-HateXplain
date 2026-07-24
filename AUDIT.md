# Repository audit

Audit date: 2026-07-24
Original code snapshot: `MicheleSmaldone/HateXplain@34ca908`

## Material findings

| Severity | Finding in the original snapshot | Consequence | Resolution here |
|---|---|---|---|
| High | Class weights were computed from `test`, not `train`. | Test-label leakage affects the training objective and weakens the archived classification comparison. | Weights are derived only from training labels. |
| High | Checkpoints and paired LIME JSONL files underlying the paper tables were absent. | Historical results cannot be recomputed or independently checked. | Results are explicitly marked archival; new runs persist configuration, metrics, and checkpoints. |
| High | Softmax and sparsemax lived on different branches and had different epoch/configuration overrides. | The reported comparison was not controlled by one configuration path. | Both variants use one code path and matched JSON configurations. |
| Medium | `variance * at_mask` repeated a Python list before `numpy.mean`; it did not scale the scores. | The configured value 5 had no effect on rationale targets. | The parameter was removed. |
| Medium | The softmax branch supervised post-softmax attention with another log-softmax, while sparsemax supervised raw attention scores with a different loss. | The historical comparison changes target normalization, score source, and loss simultaneously. | This behavior is preserved only in the archived reproduction; the core factorial fixes raw scores and crosses target, loss, and semantic content. |
| Medium | The training entry point overwrote JSON values (`epochs`, `variance`, saving, and lambda). | Configuration files were not authoritative. | Validated JSON is authoritative and serialized with every run. |
| Medium | LIME scope was inconsistent: code sliced 300 rows, the shell script requested 400 perturbations, and the README described 1,000/523 examples. | Sample size and perturbation count could not be inferred reliably from the repository. | Comparison consumes explicit JSONL paths, pairs IDs, and reports the exact sample count. |
| Medium | Majority vote used `max(set(labels), key=labels.count)`. | Ties were dependent on set iteration order. | A two-of-three majority is required; ties return `None`. |
| Low | The attention-hook implementation retained mutable state and duplicated hundreds of commented lines. | Harder review, brittle maintenance, and stale graph state risk. | Raw scores are recomputed explicitly from the selected layer's query/key projections. |

## Removed from the active research surface

The original worktree was 102 MB and 176 tracked files. The cleaned repository
retains the official dataset/split, license, focused implementation, configurations,
tests, and documentation. The following were intentionally excluded:

- tracked W&B offline binaries and copied environment files;
- generated hidden-state CSV files and plots;
- the vendored ERASER checkout;
- duplicated `Models/Research` notebooks and scripts;
- legacy non-BERT CNN/RNN models, GloVe conversion utilities, and generated vocabularies;
- notebook checkpoints, debug prints, large commented-out implementations, and
  one-off cluster scripts;
- stale environment specifications mixing incompatible TensorFlow, Keras, PyTorch,
  and Transformers stacks.

The full original Git history and all remote branches were saved locally before
reinitialization as `tmp/backups/HateXplain-source-34ca908.bundle` in the parent
workspace. The original worktree is also retained locally beneath
`tmp/backups/HateXplain-original-worktree/`.

## Facts carried into the paper

- BERT self-attention is still normalized with softmax.
- Sparsemax is used for non-normal rationale targets and for the auxiliary loss on
  raw final-layer `[CLS]` scores.
- Six heads in zero-indexed layer 11 are supervised.
- The official split contains 15,383/1,922/1,924 train/validation/test examples.
- The historical sparsemax script effectively trained for ten epochs with
  `lambda = 0.001`.
- The paper's numerical results remain archival because their model and explanation
  artifacts are missing.
- The controlled protocol excludes normal posts from auxiliary rationale
  supervision; the historical uniform target is preserved only in archival configs.
- Human and deterministic length-matched random rationales are first-class
  configuration choices, as are mean, majority, union, and per-annotator masks.

## Remaining risks

- The refactored training pipeline has unit and smoke coverage but has not yet been
  run through a full GPU training cycle.
- The two historical variants do not isolate one causal factor; they must not be
  pooled with the new `2 x 2 x 2` factorial.
- The faithfulness, second-encoder, external-dataset, and per-annotator stages are
  intentionally gated and do not yet have runnable implementations.
- LIME is a post-hoc explainer; the new protocol therefore requires independent
  gradient-based attributions, multiple perturbation operators, random baselines,
  and normalized curves.
