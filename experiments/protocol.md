# Confirmatory protocol

## Primary estimands

- Classification utility: test macro-F1, with accuracy and per-class F1 secondary.
- Rationale plausibility: token average precision and token F1 against held-out
  human rationales.
- Selectivity: human-rationale supervision minus the paired length-matched random
  condition within each target/loss cell.
- Faithfulness: independent-attribution comprehensiveness and sufficiency,
  normalized perturbation curves, and model-randomization sensitivity.

The core analysis fits a factorial model with target geometry, loss geometry, and
rationale content as fixed effects and seed as a paired block. Report effect
sizes and 95% confidence intervals, not only null-hypothesis tests. Correct the
predeclared family of primary contrasts with Holm's method. Use two one-sided
equivalence tests when the claim is that two conditions are practically
indistinguishable.

## Staging and leakage control

1. Run `00` only to reconstruct the historical result.
2. Run the single-seed validation-only sweep in `02`.
3. Freeze lambda values, metrics, perturbation operators, and random-rationale
   generator before examining core test outcomes.
4. Run `01` and `03` with the same five seeds and paired data order.
5. Run `04` on frozen checkpoints. Attribution methods must not reuse the
   supervised attention tensor.
6. Proceed to `05` and `06` only after the core pipeline passes predeclared
   sanity checks.

The primary confirmatory result uses all five seeds. A failed or divergent run is
reported and rerun with the same seed only after documenting the failure mode.

## Manipulation and sanity checks

- Verify that sparsemax cells contain more exact zero mass than softmax cells.
- Verify that matched-random rationales equal the human rationale length for each
  eligible post and are deterministic from post ID and seed.
- Verify that all core cells use identical splits, truncation, optimizer, layer,
  and head set.
- Report results both with normal posts excluded from rationale supervision and,
  as sensitivity analysis, with the historical uniform target.
- Compare attention overlap with an independent gradient-based attribution; do not
  infer faithfulness from supervised attention alone.

## Decision criterion

A positive sparse/Fenchel--Young result requires all of the following:

1. a reproducible gain or favorable Pareto shift over the matched softmax loss;
2. a positive human-minus-random selectivity effect;
3. no material classification regression beyond the frozen equivalence margin;
4. corroboration by at least one independent attribution and one normalized
   perturbation test.

Exact zeros or higher overlap alone are manipulation/plausibility results, not a
faithfulness claim.
