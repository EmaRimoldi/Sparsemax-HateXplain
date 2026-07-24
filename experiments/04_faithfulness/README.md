# 04 - Faithfulness evaluation

This stage runs only on frozen checkpoints from experiments `01` and `03`.

Required evaluation views:

1. token average precision/F1 for the supervised attention map;
2. the same plausibility metrics for Integrated Gradients and Input x Gradient,
   computed from model inputs rather than the supervised tensor;
3. comprehensiveness and sufficiency under deletion, masking, and
   language-model infilling;
4. normalized AOPC for cross-model comparisons;
5. length-matched random token sets for every perturbation operator;
6. cascading parameter-randomization tests for attribution sensitivity.

Report perturbation curves rather than a single removal point. The independent
attributions are primary evidence: a supervised attention map agreeing with its
training target is expected and does not by itself demonstrate faithfulness.

Implementation is intentionally gated until the training matrices and checkpoint
format are frozen, preventing metric code from silently depending on one model
condition.
