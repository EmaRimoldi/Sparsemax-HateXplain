# 05 - Robustness and generalization

This follow-up tests whether the core effect survives beyond one encoder and one
distribution:

- repeat Pareto-relevant cells with a DeBERTa-family encoder;
- evaluate an external hate-speech dataset with token rationales when licensing
  and label mapping permit;
- report in-domain and out-of-domain classification and rationale metrics;
- inject controlled spurious lexical cues at increasing train/test correlations,
  including correlation reversal at test time.

The spurious-correlation experiment compares human, matched-random, and
no-supervision conditions. The primary outcome is degradation in macro-F1 and
independent-attribution mass assigned to the spurious cue.

The current model adapter is BERT-specific, so this directory is a preregistered
gate rather than a runnable matrix. A generic encoder adapter and a documented
dataset mapping must be merged before materializing these jobs.
