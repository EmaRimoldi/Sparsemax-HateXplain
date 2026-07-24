# 00 - Archived reproduction

This experiment reconstructs the two historical code paths:

- dense targets, cross-entropy on post-softmax attention values;
- sparse targets, sparsemax Fenchel--Young loss on raw attention scores.

Because both the target/loss pair and the supervised value change together, this
is not a causal comparison. Normal posts retain the historical uniform target.
Results belong in an archival table, separate from the controlled factorial.
