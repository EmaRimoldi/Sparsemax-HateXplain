# 03 - Controls

`no_supervision` estimates the BERT classification baseline. `sra_mse_reference`
implements a modern reference condition with a binary majority target, MSE on one
post-softmax attention head (head 7, layer 8, zero-indexed), and normal posts
excluded from rationale supervision.

The reference mirrors the main structural choices of recent supervised-rationale
attention work but remains an in-repository implementation, not a claim of exact
external-paper reproduction.
