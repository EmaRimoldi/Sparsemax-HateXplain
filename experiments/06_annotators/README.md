# 06 - Annotator sensitivity

The mean, majority, and union masks operationalize different levels of consensus.
After the aggregate sweep, the three individual masks are evaluated on records
where that annotator supplied a rationale. This prevents annotator disagreement
from being collapsed into a single unexplained soft target.

The checked-in matrix covers aggregate representations. Per-annotator cells are
gated on an eligibility report because missing rationale masks must not be
silently treated as genuine negative annotations in the primary analysis.
