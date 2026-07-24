# HateXplain data

`dataset.json` and `post_id_divisions.json` are preserved from the official
[HateXplain repository](https://github.com/punyajoy/HateXplain).

> **Content warning:** the dataset contains hateful, offensive, racist, sexist,
> homophobic, and otherwise distressing language.

Each record contains:

- `post_tokens`: the tokenized post;
- three class/target annotations;
- zero, two, or three token-level rationale masks.

The label vocabulary is `hatespeech`, `normal`, and `offensive`. The official split
file uses the keys `train`, `val`, and `test`. It excludes 919 records without a
two-of-three label majority.

See `ORIGINAL_DATA_README.md` for the documentation shipped with the source
snapshot. The dataset and source code remain subject to the included license and
the original project's citation requirements.
