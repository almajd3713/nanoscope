---
license: other
license_name: mixed-see-card
pretty_name: nanoscope tokens
size_categories:
  - 1B<n<10B
---

# nanoscope tokens

Pre-tokenized text for [nanoscope](https://github.com/almajd3713/nanoscope), so training
doesn't start with an hour of tokenizing. `nanoscope` downloads these on its own; you
shouldn't need to touch them.

Each folder is a raw `uint16` array of token ids, with documents separated by the end-of-text
token, plus a `meta.json` with the document, token and byte counts.

| Folder | Source | Tokenizer | Train | Held-out |
|---|---|---|---|---|
| `roneneldan_TinyStories/bpe-4096-20000` | TinyStories | 4096-token BPE trained on the first 20,000 stories (`tokenizer.json`) | 100k stories (22.5M tokens) or 1M stories (230M tokens) | 200 or 1,000 stories |
| `HuggingFaceFW_fineweb-edu_sample-10BT_holdout-10000/gpt2` | FineWeb-Edu `sample-10BT` | GPT-2 (tiktoken) | 2M documents (2.07B tokens) | 1,000 documents |

For FineWeb-Edu, the first 10,000 documents of the stream are held out and never used for
training. The held-out documents here are the first 1,000 of those.

## Licenses

These are derived from other people's data and keep their licenses:

- TinyStories: CDLA-Sharing-1.0
- FineWeb-Edu: ODC-By 1.0, plus the terms of Common Crawl

Credit to Eldan & Li (TinyStories) and Penedo et al. (FineWeb-Edu).
