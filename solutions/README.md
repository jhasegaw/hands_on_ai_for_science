# Solutions

Reference implementations for the homework modules that attendees write themselves.
The copies in `lectures/` are starter code: each function has its signature and docstring,
with the body replaced by `raise NotImplementedError("You need to write this part!")`.

**Files ending in `_unverified.py` are not the course author's work.** See Provenance below
before relying on one.

| File | Used by |
| --- | --- |
| `a12am_hw1_unverified.py` | `lectures/a12am_python.ipynb` (Aug 12 morning, `if` statements) |
| `a12am_hw2_unverified.py` | `lectures/a12am_containers.ipynb` (Aug 12 morning, loops and containers) |
| `a12pm_hw1_unverified.py` | `lectures/a12pm_numpy.ipynb` (Aug 12 afternoon, numpy and matplotlib) |
| `a12pm_hw2_unverified.py` | `lectures/a12pm_modules.ipynb` (Aug 12 afternoon, pandas/scipy/sklearn) |
| `a13am_hw2_unverified.py` | `lectures/a13am_generalization.ipynb` (Aug 13 morning, generalization) |
| `a13pm_hw1.py` | `lectures/a13pm_word2vec.ipynb` (Aug 13 afternoon, lexical embeddings) |
| `a13pm_hw2_unverified.py` | `lectures/a13pm_embeddings.ipynb` (Aug 13 afternoon, embeddings) |
| `a14am_hw1_unverified.py` | `lectures/a14am_transformer.ipynb` (Aug 14 morning, transformers) |
| `a14am_hw2_unverified.py` | `lectures/a14am_foundation.ipynb` (Aug 14 morning, foundation models) |
| `a14pm_hw1.py` | `lectures/a14pm_cnn.ipynb` (Aug 14 afternoon, convolutional neural nets) |
| `a14pm_hw1alt_unverified.py` | `lectures/a14pm_vision.ipynb` (Aug 14 afternoon, vision models) |
| `a14pm_hw2_unverified.py` | `lectures/a14pm_failures.ipynb` (Aug 14 afternoon, failure modes) |

## Provenance

`a13pm_hw1.py` and `a14pm_hw1.py` — no suffix — are the course author's own
implementations, preserved when those answers were removed from `lectures/`.

The ten `_unverified.py` files were written for the workshop and are **not** the authors'
originals. They aim to be readable rather than clever, and are commented as explanation.
Each reproduces the expected output its notebook actually publishes — the printed result
blocks, `X: (250, 4) float64`, `control ≈ 4.85 / treatment ≈ 6.12`, `embedding_rsa 0.319`,
`linear_probe 0.902`, the degree-3 coefficients and MSEs, and the padding and masking
examples in the docstrings — but none has been reviewed by whoever designed the
assignments. Treat them as a reference for a TA, not as an answer key.

Three are checked less thoroughly than the rest, and their banners say so:

- `a14am_hw1_unverified.py` — checked only against the examples in its docstrings, not by
  training the transformer end to end.
- `a14am_hw2_unverified.py` (`embed_sequences`) and `a14pm_hw1alt_unverified.py`
  (`embed_images`) — these follow the authors' own reference implementations in
  `lectures/make_protein_embeddings.py` and `lectures/make_image_embeddings.py`, the
  scripts that generated the shipped `.npy` files, so they should reproduce them exactly.
  They are the two functions not checked by running, because that needs the ESM-2 and
  DINOv2 models downloaded from HuggingFace.

Two published numbers do not reproduce on scikit-learn 1.3.1, and neither is caused by
these solutions:

- `a14am_foundation.ipynb` prints `composition: 0.642`; the notebook's **own** cell-21 code
  gives `0.562` here, and this solution's composition matrix is identical to it.
- `a14pm_failures.ipynb` prints the learning curve ending `0.898`; the underlying value is
  exactly `0.8975`, and Python's `round()` takes that to `0.897` while numpy's takes it
  to `0.898`.

Every one of these files repeats the unverified warning in a banner comment at the top,
because the `_unverified` suffix is lost as soon as the file is copied into place.

## Using one

To run a notebook end to end with the answers in place, copy the file over the starter
version, dropping the suffix:

```bash
cp solutions/a14pm_hw1.py            lectures/a14pm_hw1.py     # author's own
cp solutions/a12pm_hw2_unverified.py lectures/a12pm_hw2.py     # unverified
```

That overwrites the starter code, so do it on a scratch copy of the repo rather than the
one you teach from — or restore it afterwards with `git checkout lectures/`.

The self-contained Colab notebooks in `lectures/colab/` inline the starter versions, so
they carry the stubs too. If you change any file here, regenerate them with
`python3 tools/make_colab.py`.
