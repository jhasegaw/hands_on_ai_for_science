# Solutions

Reference implementations for the two homework modules that attendees write themselves.
The copies in `lectures/` are starter code: each function has its signature and docstring,
with the body replaced by `raise NotImplementedError` or
`raise RuntimeError("You need to write this part!")`.

| File | Used by |
| --- | --- |
| `a13pm_hw1.py` | `lectures/a13pm_word2vec.ipynb` (Aug 13 afternoon, lexical embeddings) |
| `a14pm_hw1.py` | `lectures/a14pm_cnn.ipynb` (Aug 14 afternoon, convolutional neural nets) |

The other homework modules — `a12am_hw1`, `a12am_hw2`, `a12pm_hw1`, `a12pm_hw2`,
`a14am_hw1` — have always shipped as starter code, so there is nothing to keep here.

To run a notebook end to end with the answers in place, copy the file over the starter
version:

```bash
cp solutions/a14pm_hw1.py lectures/a14pm_hw1.py
```

That overwrites the starter code, so do it on a scratch copy of the repo rather than the
one you teach from — or restore it afterwards with `git checkout lectures/a14pm_hw1.py`.

The self-contained Colab notebooks in `lectures/colab/` inline the starter versions, so
they carry the stubs too. If you change either file here, regenerate them with
`python3 tools/make_colab.py`.
