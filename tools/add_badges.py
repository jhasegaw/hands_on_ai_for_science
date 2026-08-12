#!/usr/bin/env python3
"""Add an 'Open in Colab' badge cell to each original lecture notebook.

The badge points at that lecture's self-contained copy in lectures/colab/.
Only a markdown cell is added, so local execution is unaffected.
"""
import json, os, re

LEC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lectures")
COLAB = "https://colab.research.google.com/github/jhasegaw/hands_on_ai_for_science/blob/main/lectures/colab"
BADGE = "https://colab.research.google.com/assets/colab-badge.svg"
MARKER = "Open In Colab"

NOTEBOOKS = [
    "a12am_python.ipynb", "a12am_containers.ipynb", "a12pm_numpy.ipynb",
    "a12pm_modules.ipynb", "a13am_neuralnets.ipynb", "a13pm_word2vec.ipynb",
    "a14am_transformer.ipynb", "a14pm_cnn.ipynb",
]


def main():
    for name in NOTEBOOKS:
        path = os.path.join(LEC, name)
        raw = open(path, "rb").read()
        # Notebooks in this repo are saved with either literal non-ASCII or \uXXXX
        # escapes, never both. Match whichever this file uses so the diff stays minimal.
        ensure_ascii = not any(b > 127 for b in raw)
        trailing_nl = raw.endswith(b"\n")   # not every notebook here has one
        nb = json.loads(raw.decode("utf-8"))

        # idempotent: never add a second badge
        if any(MARKER in "".join(c.get("source", [])) for c in nb["cells"][:4]):
            print("  skip (already has badge) %s" % name)
            continue

        target = "%s/%s_colab.ipynb" % (COLAB, name[:-6])
        text = ("[![%s](%s)](%s)\n\n"
                "**No Python installed?** Click the badge above to open a self-contained "
                "version of this notebook in Google Colab.  It runs in your browser, with "
                "nothing to download and nothing to install." % (MARKER, BADGE, target))

        cell = {"cell_type": "markdown", "metadata": {},
                "source": text.splitlines(keepends=True)}
        if nb.get("nbformat_minor", 0) >= 5:
            cell["id"] = "open-in-colab"

        nb["cells"].insert(1, cell)
        with open(path, "w") as fh:
            json.dump(nb, fh, indent=1, ensure_ascii=ensure_ascii)
            if trailing_nl:
                fh.write("\n")
        print("  %-28s badge -> colab/%s_colab.ipynb" % (name, name[:-6]))


if __name__ == "__main__":
    main()
