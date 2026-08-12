#!/usr/bin/env python3
"""Generate self-contained Colab copies of the JW lecture notebooks.

Companion to make_colab.py (which covers the original lectures) and reuses its
approach: homework .py modules are inlined as editable code cells, prose that
says "open the file" is rewritten, and nothing under lectures/*.ipynb is
modified.  Differences from make_colab.py, needed by these notebooks:

  * data files are DOWNLOADED from the repo's raw URLs at runtime instead of
    base64-embedded (some are several MB);
  * a notebook may have no modules to inline (the PyTorch primer);
  * the setup-check cells reference the homework .py files, which exist as
    cells rather than files in the Colab copies -- those checks are rewritten;
  * umap-learn is pip-installed where used (Colab does not preinstall it).
"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEC = os.path.join(ROOT, "lectures")
OUT = os.path.join(LEC, "colab")
RAW = "https://raw.githubusercontent.com/jhasegaw/hands_on_ai_for_science/main/lectures"

NOTEBOOKS = {
    "a13am_generalization.ipynb": ["a13am_hw2"],
    "a13pm_embeddings.ipynb":     ["a13pm_hw2"],
    "a14am_torch_primer.ipynb":   [],
    "a14am_foundation.ipynb":     ["a14am_hw2"],
    "a14pm_vision.ipynb":         ["a14pm_hw1alt"],
    "a14pm_failures.ipynb":       ["a14pm_hw2"],
}

EDITABLE = {"a13am_hw2", "a13pm_hw2", "a14am_hw2", "a14pm_hw1alt", "a14pm_hw2"}

DOWNLOAD_FILES = {
    "a13pm_embeddings.ipynb": ["protein_localization.csv", "protein_embeddings.npy",
                               "word_vectors_50d.npz"],
    "a14am_foundation.ipynb": ["protein_localization.csv", "protein_embeddings.npy"],
    "a14pm_vision.ipynb":     ["blood_cells.npz", "image_embeddings.npy"],
    "a14pm_failures.ipynb":   ["protein_localization.csv", "protein_embeddings.npy",
                               "blood_cells.npz", "image_embeddings.npy", "CHECKLIST.md"],
}

# per-notebook code rewrites, applied to every code cell (regex, replacement)
UMAP_FIX = (r"^try:\n    import umap",
            "%pip install -q umap-learn  # Colab does not preinstall umap-learn\n"
            "try:\n    import umap")
CODE_FIX = {
    "a13pm_embeddings.ipynb": [UMAP_FIX],
    "a14pm_vision.ipynb":     [UMAP_FIX],
}

# strip homework .py names out of the setup-check file lists (they are cells here)
HW_IN_FILES = re.compile(r"(?:,\s*)?'a1\d[ap]m_hw\w*\.py'")

PROSE = [
    (r"[Ff]our functions in `(\w+)\.py`", "Four functions, defined in the code cell below"),
    (r"[Tt]hree functions in `(\w+)\.py`", "Three functions, defined in the code cell below"),
    (r"functions in `(\w+)\.py`", "functions in the definitions cell below"),
    (r"As (?:before|always): open the file, (?:replace|delete)(?: each| the)? `raise RuntimeError`"
     r"[^.]*\.",
     "Edit the function definitions in the code cell below: replace each "
     "`raise RuntimeError` line with your own code, then re-run that cell before "
     "running the checks."),
    (r"open the file `?(\w+)\.py`?", "edit the definitions cell below"),
]


def src(text):
    return text.splitlines(keepends=True)


def defined_funcs(mod):
    body = open(os.path.join(LEC, mod + ".py")).read()
    return re.findall(r"^def\s+(\w+)", body, re.M)


def def_cells(mod, minor):
    body = open(os.path.join(LEC, mod + ".py")).read().rstrip("\n")
    blurb = (f"### Your code goes here: `{mod}`\n\n"
             "The functions below are the ones you need to write. **Edit them right here in "
             "this cell**, then re-run this cell (Shift+Enter) to update your definitions. "
             "Re-run the cells further down to test them.")
    md = {"cell_type": "markdown", "metadata": {}, "source": src(blurb)}
    code = {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src(body)}
    if minor >= 5:
        md["id"] = "inline-md-" + mod.replace("_", "-")
        code["id"] = "inline-code-" + mod.replace("_", "-")
    return [md, code]


def download_cells(nb_name, minor):
    names = DOWNLOAD_FILES.get(nb_name, [])
    if not names:
        return []
    lines = "\n".join('    "%s",' % n for n in names)
    code = ("# The data files this notebook uses, fetched from the workshop repository.\n"
            "# Run this cell once; you do not need to edit it.\n"
            "import os, urllib.request\n\n"
            "_FILES = [\n" + lines + "\n]\n\n"
            "for _name in _FILES:\n"
            "    if not os.path.exists(_name):\n"
            "        urllib.request.urlretrieve('%s/' + _name, _name)\n"
            "print('Ready:', ', '.join(_FILES))" % RAW)
    md = {"cell_type": "markdown", "metadata": {},
          "source": src("### Data files\n\nRun this cell to download the data files this "
                        "notebook uses. You do not need to edit it.")}
    cd = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
          "source": src(code)}
    if minor >= 5:
        md["id"], cd["id"] = "dl-data-md", "dl-data-code"
    return [md, cd]


def strip_module_refs(text, mods, funcs_by_mod):
    out = []
    for line in text.split("\n"):
        s = line.strip()
        drop = False
        for m in mods:
            if re.fullmatch(r"import\s+[\w\s,]*\b%s\b[\w\s,]*" % m, s):
                drop = True
            if re.fullmatch(r"from\s+%s\s+import\s+.*" % m, s):
                drop = True
            if re.fullmatch(r"importlib\.reload\(\s*%s\s*\)" % m, s):
                drop = True
            if re.fullmatch(r"help\(\s*%s\s*\)" % m, s):
                indent = line[:len(line) - len(line.lstrip())]
                out.extend(indent + "help(%s)" % f for f in funcs_by_mod[m])
                drop = True
        if drop:
            continue
        for m in mods:
            line = re.sub(r"\b%s\." % m, "", line)
        out.append(line)
    text = "\n".join(out)
    if "importlib." not in text:
        text = "\n".join(l for l in text.split("\n")
                         if not re.fullmatch(r"\s*import\s+importlib\s*", l))
    return re.sub(r"\n{3,}", "\n\n", text).strip("\n")


def fix_code(nb_name, text):
    text = HW_IN_FILES.sub("", text)
    for pat, rep in CODE_FIX.get(nb_name, []):
        text = re.sub(pat, rep, text, flags=re.M)
    return text


def fix_prose(text):
    for pat, rep in PROSE:
        text = re.sub(pat, rep, text)
    return text


def main():
    os.makedirs(OUT, exist_ok=True)
    for nb_name, mods in NOTEBOOKS.items():
        nb = json.load(open(os.path.join(LEC, nb_name)))
        minor = nb.get("nbformat_minor", 0)
        funcs = {m: defined_funcs(m) for m in mods}

        first = None
        if mods:
            for i, c in enumerate(nb["cells"]):
                if c["cell_type"] == "code" and any(m in "".join(c["source"]) for m in mods):
                    first = i
                    break

        first_data = None
        if DOWNLOAD_FILES.get(nb_name):
            for i, c in enumerate(nb["cells"]):
                if c["cell_type"] == "code" and any(d in "".join(c["source"])
                                                    for d in DOWNLOAD_FILES[nb_name]):
                    first_data = i
                    break

        cells, leftover = [], []
        for i, c in enumerate(nb["cells"]):
            if i == first_data:
                cells.extend(download_cells(nb_name, minor))
            if i == first:
                for m in mods:
                    cells.extend(def_cells(m, minor))
            c = dict(c)
            text = "".join(c["source"])
            if c["cell_type"] == "markdown" and "Open In Colab" in text:
                continue
            if c["cell_type"] == "code":
                new = strip_module_refs(fix_code(nb_name, text), mods, funcs)
                if not new.strip():
                    continue
                c["source"] = src(new)
                c["outputs"], c["execution_count"] = [], None
            else:
                new = fix_prose(text)
                new = new.replace('src="figs/', 'src="%s/figs/' % RAW)
                c["source"] = src(new)
                if re.search(r"`\w+_hw\w*\.py`", new):
                    leftover.append(i)
            cells.append(c)

        nb["cells"] = cells

        hdr = ("### Self-contained Colab notebook\n\n"
               "Everything this notebook needs is either in it or downloaded by its own "
               "cells. Just run the cells in order. No GPU is required.\n")
        hcell = {"cell_type": "markdown", "metadata": {}, "source": src(hdr)}
        if minor >= 5:
            hcell["id"] = "colab-header"
        nb["cells"].insert(1, hcell)

        title = nb_name[:-6]
        dest = os.path.join(OUT, title + "_colab.ipynb")
        with open(dest, "w") as fh:
            json.dump(nb, fh, indent=1, ensure_ascii=False)
            fh.write("\n")
        print("  %-28s -> colab/%s_colab.ipynb (%d cells)" % (nb_name, title, len(nb["cells"])))
        if leftover:
            print("       prose still mentioning a hw .py file, needs eyes: cells %s"
                  % ", ".join(map(str, leftover)))


if __name__ == "__main__":
    main()
