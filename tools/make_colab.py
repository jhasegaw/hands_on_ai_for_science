#!/usr/bin/env python3
"""Generate self-contained Colab copies of the lecture notebooks.

Local notebooks are read-only inputs; nothing under lectures/*.ipynb is modified.
Each output inlines the local .py modules as ordinary code cells and rewrites
every call site from `module.func(...)` to `func(...)`.
"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEC = os.path.join(ROOT, "lectures")
OUT = os.path.join(LEC, "colab")
RAW = "https://raw.githubusercontent.com/jhasegaw/hands_on_ai_for_science/main/lectures"

# notebook -> local modules to inline, in the order students meet them
NOTEBOOKS = {
    "a12am_python.ipynb":      ["a12am_hw1"],
    "a12am_containers.ipynb":  ["a12am_hw2"],
    "a12pm_numpy.ipynb":       ["a12pm_hw1"],
    "a12pm_modules.ipynb":     ["a12pm_hw2"],
    "a13am_neuralnets.ipynb":  ["a13pm_hw1"],
    "a13pm_word2vec.ipynb":    ["a13pm_hw1"],
    "a14am_transformer.ipynb": ["a14am_utils", "a14am_hw1"],
    "a14pm_cnn.ipynb":         ["a14pm_getdata", "a14pm_hw1"],
}

# modules students are meant to edit (vs. helper modules they just use)
EDITABLE = {"a12am_hw1", "a12am_hw2", "a12pm_hw1", "a12pm_hw2", "a13pm_hw1", "a14am_hw1", "a14pm_hw1"}

# Data files a notebook reads from disk. Embedded so read_csv(path) calls work unchanged.
DATA_FILES = {
    "a12pm_modules.ipynb": ["a12pm_measurements.csv", "a12pm_mouse_metadata.csv"],
}

GPU = {"a14am_transformer.ipynb"}

report = {}

# a12am_python teaches `import` / `importlib.reload` as the lesson itself, using
# a12am_hw1.py as its worked example. Inlining removes that file, so the example is
# retargeted to a stdlib module and the reload lesson is restated for a notebook.
# Keyed by ORIGINAL cell index; None deletes the cell.
OVERRIDES = {
    "a12am_python.ipynb": {
        47: ("markdown",
             "A **module** in python is an external file that contains some code you want to "
             "run.  **Python is useful primarily because so many people have written so many "
             "amazing modules.** You can run their code by just downloading it, and importing "
             "it using the `import` command.\n\n"
             "* The `import` command runs the code in a file\n"
             "* The file can be one in the same directory (a file you wrote) **or** it can be "
             "a module that's distributed with python. Some of the modules you can import are "
             "listed [here](https://docs.python.org/3/library/index.html).\n\n"
             "For example, `math` is a module that comes with python.  We can import it using "
             "`import math`, then, **after we import it**, we can use the `help` command to "
             "find out what's inside it."),
        48: ("code", "import math\nhelp(math.sqrt)"),
        49: ("markdown",
             "**WARNING:** The same idea applies to code you write in this notebook.\n\n"
             "1. Run a cell that defines a function\n"
             "2. Edit the function to change some stuff\n"
             "3. Use the function again\n\n"
             "Python is still using the **old** version until you re-run the cell that defines "
             "it!  Whenever you edit a function, re-run its cell (Shift+Enter) before testing "
             "it again.\n\n"
             "(In the downloadable version of this course the functions live in a separate "
             "`.py` file, and you reload them with `importlib.reload`.  In this self-contained "
             "notebook, re-running the cell does the same job.)"),
        50: None,
    },
}


def src(text):
    return text.splitlines(keepends=True)


def defined_funcs(mod):
    body = open(os.path.join(LEC, mod + ".py")).read()
    return re.findall(r"^def\s+(\w+)", body, re.M)


def def_cells(mod, nbformat_minor):
    """A markdown header + a code cell holding the module's full source."""
    body = open(os.path.join(LEC, mod + ".py")).read().rstrip("\n")
    if mod in EDITABLE:
        blurb = (f"### Your code goes here: `{mod}`\n\n"
                 "The functions below are the ones you need to write. **Edit them right here in "
                 "this cell**, then re-run this cell (Shift+Enter) to update your definitions. "
                 "Re-run the cells further down to test them.")
    else:
        blurb = (f"### Helper code: `{mod}`\n\n"
                 "These are provided for you. Just run this cell — you do not need to edit it.")
    md = {"cell_type": "markdown", "metadata": {}, "source": src(blurb)}
    code = {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src(body)}
    if nbformat_minor >= 5:
        md["id"] = "inline-md-" + mod.replace("_", "-")
        code["id"] = "inline-code-" + mod.replace("_", "-")
    return [md, code]


def data_cells(nb_name, nbformat_minor):
    """A cell that recreates the notebook's data files from embedded bytes.

    base64 rather than a pasted CSV literal so quoting/encoding in the data can
    never break the cell. Downstream `pd.read_csv('name.csv')` calls are untouched.
    """
    import base64
    names = DATA_FILES.get(nb_name, [])
    if not names:
        return []
    entries = []
    for n in names:
        b64 = base64.b64encode(open(os.path.join(LEC, n), "rb").read()).decode()
        entries.append("    %r:\n        %r," % (n, b64))
    code = ("# The data files for this notebook, embedded so nothing needs downloading.\n"
            "# Run this cell once; it recreates the .csv files next to the notebook.\n"
            "import base64, pathlib\n\n"
            "_DATA = {\n" + "\n".join(entries) + "\n}\n\n"
            "for _name, _b64 in _DATA.items():\n"
            "    pathlib.Path(_name).write_bytes(base64.b64decode(_b64))\n\n"
            "print('Created:', ', '.join(_DATA))")
    md = {"cell_type": "markdown", "metadata": {},
          "source": src("### Data files\n\nRun this cell to create the data files this notebook reads. "
                        "You do not need to edit it.")}
    cd = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src(code)}
    if nbformat_minor >= 5:
        md["id"], cd["id"] = "inline-data-md", "inline-data-code"
    return [md, cd]


def strip_module_refs(text, mods, funcs_by_mod):
    """Remove import/reload plumbing and de-qualify calls."""
    out = []
    for line in text.split("\n"):
        s = line.strip()
        drop = False
        for m in mods:
            # import a12am_hw1 / import a12am_hw1, importlib / import importlib, a12am_hw1
            if re.fullmatch(r"import\s+[\w\s,]*\b%s\b[\w\s,]*" % m, s):
                drop = True
            if re.fullmatch(r"from\s+%s\s+import\s+.*" % m, s):
                drop = True
            if re.fullmatch(r"importlib\.reload\(\s*%s\s*\)" % m, s):
                drop = True
            # help(module) -> help() on each inlined function
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
    # importlib is only plumbing for these modules; drop it if now unused
    if "importlib." not in text:
        text = "\n".join(l for l in text.split("\n")
                         if not re.fullmatch(r"\s*import\s+importlib\s*", l))
    return re.sub(r"\n{3,}", "\n\n", text).strip("\n")


PROSE = [
    (r"open up the file `(\w+)\.py` from the file listing in one of your browser tabs",
     "edit the function definitions in the code cell above"),
    (r"go to your browser window that includes a file listing, and click on `(\w+)\.py`\.\s*"
     r"That should open a text editor, where you can edit the file\.",
     "scroll up to the code cell holding the function definitions and edit it there."),
    (r"[Ii]n (?:the file )?[`]{1,3}(\w+)\.py[`]{1,3},? (solve|the)",
     lambda m: "In the definitions cell above, %s" % m.group(2)),
    (r"[`]{1,3}submitted\.py[`]{1,3}", "the definitions cell above"),
    (r"Solutions go in `(\w+)\.py`", "Solutions go in the definitions cell above"),
    (r"### Functions in `(\w+)\.py`", "### Functions to write"),
    (r"## Functions in `(\w+)\.py`", "## Functions to write"),
    (r"the attached file `(\w+)\.py`", "the definitions cell above"),
    (r"the provided synthetic generator in `(\w+)\.py`", "the provided synthetic generator above"),
]


def fix_prose(text):
    for pat, rep in PROSE:
        text = re.sub(pat, rep, text)
    return text


def inline_attachments(cell, text):
    """Colab cannot render `attachment:` images. Rewrite them as inline data: URIs.

    The image bytes already live in the cell's `attachments`; moving them into the
    markdown makes them render, and lets us drop the now-duplicate attachments blob.
    """
    att = cell.get("attachments") or {}
    if not att:
        return text, 0
    n = 0
    for name, payload in att.items():
        mime, data = next(iter(payload.items()))
        if isinstance(data, list):
            data = "".join(data)
        data = data.replace("\n", "")
        new, k = re.subn(r"attachment:" + re.escape(name),
                         "data:%s;base64,%s" % (mime, data), text)
        text, n = new, n + k
    cell.pop("attachments", None)
    return text, n


# Install commands that assume a local conda/pip env. Colab already has these.
CODE_FIX = [
    (r"^files = \[('a12pm_measurements\.csv', 'a12pm_mouse_metadata\.csv'), 'a12pm_hw2\.py'\]$",
     r"files = [\1]  # a12pm_hw2.py is defined in a cell below, not a file"),
    (r"^!conda install .*$",
     "# On Colab, PyTorch is already installed - nothing to do here."),
    (r"^!pip install -q numpy matplotlib$",
     "# On Colab, numpy and matplotlib are already installed - nothing to do here."),
]


def fix_code(text):
    for pat, rep in CODE_FIX:
        text = re.sub(pat, rep, text, flags=re.M)
    return text


def main():
    os.makedirs(OUT, exist_ok=True)
    for nb_name, mods in NOTEBOOKS.items():
        nb = json.load(open(os.path.join(LEC, nb_name)))
        minor = nb.get("nbformat_minor", 0)
        funcs = {m: defined_funcs(m) for m in mods}

        # Apply per-notebook overrides first, so definition placement sees the final layout.
        ov = OVERRIDES.get(nb_name, {})
        if ov:
            rebuilt = []
            for i, c in enumerate(nb["cells"]):
                if i in ov:
                    if ov[i] is None:
                        continue
                    kind, text = ov[i]
                    c = {"cell_type": kind, "metadata": {}, "source": src(text)}
                    if kind == "code":
                        c["outputs"], c["execution_count"] = [], None
                    if minor >= 5:
                        c["id"] = "override-%d" % i
                rebuilt.append(c)
            nb["cells"] = rebuilt

        # first cell that mentions any of the modules -> definitions go just before it
        first = None
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code" and any(m in "".join(c["source"]) for m in mods):
                first = i
                break
        if first is None:
            print("  !! %s: no module use found" % nb_name)
            continue

        # first cell that mentions a data file -> the data cell goes just before it,
        # so the notebook's own environment check sees the files present
        first_data = None
        if DATA_FILES.get(nb_name):
            for i, c in enumerate(nb["cells"]):
                if c["cell_type"] == "code" and any(d in "".join(c["source"])
                                                    for d in DATA_FILES[nb_name]):
                    first_data = i
                    break

        cells, leftover, inlined = [], [], [0]
        for i, c in enumerate(nb["cells"]):
            if i == first_data:
                cells.extend(data_cells(nb_name, minor))
            if i == first:
                for m in mods:
                    cells.extend(def_cells(m, minor))
            c = dict(c)
            text = "".join(c["source"])
            # the originals carry an "Open in Colab" badge; inside the Colab copy
            # it would just link to this same notebook, so drop it
            if c["cell_type"] == "markdown" and "Open In Colab" in text:
                continue
            if c["cell_type"] == "code":
                new = fix_code(strip_module_refs(text, mods, funcs))
                if not new.strip():
                    continue          # cell was pure import plumbing
                c["source"] = src(new)
                c["outputs"], c["execution_count"] = [], None
            else:
                new = fix_prose(text)
                new = new.replace('src="figs/', 'src="%s/figs/' % RAW)
                new, n_att = inline_attachments(c, new)
                inlined[0] += n_att
                c["source"] = src(new)
                if re.search(r"`?\w+\.py`?", new):
                    leftover.append(i)
            cells.append(c)

        nb["cells"] = cells

        # Colab header
        title = nb_name[:-6]
        hdr = ("### Self-contained Colab notebook\n\n"
               "Everything you need is in this notebook — there is nothing to install and nothing "
               "to download. Just run the cells in order.\n")
        if nb_name in GPU:
            hdr += ("\nThis notebook needs a GPU: go to **Runtime → Change runtime type → T4 GPU** "
                    "before running anything. The free Colab tier covers it.\n")
        hcell = {"cell_type": "markdown", "metadata": {}, "source": src(hdr)}
        if minor >= 5:
            hcell["id"] = "colab-header"
        nb["cells"].insert(1, hcell)

        dest = os.path.join(OUT, title + "_colab.ipynb")
        with open(dest, "w") as fh:
            json.dump(nb, fh, indent=1, ensure_ascii=False)
            fh.write("\n")
        report[nb_name] = (len(nb["cells"]), leftover)
        print("  %-28s -> colab/%s_colab.ipynb (%d cells, %d image(s) inlined)"
              % (nb_name, title, len(nb["cells"]), inlined[0]))
        if leftover:
            print("       prose still mentioning a .py file, needs your eyes: cells %s"
                  % ", ".join(map(str, leftover)))


if __name__ == "__main__":
    main()
