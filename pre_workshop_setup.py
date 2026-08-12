"""
Pre-workshop setup check for Hands-On AI for Science (Aug 12-14, 2026).

Run this BEFORE the workshop, on the wifi you have at home or office:

    python pre_workshop_setup.py

It checks that every package the workshop needs is installed, downloads and
caches the two pretrained models used on Friday (~120 MB total, a one-time
download), and verifies the workshop data files are present.  At the end it
prints a clear PASS or a list of exactly what to fix.

If it tells you to install something, install it, then run this script
again in a NEW terminal window (and during the workshop, remember the same
rule for notebooks: after installing anything, restart the kernel).
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LECTURES = os.path.join(HERE, "lectures")

REQUIRED = ["numpy", "scipy", "pandas", "matplotlib", "sklearn", "jupyter",
            "tqdm", "torch", "transformers", "torchvision"]
OPTIONAL = ["umap"]  # umap-learn: one plot per session degrades gracefully without it

MODELS = ["facebook/esm2_t6_8M_UR50D", "facebook/dinov2-small"]

DATA_FILES = ["protein_localization.csv", "protein_embeddings.npy",
              "word_vectors_50d.npz", "blood_cells.npz", "image_embeddings.npy",
              "CHECKLIST.md"]

INSTALL_HINT = {"sklearn": "scikit-learn", "umap": "umap-learn"}


def main():
    problems = []
    print(f"Python {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info < (3, 9):
        problems.append("Python 3.9+ is required -- install Anaconda "
                        "(see lectures/a12am_python.ipynb, Section 1).")

    print("\n--- Packages ---")
    missing = []
    for pkg in REQUIRED + OPTIONAL:
        ok = importlib.util.find_spec(pkg) is not None
        tag = "OK" if ok else ("MISSING (optional)" if pkg in OPTIONAL else "MISSING")
        print(f"  {pkg:14s} {tag}")
        if not ok and pkg not in OPTIONAL:
            missing.append(INSTALL_HINT.get(pkg, pkg))
    if missing:
        problems.append("Install missing packages:  pip install " + " ".join(missing))

    print("\n--- Data files (lectures/) ---")
    if not os.path.isdir(LECTURES):
        problems.append("The lectures/ folder was not found next to this script -- "
                        "run the script from inside the downloaded repository.")
    else:
        for f in DATA_FILES:
            ok = os.path.exists(os.path.join(LECTURES, f))
            print(f"  {f:28s} {'OK' if ok else 'MISSING'}")
            if not ok:
                problems.append(f"lectures/{f} is missing -- re-download the repository "
                                "(git pull, or a fresh Download ZIP).")

    print("\n--- Pretrained models ---")
    if any(m in ("torch", "transformers", "torchvision") for m in missing):
        problems.append("Model download skipped until torch/transformers/torchvision "
                        "are installed -- run this script again afterward.")
        print("  skipped (packages missing)")
    else:
        print("  NOTE: the first download prints progress bars and a red/orange")
        print("  'LOAD REPORT' mentioning UNEXPECTED and MISSING keys.")
        print("  That report is EXPECTED and is not an error.\n")
        from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
        for name in MODELS:
            try:
                if "esm" in name:
                    AutoTokenizer.from_pretrained(name)
                else:
                    AutoImageProcessor.from_pretrained(name)
                AutoModel.from_pretrained(name)
                print(f"  {name:32s} cached OK")
            except Exception as e:
                problems.append(f"Could not download {name} ({type(e).__name__}) -- "
                                "check your internet connection and re-run.")
                print(f"  {name:32s} FAILED")

    print("\n" + "=" * 60)
    if problems:
        print("NOT READY YET -- fix the following, then run this script again:\n")
        for p in problems:
            print("  *", p)
        sys.exit(1)
    print("ALL CHECKS PASSED -- this machine is ready for the workshop.")
    print("(Nothing else downloads during the sessions; venue wifi not required.)")


if __name__ == "__main__":
    main()
