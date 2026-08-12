"""
Generate protein_localization.csv -- the protein dataset shared by the
Aug 13 afternoon embeddings session and the Aug 14 foundation-model and
failure-modes sessions.

Attendees do not need to run this file: the CSV it produces is already in
the repository, and it is the frozen artifact (UniProt content drifts over
time, so re-running this script may produce slightly different rows).
The script is kept so the dataset's provenance and construction are clear.

Data source: UniProt/Swiss-Prot (https://www.uniprot.org), REST API,
retrieved 2026-08-09.  UniProt data is distributed under the Creative
Commons Attribution 4.0 license (CC BY 4.0).

Construction:
  1. For each of three subcellular-localization classes (cytoplasm,
     secreted, membrane), query reviewed human proteins of length 80-300
     whose subcellular-location annotation mentions ONLY that class among
     the three (single-location filter), and which have a gene name.
  2. Randomly sample 150 per class (seed 0).
  3. Swap in a small set of widely recognizable proteins (insulin, IGF-1,
     IGF-2, glucagon, IL-6, lysozyme, myoglobin, calmodulin, defensin),
     replacing random rows of the same class, so nearest-neighbor demos
     return names people recognize.
  4. Swap in members of several protein families (interferons-alpha,
     defensins, CXCL/CCL chemokines, GST-A/M, gamma-crystallins, S100,
     IFITM, MS4A, tetraspanins, glycophorins, claudins) subject to the
     same filters.  Families give the dataset near-duplicate sequences,
     which the Aug 14 failure-modes session needs to demonstrate
     identity-based train/test leakage.
  5. Map every accession to its UniRef50 cluster (UniProt idmapping API).
     UniRef50 groups sequences at >=50% identity; the `uniref50` column is
     the grouping variable for honest (grouped) train/test splits.

Result: 450 proteins, 150 per class, 428 distinct UniRef50 clusters, of
which 12 have more than one member (34 proteins; largest family: 10
interferon-alpha paralogs).

Columns: accession, gene, protein_name, location (label), length,
uniref50 (group id), sequence.
"""

import json
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

API = "https://rest.uniprot.org"
FIELDS = "accession,gene_primary,protein_name,length,cc_subcellular_location,sequence"
CLASSES = {"cytoplasm": "SL-0086", "secreted": "SL-0243", "membrane": "SL-0162"}
KEYWORDS = {"cytoplasm": "Cytoplasm", "secreted": "Secreted", "membrane": "Membrane"}

CELEBRITIES = ["P01308", "P05019", "P01344", "P01275", "P05231", "P61626",
               "P02144", "P0DP23", "P59665"]
FAMILY_QUERY = ("gene:IFNA* OR gene:DEFA* OR gene:CXCL* OR gene:CCL* OR "
                "gene:GSTA* OR gene:GSTM* OR gene:CRYG* OR gene:S100A* OR "
                "gene:IFITM* OR gene:MS4A* OR gene:TSPAN* OR gene:CLDN* OR "
                "gene:GYPA OR gene:GYPB OR gene:GYPC")


def search(query, size=500):
    url = (f"{API}/uniprotkb/search?query={urllib.parse.quote(query)}"
           f"&format=tsv&fields={FIELDS}&size={size}")
    df = pd.read_csv(url, sep="\t")
    df.columns = ["accession", "gene", "protein_name", "length",
                  "location_full", "sequence"]
    df["gene"] = df["gene"].astype(str).str.split(";").str[0]
    return df


def single_location(df):
    """Keep rows mentioning exactly one of the three class keywords."""
    def classify(loc):
        hits = [k for k, v in KEYWORDS.items() if v in str(loc)]
        return hits[0] if len(hits) == 1 else None
    df = df.copy()
    df["location"] = df["location_full"].apply(classify)
    return df[df["location"].notna() & (df["gene"] != "nan")]


def uniref50_map(accessions):
    req = urllib.request.Request(
        f"{API}/idmapping/run",
        data=urllib.parse.urlencode({"from": "UniProtKB_AC-ID",
                                     "to": "UniRef50",
                                     "ids": ",".join(accessions)}).encode())
    job = json.load(urllib.request.urlopen(req))["jobId"]
    while True:
        time.sleep(3)
        st = json.load(urllib.request.urlopen(f"{API}/idmapping/status/{job}"))
        if st.get("jobStatus") in (None, "FINISHED") or "results" in st:
            break
    res = json.load(urllib.request.urlopen(f"{API}/idmapping/stream/{job}"))
    return {r["from"]: (r["to"]["id"] if isinstance(r["to"], dict) else r["to"])
            for r in res["results"]}


def main():
    base = "reviewed:true AND organism_id:9606 AND length:[80 TO 300]"

    # 1-2: per-class pools, sampled to 150
    rng = np.random.default_rng(0)
    parts = []
    for name, sl in CLASSES.items():
        pool = single_location(search(f"{base} AND cc_scl_term:{sl}"))
        pool = pool[pool["location"] == name]
        idx = rng.choice(len(pool), size=min(150, len(pool)), replace=False)
        parts.append(pool.iloc[sorted(idx)])
    data = pd.concat(parts, ignore_index=True)

    # 3-4: swap in celebrities, then family members
    def swap_in(rows, seed, protected):
        nonlocal data
        r = np.random.default_rng(seed)
        for _, row in rows.iterrows():
            if row["accession"] in set(data["accession"]):
                continue
            pool = data[(data["location"] == row["location"])
                        & ~data["accession"].isin(protected)].index
            if not len(pool):
                continue
            data = data.drop(r.choice(pool))
            data = pd.concat([data, row.to_frame().T], ignore_index=True)

    celebs = single_location(search(" OR ".join(f"accession:{a}" for a in CELEBRITIES), size=50))
    swap_in(celebs, seed=7, protected=set())
    fams = single_location(search(f"{base} AND ({FAMILY_QUERY})"))
    swap_in(fams, seed=11, protected=set(celebs["accession"]) | set(fams["accession"]))

    # 5: UniRef50 groups
    data["accession"] = data["accession"].astype(str).str.strip()
    data = data.drop_duplicates("accession")
    data["uniref50"] = data["accession"].map(uniref50_map(list(data["accession"])))

    data = (data[["accession", "gene", "protein_name", "location", "length",
                  "uniref50", "sequence"]]
            .sort_values(["location", "gene"]).reset_index(drop=True))
    data["length"] = data["length"].astype(int)
    data.to_csv("protein_localization.csv", index=False)
    print(f"wrote protein_localization.csv: {len(data)} rows,",
          dict(data['location'].value_counts()))


if __name__ == "__main__":
    main()
