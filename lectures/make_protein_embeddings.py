"""
Generate protein_embeddings.npy -- ESM-2 embeddings for every sequence in
protein_localization.csv, one row per CSV row, in CSV order.

This file serves two purposes:
  1. The Aug 13 afternoon embeddings session loads the .npy directly, so
     that Thursday needs no torch install and no model download.
  2. It is the wifi-fallback for the Aug 14 foundation-models session: if
     the live extraction fails in the room, the analysis half of that hour
     proceeds from this file.

The embedding recipe is deliberately identical to what the Aug 14 session
teaches: tokenize, forward pass, take the last hidden state, and average
over real (non-padding) positions using the attention mask.

Model: facebook/esm2_t6_8M_UR50D (ESM-2, 6 layers, hidden size 320, ~8M
parameters; Lin et al. 2023).  Runs on CPU in under a minute for the 450
sequences in the dataset.
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "facebook/esm2_t6_8M_UR50D"
BATCH_SIZE = 16


def embed_sequences(model, tokenizer, seqs, batch_size=BATCH_SIZE):
    """Masked mean-pooled embeddings, shape (len(seqs), hidden_size)."""
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(seqs), batch_size):
            batch = tokenizer(seqs[start:start + batch_size],
                              padding=True, return_tensors="pt")
            hidden = model(**batch).last_hidden_state          # (b, L, 320)
            mask = batch["attention_mask"].unsqueeze(-1)       # (b, L, 1)
            summed = (hidden * mask).sum(dim=1)                # exclude padding
            counts = mask.sum(dim=1)
            out.append((summed / counts).numpy())
    return np.concatenate(out).astype(np.float32)


def main():
    data = pd.read_csv("protein_localization.csv")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL)
    emb = embed_sequences(model, tokenizer, list(data["sequence"]))
    np.save("protein_embeddings.npy", emb)
    print(f"wrote protein_embeddings.npy: {emb.shape} ({emb.nbytes // 1024} KB)")


if __name__ == "__main__":
    main()
