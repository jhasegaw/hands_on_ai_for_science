"""
Generate image_embeddings.npy -- DINOv2 embeddings for every image in
blood_cells.npz, one row per image, in file order.

Serves the same two purposes as make_protein_embeddings.py does for the
morning session: the afternoon vision session verifies live extraction
against this file, and it is the wifi-fallback if the model can't be
loaded in the room.

Recipe (deliberately identical to what the session teaches): preprocess
with the model's AutoImageProcessor, forward pass, keep the CLS token of
the last hidden state.

Model: facebook/dinov2-small (~22M parameters; Oquab et al., 2024).
Runs on CPU in under a minute for the 600 images.
"""

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

MODEL = "facebook/dinov2-small"
BATCH_SIZE = 32


def embed_images(model, processor, images, batch_size=BATCH_SIZE):
    """CLS-token embeddings, shape (len(images), hidden_size)."""
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = processor(images=list(images[start:start + batch_size]),
                              return_tensors="pt")
            hidden = model(**batch).last_hidden_state   # (b, 1+patches, 384)
            out.append(hidden[:, 0].numpy())            # CLS token
    return np.concatenate(out).astype(np.float32)


def main():
    images = np.load("blood_cells.npz")["images"]
    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL)
    emb = embed_images(model, processor, images)
    np.save("image_embeddings.npy", emb)
    print(f"wrote image_embeddings.npy: {emb.shape} ({emb.nbytes // 1024} KB)")


if __name__ == "__main__":
    main()
