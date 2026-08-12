"""
Generate blood_cells.npz -- the image dataset for the Aug 14 afternoon
computer-vision session.

Attendees do not need to run this file: the .npz it produces (5.2 MB) is
already in the repository.  Running it downloads the full 156 MB BloodMNIST
archive, which is exactly the kind of conference-wifi load the workshop
avoids.

Source: BloodMNIST, part of MedMNIST v2 (Yang et al., 2023, Scientific
Data), itself derived from the peripheral-blood-cell dataset of Acevedo et
al. (2020).  License: CC BY 4.0.  Images are 64x64 RGB microscope images
of individual blood cells, 8 cell types.

The subset keeps a balanced, seeded sample: 75 images per class, 600 total.

Load with:
    z = np.load('blood_cells.npz')
    images, labels = z['images'], z['labels']   # (600,64,64,3) uint8, (600,)
"""

import urllib.request

import numpy as np

URL = "https://zenodo.org/records/10519652/files/bloodmnist_64.npz"

CLASS_NAMES = ["basophil", "eosinophil", "erythroblast", "immature granulocyte",
               "lymphocyte", "monocyte", "neutrophil", "platelet"]


def main():
    print("downloading", URL, "(~156 MB)...")
    urllib.request.urlretrieve(URL, "bloodmnist_64.npz")
    z = np.load("bloodmnist_64.npz")
    X_all, y_all = z["train_images"], z["train_labels"].ravel()

    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.where(y_all == c)[0], 75, replace=False)
                          for c in range(8)])
    rng.shuffle(idx)
    np.savez_compressed("blood_cells.npz", images=X_all[idx], labels=y_all[idx])
    print("wrote blood_cells.npz: 600 images, 75 per class")


if __name__ == "__main__":
    main()
